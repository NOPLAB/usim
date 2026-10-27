"""Grouped Isaac session: one Kit runtime and one shared native world."""

from __future__ import annotations

from types import TracebackType
from typing import TYPE_CHECKING

from usim.capability import (
    CapabilityContext,
    CapabilityRequest,
    Phase,
    SupportReport,
    assess_context,
)
from usim.groups import ExecutionGroup
from usim.model import ImportRequirements, ModelError, ModelSource
from usim.scene import SceneBindings, SceneEditor
from usim.specs import ControlSpec, EntitySpec, ExecutionSpec, SceneSpec
from usim_isaacsim.access import Agents, Controls, States
from usim_isaacsim.editing import Edits
from usim_isaacsim.execution import ContinuousExecution, EpisodeExecution
from usim_isaacsim.native_runtime import ImportRuntime
from usim_isaacsim.scene_native import NativeScene
from usim_isaacsim.sensors import Sensors

if TYPE_CHECKING:
    from usim_isaacsim.contracts import Provider as Backend


class Session:
    """Mutable scene owner; bound entity facets fail after topology generation changes."""

    def __init__(self, backend: Backend, spec: SceneSpec, execution: ExecutionSpec) -> None:
        self.backend = backend
        self._execution = execution
        self._phase = Phase.BUILDING
        self._runtime = ImportRuntime(
            headless=execution.render_mode != 'human',
            rendering=any(
                sensor.kind != 'lidar' for agent in spec.agents for sensor in agent.sensors
            ),
        )
        self.models = self._runtime.models
        self._native = NativeScene(self._runtime, execution)
        self._agents = {agent.name: backend.resolve_agent(agent) for agent in spec.agents}
        self._specs = {entity.name: entity for entity in spec.entities}
        for agent in self._agents.values():
            if not isinstance(agent.model, ModelSource):
                raise ModelError('An agent recipe must resolve to a model source before import')
            self._specs[agent.name] = EntitySpec(
                agent.name, agent.model, agent.pose, agent.requirements
            )
        if '_usim_world' in self._specs:
            raise ModelError('_usim_world is reserved for the scene world')
        self._world = (
            EntitySpec(
                '_usim_world',
                spec.world,
                requirements=ImportRequirements(role='scene', preserve=frozenset()),
            )
            if spec.world is not None
            else None
        )
        if sum(entity.requirements.role == 'articulation' for entity in self._specs.values()) > 1:
            raise ModelError('Grouped Isaac currently supports one articulation per shared world')
        edits = Edits(self)
        self.scene = SceneEditor(
            tuple(self._specs),
            SceneBindings(edits.assess, edits.apply, lambda: self.phase, self.fail),
        )
        self.state, self.control, self.agents = States(self), Controls(self), Agents(self)
        self._sensors = Sensors(self)
        self._episode = EpisodeExecution(self) if execution.mode == 'episode' else None
        self.execution = ExecutionGroup(
            episode=self._episode,
            continuous=ContinuousExecution(self) if execution.mode == 'continuous' else None,
        )
        succeeded = False
        try:
            self._runtime.start()
            self._build()
            self._phase = Phase.READY
            succeeded = True
        finally:
            if not succeeded:
                self.close()

    @property
    def phase(self) -> Phase:
        return self._phase

    def assess(self, request: CapabilityRequest) -> SupportReport:
        return assess_context(
            self.backend,
            request,
            CapabilityContext(self.phase, self._execution.n_envs, self._execution.mode),
        )

    def require_ready(self) -> None:
        if self._phase not in {Phase.READY, Phase.PAUSED}:
            raise ModelError(f'Isaac operation requires a safe point, not {self.phase.value}')
        self._runtime.require_running()

    def fail(self) -> None:
        self._phase = Phase.FAILED

    def primary_entity(self) -> str:
        self.require_ready()
        names = [
            name
            for name, entry in self._native.entities.items()
            if entry.spec.requirements.role == 'articulation'
        ]
        if len(names) == 1:
            return names[0]
        if len(self._specs) == 1:
            return next(iter(self._specs))
        raise ModelError(
            'Select an entity with state.for_entity(name) or control.for_entity(name)'
        )

    def _build(self) -> None:
        self._native.begin()
        if self._world is not None:
            self._native.add(self._world)
        for name, spec in self._specs.items():
            control = self._agents[name].control if name in self._agents else ControlSpec()
            self._native.add(spec, control)
        self._native.initialize()
        self._sensors.build()

    def rebuild(self) -> None:
        """Reset the shared world and rebind all native views, never just swap a path."""
        self._sensors.clear()
        self._native.dispose()
        self._runtime.resources.close()
        self._build()
        if self._episode is not None:
            self._episode.steps = 0

    def close(self) -> None:
        if self._phase == Phase.CLOSED and self._runtime.app is None:
            return
        if self._phase == Phase.RUNNING:
            raise ModelError('Stop continuous execution and let run() return before closing')
        try:
            self._sensors.clear()
            self._native.dispose()
        finally:
            self._phase = Phase.CLOSED
            self._runtime.close()

    def __enter__(self) -> Session:
        self.require_ready()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
