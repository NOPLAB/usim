"""Resource and phase ownership for a grouped Genesis runtime session."""

from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from types import TracebackType
from typing import Any, TYPE_CHECKING

import numpy as np

from usim.capability import (
    CapabilityContext,
    CapabilityRequest,
    Phase,
    SupportReport,
    assess_context,
)
from usim.groups import AgentGroup, ExecutionGroup
from usim.model import ModelError
from usim.scene import SceneBindings, SceneEditor
from usim.specs import ControlSpec, ExecutionSpec, SceneSpec

from .access import ControlAccess, StateAccess
from .construction import BuiltScene, build, prepare
from .edits import Edits
from .execution import Episode
from .native import BoundEntity

if TYPE_CHECKING:
    from .contracts import Provider as Backend


class Session:
    """Own one mutable scene, its imports, and the episode safe-point state."""

    def __init__(self, backend: 'Backend', spec: SceneSpec, options: ExecutionSpec):
        self.backend = backend
        self.models = backend.models
        self.options = options
        self._phase = Phase.BUILDING
        self._resources = ExitStack()
        self.spec = spec
        self.agents = AgentGroup(spec.agents)
        self.steps = np.zeros(options.n_envs, dtype=np.int64)
        self.state = StateAccess(self)
        self.control = ControlAccess(self)
        self._episode = Episode(self)
        self.execution = ExecutionGroup(episode=self._episode)
        succeeded = False
        try:
            prepared = prepare(spec, self.models, self._resources)
            self._built: BuiltScene = build(prepared, self.options, self._resources)
            self._episode.clear_commands()
            self._phase = Phase.READY
            edits = Edits(self)
            self.scene = SceneEditor(
                tuple(self.entities),
                SceneBindings(
                    edits.assess, edits.apply, lambda: self.phase, self.fail, options.n_envs
                ),
            )
            succeeded = True
        finally:
            if not succeeded:
                self.fail()
                self._resources.close()

    @property
    def phase(self) -> Phase:
        return self._phase

    def assess(self, request: CapabilityRequest) -> SupportReport:
        return assess_context(
            self.backend,
            request,
            CapabilityContext(self.phase, self.options.n_envs, self.options.mode),
        )

    @property
    def entities(self) -> Mapping[str, BoundEntity]:
        return self._built.entities

    @property
    def controllers(self) -> Mapping[str, ControlSpec]:
        return self._built.controllers

    @property
    def native(self) -> Any:
        """The untyped optional SDK instance; grouped public access uses typed facets."""
        return self._built.native

    def invalidate_sensors(self) -> None:
        for bindings in (self._built.cameras, self._built.lidars):
            for sensors in bindings.values():
                for sensor in sensors:
                    sensor.invalidate()

    def require_usable(self) -> None:
        if self.phase not in {Phase.READY, Phase.PAUSED}:
            raise ModelError(f'Genesis operation requires a safe point, not {self.phase.value}')

    def primary(self) -> str:
        if len(self.controllers) == 1:
            return next(iter(self.controllers))
        if not self.controllers and len(self.entities) == 1:
            return next(iter(self.entities))
        raise ModelError('Select state.for_entity(name) or control.for_entity(name) explicitly')

    @contextmanager
    def mutation(self) -> Iterator[None]:
        succeeded = False
        try:
            yield
            succeeded = True
        finally:
            if not succeeded:
                self.fail()

    def fail(self) -> None:
        self._phase = Phase.FAILED

    def rebuild(self, spec: SceneSpec) -> None:
        """Build a replacement before releasing the old scene; start a new episode."""
        with ExitStack() as resources:
            prepared = prepare(spec, self.models, resources)
            built = build(prepared, self.options, resources)
            previous = self._resources
            self.spec = spec
            self._built = built
            self.agents = AgentGroup(spec.agents)
            self._episode.clear_commands()
            self.steps[:] = 0
            self._resources = resources.pop_all()
            previous.close()
            self._phase = Phase.READY

    def close(self) -> None:
        if self.phase is not Phase.CLOSED:
            try:
                self._resources.close()
            finally:
                self._phase = Phase.CLOSED

    def __enter__(self) -> 'Session':
        self.require_usable()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
