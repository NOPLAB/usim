"""Reusable grouping implementations built from explicitly supplied capabilities."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from usim.capability import SupportReport, SupportStatus, UnsupportedOperation
from usim.facets import ContinuousDriver, ControlTargets, EpisodeDriver, StateSchema
from usim.model import ModelError
from usim.specs import AgentSpec, ControlSpec
from usim.types import Observation, StepResult

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


TargetMap = Mapping[str, float | Sequence[float]]


@dataclass(frozen=True, slots=True)
class ControlBindings:
    validate: Callable[[ControlTargets], None]
    positions: Callable[[TargetMap], None] | None = None
    velocities: Callable[[TargetMap], None] | None = None
    efforts: Callable[[TargetMap], None] | None = None


class JointControlGroup:
    """Compose native target writers; validate the whole request before any write."""

    def __init__(self, spec: ControlSpec, bindings: ControlBindings):
        self._spec: ControlSpec = spec
        self._bindings: ControlBindings = bindings

    def apply(self, targets: ControlTargets) -> None:
        writes = (
            (targets.positions, self._bindings.positions, self._spec.position_joints, 'position'),
            (
                targets.velocities,
                self._bindings.velocities,
                self._spec.velocity_joints,
                'velocity',
            ),
            (targets.efforts, self._bindings.efforts, self._spec.effort_joints, 'effort'),
        )
        partitioned = bool(
            self._spec.position_joints + self._spec.velocity_joints + self._spec.effort_joints
        )
        for values, writer, selected, mode in writes:
            if values and writer is None:
                raise UnsupportedOperation(
                    SupportReport(SupportStatus.UNSUPPORTED, (f'{mode} targets are unsupported',))
                )
            available = selected if partitioned else self._spec.joint_names
            if (partitioned or available) and not set(values) <= set(available):
                raise ModelError(f'Unknown or incompatible {mode} joints: {sorted(values)}')
        self._bindings.validate(targets)
        for values, writer, _, _ in writes:
            if values and writer is not None:
                writer(values)

    def set_positions(self, targets: TargetMap) -> None:
        self.apply(ControlTargets(positions=targets))

    def set_velocities(self, targets: TargetMap) -> None:
        self.apply(ControlTargets(velocities=targets))

    def set_efforts(self, targets: TargetMap) -> None:
        self.apply(ControlTargets(efforts=targets))


@dataclass(frozen=True, slots=True)
class StateBindings:
    observation: Callable[[], Observation]
    qpos: Callable[[], 'NDArray[np.generic]']
    qvel: Callable[[], 'NDArray[np.generic]']


class StateGroup:
    """Group state readers without assuming that observation advances simulation."""

    def __init__(self, schema: StateSchema, bindings: StateBindings):
        self._schema: StateSchema = schema
        self._bindings: StateBindings = bindings

    @property
    def schema(self) -> StateSchema:
        return self._schema

    def get_observation(self) -> Observation:
        return self._bindings.observation()

    def get_qpos(self) -> 'NDArray[np.generic]':
        return self._bindings.qpos()

    def get_qvel(self) -> 'NDArray[np.generic]':
        return self._bindings.qvel()


class AgentGroup:
    """Read the specifications of agents actually bound by the owning session."""

    def __init__(self, agents: tuple[AgentSpec, ...]):
        self._agents: dict[str, AgentSpec] = {agent.name: agent for agent in agents}
        if len(self._agents) != len(agents):
            raise ModelError('Agent names must be unique')

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._agents)

    def describe(self, name: str) -> AgentSpec:
        if name not in self._agents:
            raise ModelError(f'Unknown bound agent: {name}')
        return self._agents[name]


class ExecutionGroup:
    """Expose real execution drivers, never manufacture stepping for a runner."""

    def __init__(
        self, *, episode: EpisodeDriver | None = None, continuous: ContinuousDriver | None = None
    ):
        self._episode: EpisodeDriver | None = episode
        self._continuous: ContinuousDriver | None = continuous

    @property
    def episode(self) -> EpisodeDriver:
        if self._episode is None:
            raise UnsupportedOperation(
                SupportReport(SupportStatus.UNSUPPORTED, ('Episode stepping is unavailable',))
            )
        return self._episode

    @property
    def continuous(self) -> ContinuousDriver:
        if self._continuous is None:
            raise UnsupportedOperation(
                SupportReport(SupportStatus.UNSUPPORTED, ('Continuous execution is unavailable',))
            )
        return self._continuous

    def reset(self, seed: int | None = None) -> tuple[Observation, Mapping[str, float]]:
        return self.episode.reset(seed)

    def step(self) -> StepResult:
        return self.episode.step()

    def run(self) -> None:
        self.continuous.run()

    def stop(self) -> None:
        self.continuous.stop()
