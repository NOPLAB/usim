"""Small state, control, agent and execution capabilities for grouped runtimes."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Protocol, TYPE_CHECKING

from usim.model import ModelError
from usim.specs import AgentSpec
from usim.types import Observation, StepResult

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class JointLayout:
    name: str
    kind: str
    qpos_range: tuple[int, int]
    qvel_range: tuple[int, int]
    unit: str = 'rad'


@dataclass(frozen=True, slots=True)
class SensorLayout:
    name: str
    kind: str
    shape: tuple[int, ...]
    dtype: str
    unit: str
    frame: str


@dataclass(frozen=True, slots=True)
class StateSchema:
    entity: str
    joints: tuple[JointLayout, ...] = ()
    sensors: tuple[SensorLayout, ...] = ()
    n_envs: int = 1
    batched: bool = True
    quaternion_order: str = 'xyzw'


@dataclass(frozen=True, slots=True)
class ControlTargets:
    """Named physical targets; sequences explicitly represent environment batches."""

    positions: Mapping[str, float | Sequence[float]] = field(
        default_factory=dict[str, float | Sequence[float]]
    )
    velocities: Mapping[str, float | Sequence[float]] = field(
        default_factory=dict[str, float | Sequence[float]]
    )
    efforts: Mapping[str, float | Sequence[float]] = field(
        default_factory=dict[str, float | Sequence[float]]
    )

    def __post_init__(self) -> None:
        names = tuple(self.positions) + tuple(self.velocities) + tuple(self.efforts)
        if not names or len(set(names)) != len(names) or any(not name.strip() for name in names):
            raise ModelError('Control targets require nonempty, disjoint joint names')
        for attribute, group in (
            ('positions', self.positions),
            ('velocities', self.velocities),
            ('efforts', self.efforts),
        ):
            normalized: dict[str, float | tuple[float, ...]] = {}
            for name, value in group.items():
                match value:
                    case float() | int():
                        values = (float(value),)
                        normalized[name] = float(value)
                    case _:
                        values = tuple(float(component) for component in value)
                        normalized[name] = values
                if not values:
                    raise ModelError('A control target batch must not be empty')
                if not all(isfinite(component) for component in values):
                    raise ModelError('Control targets must be finite')
            object.__setattr__(self, attribute, MappingProxyType(normalized))


class JointStateReader(Protocol):
    def get_qpos(self) -> 'NDArray[np.generic]': ...

    def get_qvel(self) -> 'NDArray[np.generic]': ...


class ObservationReader(Protocol):
    def get_observation(self) -> Observation: ...


class StateAccess(JointStateReader, ObservationReader, Protocol):
    @property
    def schema(self) -> StateSchema: ...


class StateCollection(StateAccess, Protocol):
    def for_entity(self, name: str) -> StateAccess: ...


class JointPositionControl(Protocol):
    def set_positions(self, targets: Mapping[str, float | Sequence[float]]) -> None: ...


class JointVelocityControl(Protocol):
    def set_velocities(self, targets: Mapping[str, float | Sequence[float]]) -> None: ...


class EffortControl(Protocol):
    def set_efforts(self, targets: Mapping[str, float | Sequence[float]]) -> None: ...


class JointControl(Protocol):
    """Applying targets must not advance physics."""

    def apply(self, targets: ControlTargets) -> None: ...


class ControlCollection(JointControl, Protocol):
    def for_entity(self, name: str) -> JointControl: ...


class AgentAccess(Protocol):
    def describe(self, name: str) -> AgentSpec: ...


class EpisodeDriver(Protocol):
    def reset(self, seed: int | None = None) -> tuple[Observation, Mapping[str, float]]: ...

    def step(self) -> StepResult: ...


class ContinuousDriver(Protocol):
    def run(self) -> None: ...

    def stop(self) -> None: ...
