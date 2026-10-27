"""Typed service-facing views, independent of container/session construction."""

from typing import Protocol

from usim.capability import CapabilityRequest, Phase, SupportReport
from usim.facets import ControlCollection, StateAccess
from usim.model import ImportRequirements, ModelSource
from usim.model_io import ModelIO, OwnedModel
from usim.scene import SceneEditor
from usim.specs import AgentSpec, EntitySpec, Pose

from .models import GazeboModel
from .sensors import SensorChannel
from .transport import Json, RequestPort


class Provider(Protocol):
    models: ModelIO[GazeboModel]

    def assess(self, request: CapabilityRequest) -> SupportReport: ...


class ControlsView(ControlCollection, Protocol):
    def disable(self) -> None: ...

    def invalidate(self) -> None: ...

    def discard(self, name: str) -> None: ...


class EntityStateView(StateAccess, Protocol):
    def get_pose(self) -> Pose: ...


class StatesView(Protocol):
    def __getitem__(self, name: str) -> EntityStateView: ...


class SessionView(Protocol):
    phase: Phase
    scene: SceneEditor

    @property
    def control(self) -> ControlsView: ...

    @property
    def state(self) -> StatesView: ...

    @property
    def backend(self) -> Provider: ...

    @property
    def models(self) -> ModelIO[GazeboModel]: ...

    _models: dict[str, OwnedModel[GazeboModel]]
    _agent_specs: dict[str, AgentSpec]
    _entities: dict[str, EntitySpec]
    _sensor_channels: dict[str, tuple[SensorChannel, ...]]
    _port: RequestPort | None
    _services: frozenset[str]
    _spawn_xml: dict[str, str]

    def assess(self, request: CapabilityRequest) -> SupportReport: ...

    def _call(self, service: str, fields: dict[str, Json]) -> dict[str, Json]: ...

    def _fail(self) -> None: ...

    def _load(
        self, source: ModelSource, requirements: ImportRequirements
    ) -> OwnedModel[GazeboModel]: ...

    def _spawn(self, spec: EntitySpec) -> None: ...
