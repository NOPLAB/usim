"""Internal typed views separate native facets from session construction."""

from collections.abc import Mapping
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from usim.capability import CapabilityRequest, Phase, SupportReport
from usim.facets import SensorLayout, StateCollection
from usim.runtime import ModelAccess
from usim.scene import SceneEditor
from usim.specs import AgentSpec, EntitySpec, ExecutionSpec

from .native_runtime import ImportRuntime
from .scene_native import NativeScene


class Provider(Protocol):
    def assess(self, request: CapabilityRequest) -> SupportReport: ...

    def resolve_agent(self, agent: AgentSpec) -> AgentSpec: ...


class SensorReadout(Protocol):
    @property
    def rgb(self) -> NDArray[np.generic] | None: ...

    @property
    def depth(self) -> NDArray[np.generic] | None: ...

    @property
    def frames(self) -> Mapping[str, NDArray[np.generic]]: ...

    @property
    def timestamps(self) -> Mapping[str, float]: ...


class SensorAccess(Protocol):
    def layouts(self, name: str) -> tuple[SensorLayout, ...]: ...

    def observe(self, name: str) -> SensorReadout: ...

    def invalidate(self) -> None: ...


class SessionView(Protocol):
    _phase: Phase
    _agents: dict[str, AgentSpec]
    _specs: dict[str, EntitySpec]
    scene: SceneEditor

    @property
    def phase(self) -> Phase: ...

    @property
    def backend(self) -> Provider: ...

    @property
    def models(self) -> ModelAccess: ...

    @property
    def state(self) -> StateCollection: ...

    @property
    def _execution(self) -> ExecutionSpec: ...

    @property
    def _runtime(self) -> ImportRuntime: ...

    @property
    def _native(self) -> NativeScene: ...

    @property
    def _sensors(self) -> SensorAccess: ...

    def require_ready(self) -> None: ...

    def primary_entity(self) -> str: ...

    def rebuild(self) -> None: ...
