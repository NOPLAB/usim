"""Internal session views keep native facets independent of their owner class."""

from collections.abc import Mapping
from contextlib import AbstractContextManager
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from usim.capability import BackendDescription, CapabilityRequest, Phase, SupportReport
from usim.facets import ControlCollection, StateCollection
from usim.groups import AgentGroup
from usim.model_io import ModelIO
from usim.scene import SceneEditor
from usim.specs import ControlSpec, ExecutionSpec, SceneSpec

from .construction import BuiltScene
from .models import ParsedAsset
from .native import BoundEntity


class Provider(Protocol):
    models: ModelIO[ParsedAsset]

    def describe(self) -> BackendDescription: ...

    def assess(self, request: CapabilityRequest) -> SupportReport: ...


class NativeScene(Protocol):
    def step(self) -> None: ...

    def reset(self, *, envs_idx: NDArray[np.generic]) -> None: ...


class SessionView(Protocol):
    _phase: Phase

    @property
    def backend(self) -> Provider: ...

    @property
    def models(self) -> ModelIO[ParsedAsset]: ...

    @property
    def options(self) -> ExecutionSpec: ...

    @property
    def spec(self) -> SceneSpec: ...

    @property
    def agents(self) -> AgentGroup: ...

    @property
    def steps(self) -> NDArray[np.int64]: ...

    @property
    def state(self) -> StateCollection: ...

    @property
    def control(self) -> ControlCollection: ...

    @property
    def scene(self) -> SceneEditor: ...

    @property
    def _built(self) -> BuiltScene: ...

    @property
    def phase(self) -> Phase: ...

    @property
    def entities(self) -> Mapping[str, BoundEntity]: ...

    @property
    def controllers(self) -> Mapping[str, ControlSpec]: ...

    @property
    def native(self) -> NativeScene: ...

    def require_usable(self) -> None: ...

    def primary(self) -> str: ...

    def mutation(self) -> AbstractContextManager[None]: ...

    def fail(self) -> None: ...

    def rebuild(self, spec: SceneSpec) -> None: ...

    def invalidate_sensors(self) -> None: ...
