"""Grouped backend discovery without importing optional native dependencies."""

from typing import Final, Protocol, TYPE_CHECKING, runtime_checkable

from usim.capability import BackendDescription, CapabilityRequest, Phase, SupportReport
from usim.facets import AgentAccess, ControlCollection, StateCollection
from usim.groups import ExecutionGroup
from usim.model import ImportContext, ImportRequirements, ModelFormat, ModelLease, ModelSource
from usim.model_io import ImportPlan
from usim.specs import ExecutionSpec, SceneSpec

if TYPE_CHECKING:
    from types import TracebackType

    from usim.scene import SceneEditor

DEFAULT_EXECUTION: Final = ExecutionSpec()


class ModelAccess(Protocol):
    @property
    def native_formats(self) -> frozenset[ModelFormat]: ...

    @property
    def formats(self) -> frozenset[ModelFormat]: ...

    def plan(self, source: ModelSource, requirements: ImportRequirements = ...) -> ImportPlan: ...

    def assess(
        self, source: ModelSource, requirements: ImportRequirements = ...
    ) -> SupportReport: ...

    def load(self, plan: ImportPlan, *, context: ImportContext | None = None) -> ModelLease: ...


@runtime_checkable
class RuntimeSession(Protocol):
    @property
    def phase(self) -> Phase: ...

    def assess(self, request: CapabilityRequest) -> SupportReport: ...

    @property
    def scene(self) -> 'SceneEditor': ...

    @property
    def models(self) -> ModelAccess: ...

    @property
    def state(self) -> StateCollection: ...

    @property
    def control(self) -> ControlCollection: ...

    @property
    def agents(self) -> AgentAccess: ...

    @property
    def execution(self) -> ExecutionGroup: ...

    def close(self) -> None: ...

    def __enter__(self) -> 'RuntimeSession': ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: 'TracebackType | None',
    ) -> None: ...


@runtime_checkable
class BackendProvider(Protocol):
    @property
    def models(self) -> ModelAccess: ...

    def describe(self) -> BackendDescription: ...

    def assess(self, request: CapabilityRequest) -> SupportReport: ...

    def open(
        self, scene: SceneSpec, execution: ExecutionSpec = DEFAULT_EXECUTION
    ) -> RuntimeSession: ...


class BackendFactory(Protocol):
    def __call__(self) -> BackendProvider | None: ...


class BackendEntryPoint(Protocol):
    def load(self) -> BackendFactory: ...


def list_backends() -> list[str]:
    """List installed grouped providers using metadata only."""
    from importlib.metadata import entry_points

    return sorted(entry.name for entry in entry_points(group='usim.backends'))


def _create_registered_backend(selected: BackendEntryPoint) -> BackendProvider:
    """Validate a third-party entry point at the dynamic import boundary."""
    from usim.model import ModelError

    backend = selected.load()()
    if not isinstance(backend, BackendProvider):
        raise ModelError('Entry point did not construct a valid backend provider')
    return backend


def create_backend(name: str) -> BackendProvider:
    """Load only the selected backend's grouped implementation, not its engine."""
    from importlib.metadata import entry_points

    providers = {entry.name: entry for entry in entry_points(group='usim.backends')}
    if name in providers:
        return _create_registered_backend(providers[name])
    from usim.model import ModelError

    raise ModelError(f'Unknown backend: {name}; install its usim engine distribution')
