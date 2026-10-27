"""Dependency-free request-specific capability contracts."""

from dataclasses import dataclass, replace
from enum import Enum
from typing import NoReturn, Protocol


def assert_never(value: NoReturn) -> NoReturn:
    """Exhaustive-match guard available throughout the supported Python range."""
    raise AssertionError(f'Unexpected variant: {value!r}')


class SupportStatus(str, Enum):
    SUPPORTED = 'supported'
    CONDITIONAL = 'conditional'
    UNSUPPORTED = 'unsupported'
    UNKNOWN = 'unknown'


class Phase(str, Enum):
    BUILDING = 'building'
    READY = 'ready'
    RUNNING = 'running'
    PAUSED = 'paused'
    CLOSED = 'closed'
    FAILED = 'failed'


@dataclass(frozen=True, slots=True)
class CapabilityRequest:
    operation: str
    phase: Phase = Phase.BUILDING
    require_same_episode: bool = True
    n_envs: int = 1


@dataclass(frozen=True, slots=True)
class SupportReport:
    status: SupportStatus
    reasons: tuple[str, ...] = ()
    required_phase: Phase | None = None
    effects: tuple[str, ...] = ()
    scope: str = 'scene'

    @property
    def supported(self) -> bool:
        return self.status is SupportStatus.SUPPORTED

    def require(self) -> None:
        if not self.supported:
            raise UnsupportedOperation(self)


class UnsupportedOperation(RuntimeError):
    """An unsupported request with machine-readable assessment attached."""

    def __init__(self, report: SupportReport):
        self.report: SupportReport = report
        super().__init__('; '.join(report.reasons) or report.status.value)


class CapabilityProvider(Protocol):
    def assess(self, request: CapabilityRequest) -> SupportReport: ...


@dataclass(frozen=True, slots=True)
class CapabilityContext:
    phase: Phase
    n_envs: int
    execution_mode: str


def assess_context(
    provider: CapabilityProvider, request: CapabilityRequest, context: CapabilityContext
) -> SupportReport:
    """Bind a live assessment to actual state, rather than caller-supplied phase."""
    if request.operation in {'execution.episode', 'execution.continuous'}:
        if request.operation != 'execution.' + context.execution_mode:
            return SupportReport(
                SupportStatus.UNSUPPORTED, ('This session uses another execution mode',)
            )
    return provider.assess(replace(request, phase=context.phase, n_envs=context.n_envs))


@dataclass(frozen=True, slots=True)
class BackendDescription:
    name: str
    operations: frozenset[str]
    model_formats: frozenset[str]
    execution_modes: frozenset[str]
    registration_required: bool = False
