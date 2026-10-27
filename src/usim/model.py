"""Portable model assets and the contracts used by loaders and transports."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Generic, Protocol, TypeVar, TYPE_CHECKING
from xml.etree import ElementTree

if TYPE_CHECKING:
    from types import TracebackType


class ModelFormat(str, Enum):
    URDF = 'urdf'
    MJCF = 'mjcf'
    USD = 'usd'
    SDF = 'sdf'


class ModelError(ValueError):
    """A structured model boundary error with normal exception traceback behavior."""

    def __init__(self, detail: str):
        self.detail: str = detail
        super().__init__(detail)


@dataclass(frozen=True, slots=True)
class ModelSource:
    path: Path
    format: ModelFormat
    selector: str | None = None
    resources: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, 'format', ModelFormat(self.format))
        path = Path(self.path).resolve()
        if not path.is_file():
            raise ModelError(f'Missing model: {path}')
        object.__setattr__(self, 'path', path)
        resources = tuple(Path(resource).resolve() for resource in self.resources)
        if any(not resource.exists() for resource in resources):
            raise ModelError('Model resource roots must exist')
        object.__setattr__(self, 'resources', resources)

    @classmethod
    def from_path(cls, path: str | Path, *, selector: str | None = None) -> 'ModelSource':
        source = Path(path).resolve()
        suffix = source.suffix.lower()
        formats = {
            '.urdf': ModelFormat.URDF,
            '.sdf': ModelFormat.SDF,
            '.world': ModelFormat.SDF,
            '.usd': ModelFormat.USD,
            '.usda': ModelFormat.USD,
            '.usdc': ModelFormat.USD,
            '.usdz': ModelFormat.USD,
        }
        if suffix == '.xml':
            tag = ElementTree.parse(source).getroot().tag
            xml_formats = {
                'mujoco': ModelFormat.MJCF,
                'robot': ModelFormat.URDF,
                'sdf': ModelFormat.SDF,
            }
            if tag not in xml_formats:
                raise ModelError(f'Unrecognized model XML root: {tag}')
            return cls(source, xml_formats[tag], selector)
        if suffix not in formats:
            raise ModelError(f'Unrecognized model extension: {suffix}')
        return cls(source, formats[suffix], selector)


@dataclass(frozen=True, slots=True)
class ImportRequirements:
    role: str = 'articulation'
    preserve: frozenset[str] = frozenset({'joint_structure', 'inertia', 'collision'})
    base: str = 'authored'
    allow_loss: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.role not in {'visual', 'rigid_body', 'articulation', 'scene'}:
            raise ModelError(f'Unknown model role: {self.role}')
        if self.base not in {'authored', 'fixed', 'floating'}:
            raise ModelError(f'Unknown base mode: {self.base}')
        if self.preserve & self.allow_loss:
            raise ModelError('A feature cannot be required and allowed to be lost')


@dataclass(frozen=True, slots=True)
class ImportContext:
    directory: Path
    requirements: ImportRequirements = field(default_factory=ImportRequirements)


@dataclass(frozen=True, slots=True)
class ImportReport:
    route: tuple[str, ...] = ()
    preserved: frozenset[str] = frozenset()
    changed: frozenset[str] = frozenset()
    lost: frozenset[str] = frozenset()
    unknown: frozenset[str] = frozenset()
    joint_mapping: tuple[tuple[str, str], ...] = ()
    link_mapping: tuple[tuple[str, str], ...] = ()
    notes: tuple[str, ...] = ()


T = TypeVar('T')


@dataclass(frozen=True, slots=True)
class ImportedModel(Generic[T]):
    source: ModelSource
    value: T
    report: ImportReport = field(default_factory=ImportReport)


class FormatLoader(Protocol[T]):
    @property
    def format(self) -> ModelFormat: ...

    def assess(self, source: ModelSource, requirements: ImportRequirements) -> ImportReport: ...

    def load(self, source: ModelSource, context: ImportContext) -> ImportedModel[T]: ...


@dataclass(frozen=True, slots=True)
class ConversionAssessment:
    supported: bool
    report: ImportReport = field(default_factory=ImportReport)
    reasons: tuple[str, ...] = ()


class ModelConverter(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def source_format(self) -> ModelFormat: ...

    @property
    def target_format(self) -> ModelFormat: ...

    @property
    def cost(self) -> float: ...

    def assess(
        self, source: ModelSource, requirements: ImportRequirements
    ) -> ConversionAssessment: ...

    def convert(self, source: ModelSource, context: ImportContext) -> ModelSource: ...


class ModelLease(Protocol):
    """Portable view of an owned import; native values remain backend-specific."""

    @property
    def source(self) -> ModelSource: ...

    @property
    def report(self) -> ImportReport: ...

    @property
    def closed(self) -> bool: ...

    def close(self) -> None: ...

    def __enter__(self) -> 'ModelLease': ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: 'TracebackType | None',
    ) -> None: ...
