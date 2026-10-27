"""Explicit native loaders, directed transports, and owned import artifacts."""

from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from types import TracebackType
from typing import Final, Generic, TypeVar
from uuid import UUID, uuid4
from xml.etree.ElementTree import ParseError

from usim.capability import SupportReport, SupportStatus
from usim.model import (
    FormatLoader,
    ImportContext,
    ImportReport,
    ImportRequirements,
    ImportedModel,
    ModelConverter,
    ModelError,
    ModelFormat,
    ModelSource,
)
from usim.model_inspection import combine_reports, fingerprint, require_fidelity

T = TypeVar('T')
DEFAULT_REQUIREMENTS: Final = ImportRequirements()


@dataclass(frozen=True, slots=True)
class ImportPlan:
    """A single route bound to one request, asset snapshot, and composition."""

    source: ModelSource
    requirements: ImportRequirements
    native_format: ModelFormat
    route: tuple[str, ...]
    cost: float
    report: ImportReport
    support: SupportReport
    steps: tuple[int, ...] = field(repr=False)
    loader_index: int = field(repr=False)
    owner: UUID = field(repr=False)
    composition: str = field(repr=False)
    fingerprint: str = field(repr=False)


class OwnedModel(Generic[T]):
    """Hold imported values and artifacts until explicit, idempotent close."""

    def __init__(self, imported: ImportedModel[T], resources: ExitStack) -> None:
        self.value: T = imported.value
        self.source: ModelSource = imported.source
        self.report: ImportReport = imported.report
        self._resources: ExitStack = resources
        self.closed: bool = False

    def close(self) -> None:
        if not self.closed:
            self._resources.close()
            self.closed = True

    def __enter__(self) -> 'OwnedModel[T]':
        if self.closed:
            raise ModelError('Imported model is already closed')
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class ModelIO(Generic[T]):
    """Compose only explicitly supplied providers; never discover engines."""

    _loaders: tuple[FormatLoader[T], ...]
    _transports: tuple[ModelConverter, ...] = ()
    _owner: UUID = field(default_factory=uuid4, repr=False)

    @classmethod
    def compose(
        cls,
        loaders: Sequence[FormatLoader[T]],
        transports: Sequence[ModelConverter] = (),
    ) -> 'ModelIO[T]':
        result = cls(tuple(loaders), tuple(transports))
        _ = result._composition()
        return result

    def _composition(self) -> str:
        if len({loader.format for loader in self._loaders}) != len(self._loaders):
            raise ModelError('Supply exactly one native loader per format')
        if len({edge.name for edge in self._transports}) != len(self._transports):
            raise ModelError('Transport names must be unique within a composition')
        for edge in self._transports:
            if not isfinite(edge.cost) or edge.cost < 0:
                raise ModelError(f'Transport {edge.name} requires a finite nonnegative cost')
        loaders = tuple((id(loader), loader.format) for loader in self._loaders)
        transports = tuple(
            (id(edge), edge.name, edge.source_format, edge.target_format, edge.cost)
            for edge in self._transports
        )
        return repr((loaders, transports))

    @property
    def native_formats(self) -> frozenset[ModelFormat]:
        return frozenset(loader.format for loader in self._loaders)

    @property
    def formats(self) -> frozenset[ModelFormat]:
        """Potentially reachable formats, not a promise about every asset."""
        reachable = set(self.native_formats)
        while True:
            previous = reachable.copy()
            reachable.update(
                edge.source_format for edge in self._transports if edge.target_format in reachable
            )
            if previous == reachable:
                return frozenset(reachable)

    def _routes(self, start: ModelFormat) -> list[tuple[float, tuple[int, ...], int]]:
        routes: list[tuple[float, tuple[int, ...], int]] = []
        pending: list[tuple[ModelFormat, tuple[int, ...], frozenset[ModelFormat], float]] = [
            (start, (), frozenset({start}), 0.0)
        ]
        while pending:
            current, steps, visited, cost = pending.pop()
            for index, loader in enumerate(self._loaders):
                if loader.format == current:
                    routes.append((cost, steps, index))
            for index, edge in enumerate(self._transports):
                if edge.source_format == current and edge.target_format not in visited:
                    total = cost + edge.cost
                    if not isfinite(total):
                        raise ModelError('Transport route cost overflowed')
                    pending.append(
                        (
                            edge.target_format,
                            steps + (index,),
                            visited | {edge.target_format},
                            total,
                        )
                    )
        return sorted(routes, key=lambda item: (item[0], len(item[1]), item[1], item[2]))

    def plan(
        self, source: ModelSource, requirements: ImportRequirements = DEFAULT_REQUIREMENTS
    ) -> ImportPlan:
        composition = self._composition()
        snapshot = fingerprint(source, requirements)
        reasons: list[str] = []
        for cost, steps, loader_index in self._routes(source.format):
            loader = self._loaders[loader_index]
            route = tuple(self._transports[index].name for index in steps)
            route += (f'load:{loader.format.value}',)
            try:
                if steps:
                    assessment = self._transports[steps[0]].assess(source, requirements)
                    if not assessment.supported:
                        raise ModelError(
                            '; '.join(assessment.reasons) or 'Transport rejected source'
                        )
                    report = assessment.report
                else:
                    report = loader.assess(source, requirements)
                require_fidelity(report, requirements)
            except (ModelError, OSError, ParseError, ImportError) as error:
                reasons.append(f'{" -> ".join(route)}: {error}')
                continue
            if fingerprint(source, requirements) != snapshot or self._composition() != composition:
                raise ModelError('Source or providers changed while planning; create a new plan')
            support = SupportReport(
                SupportStatus.CONDITIONAL if steps else SupportStatus.SUPPORTED,
                ('Intermediate assets and native loader must pass assessment during load',)
                if steps
                else (),
                scope='model',
            )
            if steps:
                report = replace(
                    report,
                    preserved=frozenset(),
                    unknown=report.unknown | report.preserved | requirements.preserve,
                )
            return ImportPlan(
                source,
                requirements,
                loader.format,
                route,
                cost,
                combine_reports((report,), route),
                support,
                steps,
                loader_index,
                self._owner,
                composition,
                snapshot,
            )
        detail = '; '.join(reasons) or (
            f'No directed route from {source.format.value} to a supplied native loader; '
            'supply a matching loader or a directed transport'
        )
        raise ModelError(detail)

    def assess(
        self, source: ModelSource, requirements: ImportRequirements = DEFAULT_REQUIREMENTS
    ) -> SupportReport:
        try:
            return self.plan(source, requirements).support
        except (ModelError, OSError, ParseError, ImportError) as error:
            return SupportReport(SupportStatus.UNSUPPORTED, (str(error),), scope='model')

    def load(self, plan: ImportPlan, *, context: ImportContext | None = None) -> OwnedModel[T]:
        """Execute precisely the planned route, validating actual content at each boundary."""
        if (
            plan.owner != self._owner
            or plan.composition != self._composition()
            or plan.fingerprint != fingerprint(plan.source, plan.requirements)
        ):
            raise ModelError('Import plan is stale or belongs to another composition; plan again')
        if context is not None and context.requirements != plan.requirements:
            raise ModelError('Import context requirements differ from the plan')
        with ExitStack() as resources:
            directory = resources.enter_context(
                TemporaryDirectory(
                    prefix='usim-import-', dir=context.directory if context else None
                )
            )
            active_context = ImportContext(Path(directory), plan.requirements)
            source = plan.source
            reports: list[ImportReport] = []
            for index in plan.steps:
                edge = self._transports[index]
                assessment = edge.assess(source, plan.requirements)
                if not assessment.supported:
                    raise ModelError(
                        '; '.join(assessment.reasons) or f'{edge.name} rejected source'
                    )
                require_fidelity(assessment.report, plan.requirements)
                reports.append(assessment.report)
                source = edge.convert(source, active_context)
                if source.format != edge.target_format:
                    raise ModelError(f'{edge.name} returned the wrong target format')
                _ = fingerprint(source, plan.requirements)
            loader = self._loaders[plan.loader_index]
            report = loader.assess(source, plan.requirements)
            require_fidelity(report, plan.requirements)
            imported = loader.load(source, active_context)
            if imported.source != source:
                raise ModelError(
                    'Native loader returned a different source than the planned input'
                )
            require_fidelity(imported.report, plan.requirements)
            final = loader.assess(imported.source, plan.requirements)
            require_fidelity(final, plan.requirements)
            native_report = combine_reports((report, imported.report, final), ())
            reports.append(
                replace(
                    native_report,
                    joint_mapping=imported.report.joint_mapping,
                    link_mapping=imported.report.link_mapping,
                )
            )
            combined = combine_reports(reports, plan.route)
            return OwnedModel(replace(imported, report=combined), resources.pop_all())
