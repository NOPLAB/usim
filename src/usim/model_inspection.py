"""Dependency-free fingerprints and conservative import evidence."""

from collections.abc import Sequence
from hashlib import sha256

from usim.model import ImportReport, ImportRequirements, ModelError, ModelSource


def fingerprint(source: ModelSource, requirements: ImportRequirements) -> str:
    """Bind a request to asset bytes, selection, and declared resource trees."""
    digest = sha256(repr((source, requirements)).encode())
    paths = {source.path}
    for root in source.resources:
        if not root.exists():
            raise ModelError(f'Missing model resource: {root}')
        paths.add(root)
        if root.is_dir():
            paths.update(root.rglob('*'))
    for path in sorted(paths):
        digest.update(str(path).encode())
        if path.is_file():
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
        elif path.is_dir():
            digest.update(b'directory')
        else:
            raise ModelError(f'Missing model asset: {path}')
    return digest.hexdigest()


def require_fidelity(report: ImportReport, requirements: ImportRequirements) -> None:
    """Reject unproven preservation and losses without explicit permission."""
    proven = report.preserved - report.lost - report.unknown - report.changed
    missing = requirements.preserve - proven
    forbidden_loss = report.lost - requirements.allow_loss
    reasons: list[str] = []
    if missing:
        reasons.append(f'Preservation is not proven for: {", ".join(sorted(missing))}')
    if forbidden_loss:
        reasons.append(f'Explicit allow_loss is required for: {", ".join(sorted(forbidden_loss))}')
    if reasons:
        raise ModelError('; '.join(reasons))


def combine_reports(reports: Sequence[ImportReport], route: tuple[str, ...]) -> ImportReport:
    """Preservation requires evidence at every stage, not a union of claims."""
    preserved = set(reports[0].preserved)
    changed: set[str] = set()
    lost: set[str] = set()
    unknown: set[str] = set()
    features: set[str] = set()
    for report in reports:
        preserved.intersection_update(report.preserved)
        changed.update(report.changed)
        lost.update(report.lost)
        unknown.update(report.unknown)
        features.update(report.preserved | report.changed | report.lost | report.unknown)
    unknown.update(features - preserved - changed - lost)
    preserved.difference_update(changed | lost | unknown)
    return ImportReport(
        route=route,
        preserved=frozenset(preserved),
        changed=frozenset(changed),
        lost=frozenset(lost),
        unknown=frozenset(unknown),
        joint_mapping=compose_mapping(tuple(report.joint_mapping for report in reports)),
        link_mapping=compose_mapping(tuple(report.link_mapping for report in reports)),
        notes=tuple(note for report in reports for note in report.notes),
    )


def compose_mapping(
    mappings: tuple[tuple[tuple[str, str], ...], ...],
) -> tuple[tuple[str, str], ...]:
    """Only emit mappings traced through every stage's explicit evidence."""
    result = mappings[0]
    for mapping in mappings[1:]:
        following = dict(mapping)
        result = tuple((old, following[mid]) for old, mid in result if mid in following)
    return result
