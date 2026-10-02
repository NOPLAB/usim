"""Fetch the fixed audited robot descriptions, without importing simulator code."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any

MANIFEST = Path(__file__).with_name('sources.json')
DEFAULT_OUT = Path(__file__).resolve().parents[2] / 'assets' / 'robots'


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text(encoding='utf-8'))


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def member_path(name: str, root: str) -> PurePosixPath:
    """Reject POSIX and Windows escapes before considering the inclusion filter."""
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or '\\' in name
        or any(part in {'..', '.'} or ':' in part for part in name.split('/'))
        or not path.parts
        or path.parts[0] != root
    ):
        raise ValueError(f'unsafe archive path: {name}')
    return PurePosixPath(*path.parts[1:])


def included(path: str, source: dict[str, Any]) -> bool:
    if path in source['license_files'] or path in source.get('header_files', []):
        return True
    # Preserve root notices as well as files in a selected description tree.
    if path in {'NOTICE', 'NOTICE.txt', 'NOTICE.md'}:
        return True
    return any(
        path.startswith(rule) if rule.endswith('/') else path == rule for rule in source['include']
    )


def extract_archive(archive: Path, destination: Path, source: dict[str, Any]) -> None:
    """Preflight all members; never extract links, special files, or skipped robots."""
    with tarfile.open(archive, 'r:gz') as tar:
        selected = []
        seen = set()
        for member in tar.getmembers():
            relative = member_path(member.name, source['archive_root'])
            if not member.isfile() and not member.isdir():
                raise ValueError(f'unsafe archive member type: {member.name}')
            if member.isfile() and included(relative.as_posix(), source):
                if relative in seen:
                    raise ValueError(f'duplicate archive path: {member.name}')
                seen.add(relative)
                target = destination.joinpath(*relative.parts)
                if not target.resolve().is_relative_to(destination.resolve()):
                    raise ValueError(f'archive destination escape: {member.name}')
                selected.append((member, target))
        for member, target in selected:
            target.parent.mkdir(parents=True, exist_ok=True)
            incoming = tar.extractfile(member)
            assert incoming is not None  # Selected members are regular files.
            with incoming, target.open('xb') as outgoing:
                shutil.copyfileobj(incoming, outgoing)
    for relative, expected in source['license_files'].items():
        if checksum(destination / relative) != expected:
            raise ValueError(f'license checksum mismatch: {relative}')
    for relative in source.get('header_files', []):
        text = (destination / relative).read_text(encoding='utf-8')
        if 'Software License Agreement (BSD)' not in text or 'endorse or promote' not in text:
            raise ValueError(f'missing BSD header: {relative}')


def inventory(root: Path) -> dict[str, dict[str, int | str]]:
    files = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'symlink in source directory: {path}')
        if path.is_file() and path.name != 'provenance.json':
            files[path.relative_to(root).as_posix()] = {
                'bytes': path.stat().st_size,
                'sha256': checksum(path),
            }
    return files


def fetch_source(name: str, source: dict[str, Any], out: Path) -> dict[str, Any]:
    cache = out / 'archives'
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / f'{name}-{source["commit"]}.tar.gz'
    if not archive.exists():
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=cache, delete=False) as incoming:
                temporary = Path(incoming.name)
                with urllib.request.urlopen(source['archive_url'], timeout=120) as response:
                    shutil.copyfileobj(response, incoming)
            if checksum(temporary) != source['sha256']:
                raise ValueError(f'archive checksum mismatch: {name}')
            temporary.replace(archive)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    if checksum(archive) != source['sha256'] or archive.stat().st_size != source['archive_bytes']:
        raise ValueError(f'archive checksum/size mismatch: {name}')
    sources = out / 'sources'
    sources.mkdir(parents=True, exist_ok=True)
    destination = sources / f'{name}-{source["commit"]}'
    with tempfile.TemporaryDirectory(dir=sources) as staging:
        staged = Path(staging) / 'source'
        staged.mkdir()
        extract_archive(archive, staged, source)
        files = inventory(staged)
        evidence = dict(
            source,
            files=files,
            extracted_bytes=sum((staged / name).stat().st_size for name in files),
        )
        serialized = json.dumps(evidence, indent=2, sort_keys=True) + '\n'
        (staged / 'provenance.json').write_text(serialized, encoding='utf-8')
        if destination.exists():
            if (
                destination.is_symlink()
                or inventory(destination) != files
                or (destination / 'provenance.json').read_text(encoding='utf-8') != serialized
            ):
                raise ValueError(f'existing source differs from pinned archive: {destination}')
        else:
            staged.rename(destination)
    return dict(evidence, path=destination.relative_to(out).as_posix())


def main(argv: list[str] | None = None) -> None:
    manifest = load_manifest()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('robots', nargs='*', help='Robot IDs; omitted means all six.')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        '--list', action='store_true', help='Print pinned metadata without downloads.'
    )
    args = parser.parse_args(argv)
    unknown = set(args.robots) - manifest['robots'].keys()
    if unknown:
        parser.error(f'unknown robot IDs: {", ".join(sorted(unknown))}')
    if args.list:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    robots = {name: manifest['robots'][name] for name in sorted(args.robots or manifest['robots'])}
    source_names = {robot['source'] for robot in robots.values()}
    source_names.update(
        dependency['source']
        for robot in robots.values()
        for dependency in robot.get('dependencies', [])
    )
    source_names = sorted(source_names)
    out = args.out.resolve()
    evidence = {name: fetch_source(name, manifest['sources'][name], out) for name in source_names}
    provenance = dict(
        schema_version=manifest['schema_version'],
        audit_date=manifest['audit_date'],
        import_verified=False,
        sources=evidence,
        robots=robots,
    )
    (out / 'provenance.json').write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
