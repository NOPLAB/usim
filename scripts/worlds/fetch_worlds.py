"""Install only the six audited navigation worlds; upstream assets stay untracked."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
from typing import Any
import urllib.request
import zipfile


SOURCES = json.loads(Path(__file__).with_name('sources.json').read_text(encoding='utf-8'))


def verify(data: bytes, expected_hash: str, expected_size: int | None = None) -> bytes:
    """Reject changed bytes before extraction or installation."""
    if expected_size is not None and len(data) != expected_size:
        raise ValueError(f'Size mismatch: expected {expected_size}, got {len(data)}')
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError('SHA-256 mismatch')
    return data


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={'User-Agent': 'usim-world-fetcher/1.0'})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def subset_archive(source: dict[str, Any]) -> bytes:
    """Build a byte-stable ZIP from Willow Garage files, not the whole model database."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_STORED) as archive:
        for name, pin in sorted(source['files'].items()):
            url = (
                f'https://raw.githubusercontent.com/osrf/gazebo_models/{source["revision"]}/{name}'
            )
            data = verify(download(url), pin['sha256'], pin['size'])
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    return buffer.getvalue()


def archive_bytes(source: dict[str, Any], cache: Path) -> bytes:
    """Cache archives by digest and verify even cached downloads on every use."""
    path = cache / f'{source["archive_sha256"]}.zip'
    if path.exists():
        return verify(path.read_bytes(), source['archive_sha256'], source['archive_size'])
    if source.get('archive_kind') == 'deterministic-subset-zip':
        data = subset_archive(source)
    else:
        data = download(source['archive_url'])
    verify(data, source['archive_sha256'], source['archive_size'])
    cache.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def safe_path(name: str) -> PurePosixPath:
    """Reject traversal and Windows-specific path interpretations before selection."""
    path = PurePosixPath(name)
    if (
        not path.parts
        or path.is_absolute()
        or '\\' in name
        or ':' in name
        or '\x00' in name
        or '..' in path.parts
    ):
        raise ValueError(f'Unsafe archive path: {name!r}')
    return path


def install_path(name: str, source: dict[str, Any]) -> Path | None:
    """Map only the audited resource roots into their local model layout."""
    path = safe_path(name)
    root = source['archive_root']
    if root:
        if path.parts[0] != root:
            return None
        path = PurePosixPath(*path.parts[1:])
    relative = path.as_posix()
    if not any(
        relative == allowed or (allowed.endswith('/') and relative.startswith(allowed))
        for allowed in source['include']
    ):
        return None
    if relative.startswith('willowgarage/'):
        return Path('models') / Path(*path.parts)
    prefix = source.get('install_prefix', '')
    return Path(prefix) / Path(*path.parts)


def extract(data: bytes, source: dict[str, Any], destination: Path) -> None:
    """Extract regular files only, with no archive-controlled permissions or links."""
    selected = []
    seen = set()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for info in archive.infolist():
            relative = install_path(info.orig_filename, source)
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                raise ValueError(f'Non-regular archive member: {info.filename}')
            if info.is_dir() or relative is None:
                continue
            # Case-insensitive collision detection also protects Windows installations.
            key = relative.as_posix().casefold()
            if key in seen:
                raise ValueError(f'Duplicate archive member: {info.filename}')
            seen.add(key)
            selected.append((info, relative))
        for info, relative in selected:
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(info))


def finish_install(source: dict[str, Any], destination: Path) -> None:
    """Retain rights evidence and add wrappers for sources shipped as models."""
    if 'license_download_url' in source:
        license_data = verify(download(source['license_download_url']), source['license_sha256'])
        (destination / source['license_path']).write_bytes(license_data)
        (destination / 'LICENSE-METADATA.json').write_text(
            json.dumps(source['license_evidence'], indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
    verify((destination / source['license_path']).read_bytes(), source['license_sha256'])
    (destination / 'ATTRIBUTION.txt').write_text(
        source['attribution'] + '\n' + source['repository'] + '\n' + source['license_url'] + '\n',
        encoding='utf-8',
    )
    if 'wrapper_model' in source:
        model = source['wrapper_model']
        world = destination / source['entrypoints'][0]
        world.parent.mkdir(parents=True, exist_ok=True)
        world.write_text(
            '<?xml version="1.0"?>\n'
            f'<sdf version="1.6"><world name="{model}">\n'
            '  <include><uri>model://sun</uri></include>\n'
            '  <include><uri>model://ground_plane</uri></include>\n'
            f'  <include><uri>model://{model}</uri></include>\n'
            '</world></sdf>\n',
            encoding='utf-8',
        )
    for entrypoint in source['entrypoints']:
        if not (destination / entrypoint).is_file():
            raise ValueError(f'Missing world entrypoint: {entrypoint}')
    files = {}
    for path in sorted(destination.rglob('*')):
        if path.is_file():
            data = path.read_bytes()
            files[path.relative_to(destination).as_posix()] = {
                'sha256': hashlib.sha256(data).hexdigest(),
                'size': len(data),
            }
    (destination / 'PROVENANCE.json').write_text(
        json.dumps(
            {
                'source': source,
                'files': files,
                'installed_size': sum(pin['size'] for pin in files.values()),
                'modifications': (
                    'Unchanged upstream model; generated SDF world wrapper and notices.'
                    if 'wrapper_model' in source
                    else 'Unchanged upstream models and worlds; generated attribution notice.'
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + '\n',
        encoding='utf-8',
    )


def install(name: str, root: Path) -> Path:
    source = SOURCES[name]
    worlds = root.resolve() / 'assets' / 'worlds'
    destination = worlds / name
    if destination.exists():
        provenance = json.loads((destination / 'PROVENANCE.json').read_text(encoding='utf-8'))
        if provenance['source'] != source:
            raise ValueError(f'{destination}: installed source does not match the pinned record')
        recorded = provenance['files']
        actual = {
            path.relative_to(destination).as_posix()
            for path in destination.rglob('*')
            if path.is_file()
        }
        if actual != set(recorded) | {'PROVENANCE.json'}:
            raise ValueError(f'{destination}: installed files do not match provenance')
        for relative, pin in recorded.items():
            path = destination / Path(*safe_path(relative).parts)
            verify(path.read_bytes(), pin['sha256'], pin['size'])
        return destination
    data = archive_bytes(source, worlds / '_archives')
    worlds.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{name}-', dir=worlds) as temporary:
        staging = Path(temporary) / name
        staging.mkdir()
        extract(data, source, staging)
        finish_install(source, staging)
        staging.rename(destination)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('worlds', nargs='*', help='Allowlisted names; use --list to see them')
    parser.add_argument('--all', action='store_true', help='Install all six audited sources')
    parser.add_argument('--list', action='store_true', help='List sources without network access')
    parser.add_argument(
        '--root',
        type=Path,
        default=Path(os.environ.get('USIM_ROOT', Path(__file__).resolve().parents[2])),
        help='usim checkout root (default: USIM_ROOT or this script checkout)',
    )
    args = parser.parse_args(argv)
    if args.list:
        for name, source in SOURCES.items():
            print(f'{name}: {source["license"]}, revision {source["revision"]}')
        return 0
    if args.all and args.worlds:
        parser.error('Use --all or named worlds, not both')
    names = list(SOURCES) if args.all else args.worlds
    if not names:
        parser.error('Select named worlds or --all')
    if len(names) != len(set(names)) or any(name not in SOURCES for name in names):
        parser.error('Select each allowlisted world at most once; see --list')
    for name in names:
        print(install(name, args.root))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
