"""Acquire all published Tsudanuma originals with streaming integrity records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

if not __package__:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'usim_env_tsudanuma'

from usim_env_pipeline.models import PipelineInputError
from usim_env_pipeline.acquire import (
    JSONValue,
    acquire,
    download_url,
    official_url,
    require_space,
)

from .sources import (
    ASSETS,
    ATTRIBUTION,
    CATALOG,
    CATALOG_COMMIT,
    CATALOG_URL,
    LICENSE,
    LICENSE_URL,
    PROVIDER,
)

DEFAULT_OUTPUT = Path('assets/tsudanuma/upstream')


def validate_pcd(path: Path, catalog_points: int) -> dict[str, JSONValue]:
    header = {}
    with path.open('rb') as stream:
        for _ in range(100):
            line = stream.readline(4097)
            if not line or len(line) > 4096:
                raise PipelineInputError('Invalid or oversized PCD header')
            text = line.decode('ascii').strip()
            if not text or text.startswith('#'):
                continue
            key, *values = text.split()
            header[key] = values
            if key == 'DATA':
                offset = stream.tell()
                break
        else:
            raise PipelineInputError('Missing PCD DATA header')
    required = {'FIELDS', 'SIZE', 'TYPE', 'WIDTH', 'HEIGHT', 'POINTS', 'DATA'}
    if not required <= header.keys():
        raise PipelineInputError('Incomplete PCD header')
    fields = header['FIELDS']
    sizes = list(map(int, header['SIZE']))
    types = header['TYPE']
    counts = list(map(int, header.get('COUNT', ['1'] * len(fields))))
    if not (len(fields) == len(sizes) == len(types) == len(counts)):
        raise PipelineInputError('Inconsistent PCD field layout')
    points = int(header['POINTS'][0])
    if points <= 0 or points != int(header['WIDTH'][0]) * int(header['HEIGHT'][0]):
        raise PipelineInputError('PCD point count mismatch')
    if not {'x', 'y', 'z', 'intensity'} <= set(fields):
        raise PipelineInputError('PCD is missing published XYZ/intensity fields')
    stride = sum(size * count for size, count in zip(sizes, counts))
    encoding = header['DATA'][0]
    if encoding == 'binary':
        if path.stat().st_size != offset + points * stride:
            raise PipelineInputError('PCD binary payload length mismatch')
    elif encoding == 'binary_compressed':
        with path.open('rb') as stream:
            stream.seek(offset)
            lengths = stream.read(8)
        if len(lengths) != 8:
            raise PipelineInputError('Truncated compressed PCD')
        compressed, uncompressed = struct.unpack('<II', lengths)
        if uncompressed != points * stride or path.stat().st_size != offset + 8 + compressed:
            raise PipelineInputError('PCD compressed payload length mismatch')
    elif encoding == 'ascii':
        rows = 0
        columns = sum(counts)
        with path.open('rb') as stream:
            stream.seek(offset)
            for line in stream:
                values = line.split()
                if not values:
                    continue
                if len(values) != columns:
                    raise PipelineInputError('PCD ASCII field count mismatch')
                for value in values:
                    float(value)
                rows += 1
        if rows != points:
            raise PipelineInputError('PCD ASCII point count mismatch')
    else:
        raise PipelineInputError(f'Unsupported original PCD encoding: {encoding}')
    return {
        'header': {key: [value for value in values] for key, values in header.items()},
        'points': points,
        'stride': stride,
        'data_offset': offset,
        'catalog_points': catalog_points,
        'catalog_points_match': points == catalog_points,
    }


def validate_rosbag(path: Path) -> dict[str, JSONValue]:
    with path.open('rb') as stream:
        magic = stream.read(13)
        if magic != b'#ROSBAG V2.0\n':
            raise PipelineInputError('Invalid ROS1 bag header')
        length_bytes = stream.read(4)
        if len(length_bytes) != 4:
            raise PipelineInputError('Truncated ROS1 bag header')
        length = struct.unpack('<I', length_bytes)[0]
        if length > 4096:
            raise PipelineInputError('Oversized ROS1 bag header')
        body = stream.read(length)
    fields = {}
    cursor = 0
    while cursor < len(body):
        if cursor + 4 > len(body):
            raise PipelineInputError('Truncated ROS1 bag header field')
        size = struct.unpack_from('<I', body, cursor)[0]
        cursor += 4
        field = body[cursor : cursor + size]
        if len(field) != size or b'=' not in field:
            raise PipelineInputError('Invalid ROS1 bag header field')
        key, value = field.split(b'=', 1)
        fields[key.decode('ascii')] = value
        cursor += size
    if fields.get('op') != b'\x03':
        raise PipelineInputError('Missing ROS1 bag file header record')
    result: dict[str, JSONValue] = {'magic': magic.decode('ascii').strip()}
    for key, size in [('index_pos', 8), ('conn_count', 4), ('chunk_count', 4)]:
        if len(fields.get(key, b'')) != size:
            raise PipelineInputError(f'Invalid ROS1 bag {key}')
        result[key] = int.from_bytes(fields[key], 'little')
    index_position = int.from_bytes(fields['index_pos'], 'little')
    if not 0 < index_position < path.stat().st_size:
        raise PipelineInputError('ROS1 bag index is missing or truncated')
    return result


def download(key: str, output_dir: Path, *, progress=None) -> dict[str, JSONValue]:
    asset = ASSETS[key]
    destination = output_dir / asset['name']
    if not destination.exists():
        require_space(output_dir, asset['approximate_bytes'])
    validator = (
        (lambda path: validate_pcd(path, asset.get('points', 0)))
        if asset['format'] == 'pcd'
        else validate_rosbag
    )
    catalog_asset: dict[str, JSONValue] = {
        'name': asset['name'],
        'id': asset['id'],
        'format': asset['format'],
        'approximate_bytes': asset['approximate_bytes'],
    }
    if 'points' in asset:
        catalog_asset['points'] = asset['points']
    if 'date' in asset:
        catalog_asset['date'] = asset['date']
    if 'duration' in asset:
        catalog_asset['duration'] = asset['duration']
    return acquire(
        download_url(asset['id']),
        destination,
        validate=validator,
        progress=progress,
        metadata={
            'official_source': official_url(asset['id']),
            'provider': PROVIDER,
            'license': LICENSE,
            'license_url': LICENSE_URL,
            'attribution': ATTRIBUTION,
            'catalog': CATALOG,
            'catalog_commit': CATALOG_COMMIT,
            'catalog_url': CATALOG_URL,
            'catalog_asset': catalog_asset,
        },
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('assets', nargs='*', metavar='{' + ','.join(ASSETS) + '}')
    args = parser.parse_args(argv)
    if any(key not in ASSETS for key in args.assets):
        parser.error('assets must be selected from ' + ', '.join(ASSETS))
    keys = args.assets or list(ASSETS)
    needed = sum(
        ASSETS[key]['approximate_bytes']
        for key in keys
        if not (args.output_dir / ASSETS[key]['name']).exists()
    )
    if needed:
        require_space(args.output_dir, needed)
    for key in keys:
        last = 0

        def report(count, length):
            nonlocal last
            if count - last >= 128 * 1024 * 1024:
                print(f'PROGRESS {key} {count}/{length or "unknown"} bytes', flush=True)
                last = count

        print(f'START {key}', flush=True)
        record = download(key, args.output_dir, progress=report)
        print('COMPLETE ' + json.dumps(record, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
