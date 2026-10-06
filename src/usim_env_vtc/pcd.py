"""MIT tooling: strict, lossless reader for the public fuRo binary XYZINormal PCD."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from typing import Any

FIELDS = ('x', 'y', 'z', 'intensity', 'normal_x', 'normal_y', 'normal_z', 'curvature')
DTYPE = np.dtype([(field, '<f4') for field in FIELDS])
PUBLIC_HASHES = {
    'map_tc19_o085_f-04_t05.pcd': '119250f3b153bd0948b05e441cee30459bed45ce05cdf5749267bc2fc703309d',
    'map_tc18_o085_f-04_t30.pcd': '6927c661dfcbc50317f3ad61907f4a20088a8bc3111767f0413e02af33a75f4e',
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def open_records(path: Path) -> tuple[np.memmap[Any, np.dtype[np.void]], dict[str, Any]]:
    """Return source-order records, without normalization or dropping any record."""
    lines = []
    declarations = {}
    with path.open('rb') as stream:
        while True:
            line = stream.readline(4097)
            if not line or len(line) > 4096:
                raise ValueError('Invalid PCD header')
            lines.append(line)
            tokens = line.decode('ascii').split()
            if tokens and not tokens[0].startswith('#'):
                declarations[tokens[0]] = tokens[1:]
            if tokens and tokens[0] == 'DATA':
                break
        offset = stream.tell()
        count = int(declarations['POINTS'][0])
        expected = {
            'DATA': ['binary'],
            'FIELDS': list(FIELDS),
            'SIZE': ['4'] * 8,
            'TYPE': ['F'] * 8,
            'COUNT': ['1'] * 8,
        }
        if any(declarations.get(k) != v for k, v in expected.items()):
            raise ValueError('Expected binary little-endian eight-float XYZINormal PCD')
        if count != int(declarations['WIDTH'][0]) * int(declarations['HEIGHT'][0]):
            raise ValueError('PCD dimension mismatch')
        payload_end = offset + count * DTYPE.itemsize
        if path.stat().st_size not in (payload_end, count * DTYPE.itemsize + 4096):
            raise ValueError('PCD byte count mismatch')
        stream.seek(payload_end)
        padding = stream.read()
        if any(padding):
            raise ValueError('Nonzero trailing PCL padding')
    return np.memmap(path, mode='r', dtype=DTYPE, offset=offset, shape=(count,)), {
        'points': count,
        'bytes': path.stat().st_size,
        'data_offset': offset,
        'trailing_zero_bytes': len(padding),
        'header': b''.join(lines).decode('ascii'),
        'fields': list(FIELDS),
        'dtype': 'eight source little-endian float32; stride 32',
    }


def xyz(records: NDArray[np.void]) -> NDArray[np.float32]:
    return np.column_stack([records[field] for field in FIELDS[:3]])


def normals(records: NDArray[np.void]) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    original = np.column_stack([records[field] for field in FIELDS[4:7]]).astype(np.float64)
    length = np.linalg.norm(original, axis=1)
    valid = np.isfinite(original).all(axis=1) & (length > 1e-8)
    original[valid] /= length[valid, None]
    return original, valid
