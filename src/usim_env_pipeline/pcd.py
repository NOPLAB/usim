"""Lossless source-order PCD reader for ASCII and little-endian binary records."""

from __future__ import annotations

import hashlib
from enum import Enum
from pathlib import Path
from typing import Final, assert_never

import numpy as np
from numpy.typing import NDArray

from .models import PipelineInputError, SourceMetadata

FIELDS: Final = ('x', 'y', 'z', 'intensity', 'normal_x', 'normal_y', 'normal_z', 'curvature')
DTYPE: Final = np.dtype([(field, '<f4') for field in FIELDS])
SCALAR_TYPES: Final = {
    ('F', 4): '<f4',
    ('F', 8): '<f8',
    ('I', 1): 'i1',
    ('I', 2): '<i2',
    ('I', 4): '<i4',
    ('I', 8): '<i8',
    ('U', 1): 'u1',
    ('U', 2): '<u2',
    ('U', 4): '<u4',
    ('U', 8): '<u8',
}


class Encoding(Enum):
    ASCII = 'ascii'
    BINARY = 'binary'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def open_records(path: Path) -> tuple[NDArray[np.void], SourceMetadata]:
    """Keep every declared field and row; reject compressed or malformed PCD."""
    lines: list[bytes] = []
    declarations: dict[str, list[str]] = {}
    with path.open('rb') as stream:
        while True:
            line = stream.readline(4097)
            if not line or len(line) > 4096 or len(lines) >= 256:
                raise PipelineInputError('Invalid PCD header')
            lines.append(line)
            tokens = line.decode('ascii').split()
            if tokens and not tokens[0].startswith('#'):
                if tokens[0] in declarations:
                    raise PipelineInputError('Duplicate PCD declaration')
                declarations[tokens[0]] = tokens[1:]
            if tokens and tokens[0] == 'DATA':
                break
        offset = stream.tell()
        required = ('FIELDS', 'SIZE', 'TYPE', 'WIDTH', 'HEIGHT', 'POINTS', 'DATA')
        if any(k not in declarations for k in required):
            raise PipelineInputError('Incomplete PCD header')
        if any(len(declarations[k]) != 1 for k in ('WIDTH', 'HEIGHT', 'POINTS', 'DATA')):
            raise PipelineInputError('Invalid PCD scalar declaration')
        names = declarations['FIELDS']
        sizes = list(map(int, declarations['SIZE']))
        kinds = declarations['TYPE']
        counts = list(map(int, declarations.get('COUNT', ['1'] * len(names))))
        count = int(declarations['POINTS'][0])
        if (
            not names
            or len(set(names)) != len(names)
            or not len(names) == len(sizes) == len(kinds) == len(counts)
            or any(c < 1 for c in counts)
            or count < 1
            or count != int(declarations['WIDTH'][0]) * int(declarations['HEIGHT'][0])
        ):
            raise PipelineInputError('PCD dimensions/fields mismatch or empty cloud')
        dtype_fields = []
        for name, size, kind, width in zip(names, sizes, kinds, counts):
            scalar = SCALAR_TYPES.get((kind, size))
            if scalar is None:
                raise PipelineInputError('Unsupported PCD scalar type')
            dtype_fields.append((name, scalar) if width == 1 else (name, scalar, (width,)))
        dtype = np.dtype(dtype_fields)
        padding = b''
        try:
            encoding = Encoding(declarations['DATA'][0])
        except ValueError as error:
            raise PipelineInputError('Only DATA ascii/binary PCD is supported') from error
        if len(declarations['DATA']) != 1:
            raise PipelineInputError('Invalid PCD DATA declaration')
        match encoding:
            case Encoding.BINARY:
                payload_end = offset + count * dtype.itemsize
                if path.stat().st_size not in (payload_end, count * dtype.itemsize + 4096):
                    raise PipelineInputError('PCD byte count mismatch')
                stream.seek(payload_end)
                padding = stream.read()
                if any(padding):
                    raise PipelineInputError('Nonzero trailing PCL padding')
                records = np.memmap(path, mode='r', dtype=dtype, offset=offset, shape=(count,))
            case Encoding.ASCII:
                records = np.loadtxt(stream, dtype=dtype, ndmin=1)
                if len(records) != count:
                    raise PipelineInputError('PCD ASCII record count mismatch')
            case unreachable:
                assert_never(unreachable)
    return records, {
        'points': count,
        'bytes': path.stat().st_size,
        'data_offset': offset,
        'trailing_zero_bytes': len(padding),
        'header': b''.join(lines).decode('ascii'),
        'fields': names,
        'dtype': str(dtype.descr),
    }


def require_xyzinormal(records: NDArray[np.void], metadata: SourceMetadata) -> None:
    """Parse a binary eight-float XYZINormal contract without relaxing its schema."""
    declarations = {
        tokens[0]: tokens[1:]
        for line in metadata.get('header', '').splitlines()
        if (tokens := line.split()) and not tokens[0].startswith('#')
    }
    expected = {
        'DATA': ['binary'],
        'FIELDS': list(FIELDS),
        'SIZE': ['4'] * 8,
        'TYPE': ['F'] * 8,
        'COUNT': ['1'] * 8,
    }
    if records.dtype != DTYPE or any(declarations.get(k) != v for k, v in expected.items()):
        raise PipelineInputError('Expected binary little-endian eight-float XYZINormal PCD')
    metadata['dtype'] = 'eight source little-endian float32; stride 32'


def xyz(records: NDArray[np.void]) -> NDArray[np.float32]:
    return np.column_stack([records[field] for field in ('x', 'y', 'z')]).astype(np.float32)


def normals(records: NDArray[np.void]) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    names = records.dtype.names or ()
    if not all(f in names for f in ('normal_x', 'normal_y', 'normal_z')):
        return np.zeros((len(records), 3)), np.zeros(len(records), dtype=bool)
    original = np.column_stack(
        [records[field] for field in ('normal_x', 'normal_y', 'normal_z')]
    ).astype(np.float64)
    length = np.linalg.norm(original, axis=1)
    valid = np.isfinite(original).all(axis=1) & (length > 1e-8)
    original[valid] /= length[valid, None]
    return original, valid
