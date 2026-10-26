"""Small source-order point clouds shared by parser and reconstruction regressions."""

from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from usim_env_pipeline.pcd import DTYPE


def write_pcd(
    path: Path,
    records: NDArray[np.void],
    *,
    encoding: Literal['ascii', 'binary'] = 'binary',
    padding: bool = False,
) -> bytes:
    names = records.dtype.names
    assert names is not None
    sizes, kinds, counts = [], [], []
    for name in names:
        dtype = records.dtype[name]
        scalar = dtype.base
        sizes.append(str(scalar.itemsize))
        kinds.append({'f': 'F', 'i': 'I', 'u': 'U'}[scalar.kind])
        counts.append(str(int(np.prod(dtype.shape)) if dtype.shape else 1))
    header = (
        '# regression fixture\nVERSION .7\nFIELDS ' + ' '.join(names) + '\n'
        'SIZE ' + ' '.join(sizes) + '\nTYPE ' + ' '.join(kinds) + '\n'
        'COUNT ' + ' '.join(counts) + '\n'
        f'WIDTH {len(records)}\nHEIGHT 1\nPOINTS {len(records)}\nDATA {encoding}\n'
    ).encode('ascii')
    if encoding == 'binary':
        payload = records.tobytes()
    else:
        rows = []
        for record in records:
            values = [value for name in names for value in np.asarray(record[name]).reshape(-1)]
            rows.append(' '.join(str(value) for value in values))
        payload = ('\n'.join(rows) + '\n').encode('ascii')
    suffix = b'\0' * (4096 - len(header)) if padding else b''
    path.write_bytes(header + payload + suffix)
    return header + payload + suffix


def plane_records(*, invalid_core: bool = False) -> NDArray[np.void]:
    x, y = np.meshgrid(29.4 + np.arange(13) * 0.1, 2.0 + np.arange(9) * 0.1)
    positions = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    if invalid_core:
        positions = np.vstack((positions, [[65.0, 2.0, 0], [65.1, 2.0, 0], [65, 2.1, 0]]))
    dtype: np.dtype[np.void] = DTYPE
    records = np.zeros(len(positions), dtype=dtype)
    for axis, name in enumerate(('x', 'y', 'z')):
        records[name] = positions[:, axis]
    records['normal_z'] = np.where(np.arange(len(records)) % 2, -2.0, 3.0)
    records['intensity'] = np.arange(len(records)) % 256
    records['curvature'] = np.arange(len(records)) * np.float32(0.001)
    if invalid_core:
        records['normal_z'][-3:] = 0
    return records
