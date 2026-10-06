"""Lossless public-map reader checks, without the optional USD/Isaac runtime."""

import numpy as np
import pytest
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / 'src' / 'usim_env_vtc' / 'pcd.py'
SPEC = importlib.util.spec_from_file_location('vtc_full_pcd', SCRIPT)
assert SPEC is not None and SPEC.loader is not None
pcd = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pcd)
DTYPE, FIELDS = pcd.DTYPE, pcd.FIELDS
normals, open_records, xyz = pcd.normals, pcd.open_records, pcd.xyz


def write_pcd(path, records, padding=b''):
    header = (
        'VERSION .7\nFIELDS ' + ' '.join(FIELDS) + '\n'
        'SIZE 4 4 4 4 4 4 4 4\nTYPE F F F F F F F F\nCOUNT 1 1 1 1 1 1 1 1\n'
        f'WIDTH {len(records)}\nHEIGHT 1\nPOINTS {len(records)}\nDATA binary\n'
    ).encode('ascii')
    path.write_bytes(header + records.tobytes() + padding)
    return header


def test_source_fields_and_order_are_preserved(tmp_path):
    records = np.zeros(3, dtype=DTYPE)
    for column, field in enumerate(FIELDS):
        records[field] = np.array([3.125, -1.5, 9.75]) + column
    path = tmp_path / 'survey.pcd'
    header = write_pcd(path, records)
    actual, metadata = open_records(path)
    assert actual.tobytes() == records.tobytes()
    assert metadata['data_offset'] == len(header)
    assert metadata['points'] == 3
    assert np.array_equal(xyz(actual), np.column_stack([records[f] for f in FIELDS[:3]]))
    before = actual.tobytes()
    normalized, valid = normals(actual)
    assert valid.all()
    assert np.allclose(np.linalg.norm(normalized, axis=1), 1)
    assert actual.tobytes() == before


def test_pcl_header_page_padding_is_retained(tmp_path):
    path = tmp_path / 'survey.pcd'
    records = np.zeros(1, dtype=DTYPE)
    header = write_pcd(path, records)
    padding = b'\0' * (4096 - len(header))
    write_pcd(path, records, padding)
    actual, metadata = open_records(path)
    assert actual.tobytes() == records.tobytes()
    assert metadata['trailing_zero_bytes'] == len(padding)


@pytest.mark.parametrize('damage', ['truncate', 'nonzero-padding', 'field-order'])
def test_invalid_binary_contract_is_rejected(tmp_path, damage):
    path = tmp_path / 'survey.pcd'
    records = np.zeros(1, dtype=DTYPE)
    header = write_pcd(path, records)
    if damage == 'truncate':
        path.write_bytes(header + records.tobytes()[:-1])
    elif damage == 'nonzero-padding':
        write_pcd(path, records, b'\1' + b'\0' * (4095 - len(header)))
    else:
        path.write_bytes(path.read_bytes().replace(b'FIELDS x y z', b'FIELDS y x z'))
    with pytest.raises(ValueError):
        open_records(path)
