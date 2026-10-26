"""Parser and configuration boundaries need no native reconstruction environment."""

import numpy as np
import pytest

from test.env_pipeline_fixtures import plane_records, write_pcd
from usim_env_pipeline import MapConfig
from usim_env_pipeline.cloud import colors, read_cloud
from usim_env_pipeline.models import PipelineInputError, Support
from usim_env_pipeline.pcd import DTYPE, normals, open_records
from usim_env_pipeline.storage import tile_name


@pytest.mark.parametrize('encoding', ['ascii', 'binary'])
def test_arbitrary_pcd_fields_preserve_order_types_and_vector_values(tmp_path, encoding):
    dtype = np.dtype(
        [
            ('label', '<u2'),
            ('z', '<f8'),
            ('x', '<f4'),
            ('descriptor', '<f4', (2,)),
            ('y', '<f4'),
            ('rgb', '<u4'),
        ]
    )
    records = np.zeros(2, dtype=dtype)
    records['label'] = [65535, 7]
    records['x'], records['y'], records['z'] = [3.125, -1.5], [2, 9], [0.25, 12.5]
    records['descriptor'] = [[1, 2], [-3, 4]]
    records['rgb'] = [0xFF0000, 0x0080FF]
    path = tmp_path / 'arbitrary.pcd'
    original = write_pcd(path, records, encoding=encoding)

    actual, metadata = open_records(path)
    cloud = read_cloud(path)

    assert actual.dtype == dtype
    assert actual.tobytes() == records.tobytes()
    assert cloud.records.tobytes() == records.tobytes()
    assert metadata['fields'] == list(dtype.names or ())
    assert metadata['points'] == 2
    assert metadata['bytes'] == len(original)
    np.testing.assert_array_equal(
        colors(actual), np.array([[1, 0, 0], [0, 128 / 255, 1]], dtype=np.float32)
    )
    assert path.read_bytes() == original


def test_normal_validation_does_not_rewrite_supplied_records(tmp_path):
    records = plane_records()[:4].copy()
    records['normal_z'] = [3, -2, 0, np.nan]
    path = tmp_path / 'normals.pcd'
    original = write_pcd(path, records)

    cloud = read_cloud(path)
    working, valid = normals(cloud.records)

    np.testing.assert_array_equal(valid, [True, True, False, False])
    np.testing.assert_array_equal(working[:2], [[0, 0, 1], [0, 0, -1]])
    assert cloud.records.tobytes() == records.tobytes()
    assert path.read_bytes() == original


@pytest.mark.parametrize('damage', ['empty', 'nan', 'infinity', 'vector-xyz'])
def test_cloud_rejects_unsupported_xyz_before_geometry(tmp_path, damage):
    dtype = (
        np.dtype([('x', '<f4', (2,)), ('y', '<f4'), ('z', '<f4')])
        if damage == 'vector-xyz'
        else DTYPE
    )
    records = np.zeros(0 if damage == 'empty' else 1, dtype=dtype)
    if damage in ('nan', 'infinity'):
        records['x'] = np.nan if damage == 'nan' else np.inf
    path = tmp_path / 'invalid.pcd'
    write_pcd(path, records)

    with pytest.raises(PipelineInputError):
        read_cloud(path)


@pytest.mark.parametrize(
    'kwargs',
    [
        {'workers': 0},
        {'tile_size': np.nan},
        {'halo': np.inf},
        {'halo': 0.69},
        {'halo': 30},
        {'normal_policy': 'invented'},
    ],
)
def test_invalid_map_configuration_is_rejected(kwargs):
    with pytest.raises(PipelineInputError):
        MapConfig(**kwargs)


@pytest.mark.parametrize(
    'kwargs',
    [
        {'radii': ()},
        {'radii': (0.2, 0.1)},
        {'radii': (0.1, 0.1)},
        {'max_edge': 0},
        {'min_double_area': np.nan},
        {'min_face_normal_dot': 1.1},
    ],
)
def test_invalid_support_is_rejected(kwargs):
    with pytest.raises(PipelineInputError):
        Support(**kwargs)


def test_large_negative_regional_keys_are_valid_usd_names():
    # Given a survey whose source frame contains large negative coordinates.
    key = (-10001, -25000)
    # When assigning the payload/prim basename.
    name = tile_name(key)
    # Then it remains a valid USD identifier without collisions with nearby cores.
    assert name.isidentifier()
    assert name != tile_name((-10000, -25000))


def test_strict_schema_accepts_header_whitespace_without_relaxing_fields(tmp_path):
    from usim_env_vtc.pcd import open_records as open_vtc

    # Given a valid strict record payload and extra spaces in a PCD declaration.
    records = plane_records()[:3]
    path = tmp_path / 'spaced.pcd'
    original = write_pcd(path, records)
    path.write_bytes(original.replace(b'DATA binary', b'DATA  binary'))
    # When parsing through the dataset adapter.
    actual, _ = open_vtc(path)
    # Then the original payload bits are preserved.
    assert actual.tobytes() == records.tobytes()
