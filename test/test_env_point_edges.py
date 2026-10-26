"""Surface-boundary and normal-change points, preserving original LOF observations."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pytest

from test.test_env_live_lof import save_records
from usim_env_pipeline.models import PipelineInputError
from usim_env_pipeline.pcd import sha256
from usim_env_pipeline.point_edges import (
    EdgeNeighborhood,
    compute_point_edges,
    point_edge_features,
    select_edges,
)


def plane() -> NDArray[np.float64]:
    x, y = np.meshgrid(np.arange(-10, 11) / 16, np.arange(-10, 11) / 16)
    return np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))


def wedge() -> NDArray[np.float64]:
    x, y = np.meshgrid(np.arange(-10, 11) / 16, np.arange(11) / 16)
    floor = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    x, z = np.meshgrid(np.arange(-10, 11) / 16, np.arange(1, 11) / 16)
    wall = np.column_stack((x.ravel(), np.zeros(x.size), z.ravel()))
    return np.vstack((floor, wall))


def test_plane_extracts_border_without_interior_or_crease():
    xyz = plane()
    features = point_edge_features(xyz, EdgeNeighborhood(radius=0.21), workers=1)
    flags = select_edges(features)
    outside = (np.abs(xyz[:, 0]) == 0.625) | (np.abs(xyz[:, 1]) == 0.625)
    inside = (np.abs(xyz[:, 0]) < 0.3) & (np.abs(xyz[:, 1]) < 0.3)
    assert np.all(flags[outside] == 1)
    assert np.all(flags[inside] == 0)
    assert np.all(features.normal_valid)
    assert not np.any(flags & 2)
    np.testing.assert_allclose(features.surface_variation, 0, atol=1e-14)


def test_right_angle_extracts_normal_change_not_entire_planes():
    xyz = wedge()
    features = point_edge_features(xyz, EdgeNeighborhood(radius=0.21), workers=1)
    flags = select_edges(features)
    corner = (np.abs(xyz[:, 0]) < 0.3) & (xyz[:, 1] < 0.125) & (xyz[:, 2] < 0.125)
    distant = (xyz[:, 1] > 0.3) | (xyz[:, 2] > 0.3)
    assert np.count_nonzero(flags[corner] & 2) > 10
    assert not np.any(flags[distant] & 2)


def test_surface_features_are_rotation_translation_invariant():
    xyz = wedge()
    rotate = np.array([[0.36, 0.48, -0.8], [-0.8, 0.6, 0], [0.48, 0.64, 0.6]])
    options = EdgeNeighborhood(radius=0.21)
    before = point_edge_features(xyz, options, workers=1)
    after = point_edge_features(xyz @ rotate + [143, -89, 21], options, workers=1)
    np.testing.assert_array_equal(select_edges(before), select_edges(after))
    np.testing.assert_allclose(before.boundary_gap_degrees, after.boundary_gap_degrees, atol=1e-3)
    np.testing.assert_allclose(before.crease_angle_degrees, after.crease_angle_degrees, atol=1e-3)


@pytest.mark.parametrize('batch_size', [1, 7, 2048])
def test_features_do_not_depend_on_batch_size(batch_size: int):
    xyz = wedge()
    expected = point_edge_features(xyz, EdgeNeighborhood(radius=0.21), workers=1)
    actual = point_edge_features(
        xyz, EdgeNeighborhood(radius=0.21), workers=1, batch_size=batch_size
    )
    np.testing.assert_array_equal(select_edges(actual), select_edges(expected))
    np.testing.assert_array_equal(actual.neighbor_counts, expected.neighbor_counts)
    np.testing.assert_allclose(actual.boundary_gap_degrees, expected.boundary_gap_degrees)
    np.testing.assert_allclose(actual.crease_angle_degrees, expected.crease_angle_degrees)


@pytest.mark.parametrize(
    'xyz',
    [
        np.zeros((20, 3)),
        np.column_stack((np.arange(20) / 16, np.zeros(20), np.zeros(20))),
        np.array([[0.0, 0, 0], [1, 0, 0]]),
        np.random.default_rng(17).normal(size=(100, 3)) * 0.01,
    ],
)
def test_no_surface_evidence_does_not_become_edge(xyz: NDArray[np.float64]):
    features = point_edge_features(xyz, workers=1)
    assert not features.normal_valid.any()
    assert not select_edges(features).any()
    assert np.isfinite(features.boundary_gap_degrees).all()
    assert np.isfinite(features.crease_angle_degrees).all()


def test_radius_is_closed_and_self_excluded():
    xyz = np.array([[0.0, 0, 0], [0.125, 0, 0], [-0.125, 0, 0], [0, 0.125, 0], [0, -0.125, 0]])
    features = point_edge_features(
        xyz, EdgeNeighborhood(neighbors=8, min_neighbors=3, radius=0.125), workers=1
    )
    assert features.neighbor_counts[0] == 4
    assert features.normal_valid[0]
    assert features.boundary_gap_degrees[0] == 90


def test_duplicate_observations_keep_original_rows_without_zero_direction_edges():
    xyz = np.repeat(plane(), 2, axis=0)
    features = point_edge_features(xyz, EdgeNeighborhood(radius=0.21), workers=1)
    flags = select_edges(features)
    center = (np.abs(xyz[:, 0]) < 0.3) & (np.abs(xyz[:, 1]) < 0.3)
    assert len(flags) == len(xyz)
    assert np.all(flags[center] == 0)


def test_selection_toggles_and_thresholds_are_non_destructive():
    features = point_edge_features(wedge(), EdgeNeighborhood(radius=0.21), workers=1)
    flags = select_edges(features, include_boundary=False)
    assert np.any(flags == 2)
    assert not np.any(flags & 1)
    assert not flags.flags.writeable
    assert not features.normal_valid.flags.writeable


def test_strict_angles_only_remove_candidates():
    features = point_edge_features(wedge(), EdgeNeighborhood(radius=0.21), workers=1)
    strict = select_edges(features, boundary_angle=300, crease_angle=80)
    normal = select_edges(features)
    assert np.all((strict & normal) == strict)
    assert np.count_nonzero(strict) < np.count_nonzero(normal)


@pytest.mark.parametrize(
    'parameters',
    [
        {'neighbors': True},
        {'neighbors': 7},
        {'min_neighbors': 65},
        {'radius': 0},
        {'radius': float('nan')},
        {'max_surface_variation': 1 / 3},
    ],
)
def test_invalid_neighborhood_is_rejected(parameters):
    with pytest.raises(PipelineInputError):
        EdgeNeighborhood(**parameters)


def test_nonfinite_xyz_is_rejected():
    with pytest.raises(PipelineInputError):
        point_edge_features(np.array([[0.0, np.inf, 0]]))


def test_invalid_angles_are_rejected():
    features = point_edge_features(plane(), workers=1)
    with pytest.raises(PipelineInputError):
        select_edges(features, crease_angle=91)


def test_cache_is_readonly_and_preserves_source_ids_and_attributes(tmp_path: Path):
    xyz = plane()
    source = save_records(tmp_path / 'records.npy', xyz)
    ids = np.arange(0, len(xyz), 2, dtype=np.int64)
    np.save(tmp_path / 'ids.npy', ids)
    original_hash = sha256(source)
    first = compute_point_edges(source, tmp_path / 'ids.npy', tmp_path / 'cache', workers=1)
    second = compute_point_edges(source, tmp_path / 'ids.npy', tmp_path / 'cache', workers=1)
    assert not first.cached and second.cached
    assert sha256(source) == original_hash
    np.testing.assert_array_equal(second.source_ids, ids)
    np.testing.assert_array_equal(second.xyz, xyz[ids])
    np.testing.assert_array_equal(select_edges(first.features), select_edges(second.features))
    assert not second.features.boundary_gap_degrees.flags.writeable
    assert not second.xyz.flags.writeable
    with pytest.raises(FrozenInstanceError):
        setattr(second, 'cached', False)


def test_selection_change_is_not_stale_or_windows_input_locked(tmp_path: Path):
    source = save_records(tmp_path / 'records.npy', plane())
    np.save(tmp_path / 'ids.npy', np.arange(441, dtype=np.int64))
    first = compute_point_edges(source, tmp_path / 'ids.npy', tmp_path / 'cache', workers=1)
    np.save(tmp_path / 'ids.npy', np.arange(440, dtype=np.int64))
    second = compute_point_edges(source, tmp_path / 'ids.npy', tmp_path / 'cache', workers=1)
    assert first.cache_dir != second.cache_dir and not second.cached
    assert len(first.source_ids) == 441 and len(second.source_ids) == 440


def test_neighborhood_change_uses_separate_cache(tmp_path: Path):
    source = save_records(tmp_path / 'records.npy', plane())
    np.save(tmp_path / 'ids.npy', np.arange(441, dtype=np.int64))
    first = compute_point_edges(source, tmp_path / 'ids.npy', tmp_path / 'cache', workers=1)
    second = compute_point_edges(
        source,
        tmp_path / 'ids.npy',
        tmp_path / 'cache',
        neighborhood=EdgeNeighborhood(radius=0.2),
        workers=1,
    )
    assert first.cache_dir != second.cache_dir and not second.cached


def test_duplicate_source_id_selection_rejected(tmp_path: Path):
    source = save_records(tmp_path / 'records.npy', plane())
    np.save(tmp_path / 'ids.npy', np.array([0, 1, 1], dtype=np.int64))
    with pytest.raises(PipelineInputError):
        compute_point_edges(source, tmp_path / 'ids.npy', tmp_path / 'cache')
