"""Supported obstacles remain closed volumes, not raw fragments or scene-wide hulls."""

import numpy as np
import pytest

pytest.importorskip('scipy')

from usim_env_pipeline.simulation_obstacles import (
    ObstacleConfig,
    occupied_obstacles,
    voxel_surface,
)


def returns(keys, count=4):
    centers = (np.asarray(keys, dtype=np.float32) + 0.5) * 0.4
    return np.repeat(centers, count, axis=0)


def test_narrow_column_and_overhead_object_retained_with_original_record_ownership():
    # Given a supported thin column, a separate overhead block and sparse noise.
    column = returns([[0, 0, z] for z in range(8)])
    overhead = returns([[3, 0, 8]], 16)
    noise = returns([[20, 0, 2]], 2)
    points = np.concatenate((column, overhead, noise))
    original = points.copy()
    # When approximating only occupied, sufficiently supported obstacle volumes.
    occupied = occupied_obstacles(points, np.arange(len(points)), ObstacleConfig(cell=0.4))
    # Then both supported objects and weak isolated returns remain physically represented.
    np.testing.assert_array_equal(occupied.ids, np.arange(len(points)))
    np.testing.assert_array_equal(points, original)
    np.testing.assert_array_equal(
        np.floor(points[occupied.ids] / 0.4).astype(int), occupied.keys[occupied.groups]
    )
    np.testing.assert_array_equal(np.bincount(occupied.groups), occupied.counts)


def test_diagonal_sparse_returns_remain_separate_occupied_cells():
    # Given three connected three-return cells and diagonal-only sparse cells.
    points = returns([[0, 0, z] for z in range(4)], 3)
    noise = returns([[10 + z, 0, z] for z in range(4)], 3)
    all_points = np.concatenate((points, noise))
    # When preserving occupancy rather than forming a transitive diagonal hull.
    occupied = occupied_obstacles(all_points, np.arange(len(all_points)), ObstacleConfig(cell=0.4))
    # Then all observations survive, with no filled cells between diagonal returns.
    np.testing.assert_array_equal(occupied.ids, np.arange(len(all_points)))
    assert len(occupied.keys) == 8


def test_voxel_union_is_closed_outward_and_does_not_bridge_empty_cells():
    # Given adjacent occupied cubes and a separate cube across an empty street.
    keys = np.array([[0, 0, 0], [1, 0, 0], [5, 0, 0]], dtype=np.int64)
    # When producing the occupied-volume boundary.
    points, faces, owners = voxel_surface(keys, 0.4)
    # Then internal faces are absent, boundaries close and the street stays empty.
    assert len(faces) == 32
    triangles = points[faces].astype(np.float64)
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert (counts == 2).all()
    volume = (
        np.einsum('ij,ij->i', triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum()
        / 6
    )
    assert volume == pytest.approx(3 * 0.4**3, rel=1e-6)
    centers = triangles.mean(1)
    assert not ((centers[:, 0] > 0.81) & (centers[:, 0] < 1.99)).any()
    assert set(owners) == {0, 1, 2}


def test_empty_obstacles_produce_empty_geometry():
    # Given no unassigned measured obstacle records.
    points = np.empty((0, 3), dtype=np.float32)
    # When producing occupancy and its geometry.
    occupied = occupied_obstacles(points, np.empty(0, dtype=np.int64), ObstacleConfig())
    positions, faces, owners = voxel_surface(occupied.keys, 0.4)
    # Then there is no fabricated geometry or collider.
    assert positions.shape == faces.shape == (0, 3)
    assert owners.shape == (0,)


def test_single_return_per_cell_fence_remains_physically_represented():
    # Given a sparsely scanned fence which has only one return in each cell.
    points = returns([[0, 0, z] for z in range(8)], 1)
    # When generating occupied volumes rather than treating density as semantics.
    occupied = occupied_obstacles(points, np.arange(len(points)), ObstacleConfig(cell=0.4))
    # Then every observed obstacle record still has physical proxy geometry.
    np.testing.assert_array_equal(occupied.ids, np.arange(len(points)))


@pytest.mark.parametrize(
    'values',
    [
        {'cell': 0},
        {'cell': float('nan')},
        {'confidence_records': 0},
        {'confidence_records': 2.5},
        {'cleanup_radius': 0},
        {'cleanup_radius': float('inf')},
        {'cleanup_min_cells': 0},
        {'cleanup_min_cells': 2.5},
        {'cleanup_min_cells': True},
    ],
)
def test_invalid_obstacle_configuration_is_rejected(values):
    from usim_env_pipeline.models import PipelineInputError

    with pytest.raises(PipelineInputError):
        ObstacleConfig(**values)
