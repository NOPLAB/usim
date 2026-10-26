"""CPU terrain repairs use bounded measured support and exact shared triangles."""

import numpy as np
import pytest

pytest.importorskip('scipy')

from usim_env_pipeline.models import PipelineInputError
from usim_env_pipeline.simulation_ground import (
    GroundConfig,
    Terrain,
    reconstruct_terrain,
    terrain_heights,
)


def measured_grid(low: float = -2, high: float = 4, step: float = 0.1):
    axis = np.arange(low, high + step / 2, step)
    x, y = np.meshgrid(axis, axis)
    z = 0.06 * x + 0.03 * y + 0.002 * np.sin(9 * x) * np.cos(7 * y)
    return np.column_stack((x.ravel(), y.ravel(), z.ravel())).astype(np.float32)


def reconstruct(points):
    return reconstruct_terrain(points, np.arange(len(points), dtype=np.int64))


def test_noisy_slope_has_shared_seams_upward_faces_and_immutable_source():
    points = measured_grid()
    original = points.copy()
    terrain = reconstruct(points)
    assert len(terrain.cells) > 100
    assert len(np.unique(terrain.keys, axis=0)) == len(terrain.keys)
    triangles = terrain.positions[terrain.faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    assert (cross[:, 2] > 0).all()
    np.testing.assert_array_equal(points, original)
    np.testing.assert_allclose(terrain.positions[:, :2], terrain.keys * 0.5)
    query = np.array([[0.5, 0.23], [0.5 - 1e-7, 0.23], [0.5 + 1e-7, 0.23]])
    values = terrain_heights(query, terrain)
    np.testing.assert_allclose(values, values[0], atol=1e-7)
    assert np.bincount(terrain.corner_indices.ravel()).max() == 4


def test_sampling_holes_are_filled_by_one_continuous_smooth_surface():
    # Given a sloping floor with both a small gap and a two-meter sampling hole.
    points = measured_grid()
    small = np.linalg.norm(points[:, :2] - [0.25, 0.25], axis=1) < 0.19
    large = ((points[:, :2] > 0) & (points[:, :2] < 2)).all(axis=1)
    # When reconstructing the continuous simulation floor.
    terrain = reconstruct(points[~(small | large)])
    # Then all missing samples have smooth heights on the same connected surface.
    query = np.array([[0.25, 0.25], [1, 1], [0.75, 1], [1.25, 1]])
    values = terrain_heights(query, terrain)
    assert np.isfinite(values).all()
    np.testing.assert_allclose(values, 0.06 * query[:, 0] + 0.03 * query[:, 1], atol=0.03)


def test_half_meter_level_change_is_connected_with_a_bounded_smooth_slope():
    # Given two measured levels with a sampling discontinuity.
    points = measured_grid()
    points[:, 2] = np.where(points[:, 0] < 0.25, 0, 0.5)
    # When creating the smooth simulation floor.
    terrain = reconstruct(points)
    # Then both elevations remain close and the transition has no missing floor.
    values = terrain_heights(np.array([[-1, 1], [1, 1], [0.25, 1]]), terrain)
    np.testing.assert_allclose(values[:2], [0, 0.5], atol=0.08)
    assert 0 < values[2] < 0.5
    triangles = terrain.positions[terrain.faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    assert (np.linalg.norm(normals[:, :2], axis=1) / normals[:, 2] <= 0.35).all()


@pytest.mark.parametrize('spacing', [0.1, 0.3])
def test_small_measured_step_has_a_continuous_smoothed_floor(spacing):
    # Given a small measured step in the authoritative ground.
    points = measured_grid(-2.25, 4.05, step=spacing)
    points[:, 2] = np.where(points[:, 0] < 0, 0, 0.16)
    # When fitting the derived simulation ground.
    terrain = reconstruct(points)
    # Then the continuous floor follows both levels and smoothly connects them.
    heights = terrain_heights(np.array([[-1, 1], [1, 1], [-0.25, 1.25]]), terrain)
    np.testing.assert_allclose(heights[:2], [0, 0.16], atol=0.03)
    assert 0 < heights[2] < 0.16


def test_rasterized_exterior_is_bounded_by_measured_ground_cells():
    # Given a non-grid-aligned boundary whose last measured cell ends at x=0.5.
    points = measured_grid()
    points = points[points[:, 0] < 0.2]
    # When fitting corners and interior probes.
    terrain = reconstruct(points)
    # Then its boundary cell is filled, but there is no unbounded exterior floor.
    values = terrain_heights(np.array([[0.3, 1], [0.6, 1]]), terrain)
    assert np.isfinite(values[0])
    assert np.isnan(values[1])


def test_incompatible_roof_support_cannot_silently_create_a_steep_floor():
    floor = measured_grid(-2, 1)
    roof = floor.copy()
    roof[:, 0] += 8
    roof[:, 2] += 10
    points = np.concatenate((floor, roof))
    with pytest.raises(PipelineInputError, match='continuous slope-bounded'):
        reconstruct(points)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_exact_piecewise_interpolation_negative_coords_and_closed_outer_edges(dtype):
    terrain = Terrain(
        np.array([[-0.5, -0.5, 0], [0, -0.5, 1], [0, 0, 3], [-0.5, 0, 2]], np.float32),
        np.array([[0, 1, 2], [0, 2, 3]], np.int32),
        np.array([[-1, -1], [0, -1], [0, 0], [-1, 0]], np.int64),
        np.array([[-1, -1]], np.int64),
        np.array([[0, 1, 2, 3]], np.int32),
        np.full((4, 12), -1, np.int64),
    )
    # Make the fourth corner nonplanar to exercise both triangles separately.
    terrain.positions[3, 2] = 4
    query = np.array(
        [
            [-0.125, -0.375],
            [-0.375, -0.125],
            [-0.25, -0.25],
            [0, -0.25],
            [-0.25, 0],
            [0, 0],
            [0.01, 0],
            [np.nan, 0],
        ],
        dtype=dtype,
    )
    values = terrain_heights(query, terrain)
    np.testing.assert_allclose(values[:6], [1.25, 2.75, 1.5, 2, 3.5, 3])
    assert np.isnan(values[6:]).all()


@pytest.mark.parametrize(
    'kwargs',
    [
        {'cell': 0},
        {'cell': np.inf},
        {'radius': -1},
        {'nearest_support': 0.8},
        {'max_error': np.nan},
        {'max_slope': 0},
        {'neighbors': 65},
        {'neighbors': 5},
        {'min_neighbors': 2},
        {'neighbors': 6.5},
        {'min_neighbors': True},
    ],
)
def test_invalid_ground_configuration(kwargs):
    with pytest.raises(PipelineInputError):
        GroundConfig(**kwargs)


def test_neighbor_provenance_is_deterministic_and_uses_only_actual_source_ids():
    points = measured_grid()
    selected = np.arange(0, len(points), 2, dtype=np.int64)
    first = reconstruct_terrain(points, selected)
    second = reconstruct_terrain(points, selected[::-1])
    for name in ('positions', 'faces', 'keys', 'cells', 'corner_indices', 'support_ids'):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    assert np.isin(first.support_ids[first.support_ids >= 0], selected).all()
    for position, support in zip(first.positions, first.support_ids):
        support = support[support >= 0]
        if not len(support):
            continue
        assert len(support) >= GroundConfig().min_neighbors
        distances = np.linalg.norm(points[support, :2] - position[:2], axis=1)
        assert distances.max() <= GroundConfig().radius + 1e-7
        assert distances.min() <= GroundConfig().nearest_support + 1e-7


@pytest.mark.parametrize('count', [0, 5, 30])
def test_empty_small_and_collinear_support_returns_typed_empty_arrays(count):
    points = np.column_stack((np.arange(count) * 0.1, np.zeros((count, 2)))).astype(np.float32)
    terrain = reconstruct(points)
    for name, shape, dtype in (
        ('positions', (0, 3), np.float32),
        ('faces', (0, 3), np.int32),
        ('keys', (0, 2), np.int64),
        ('cells', (0, 2), np.int64),
        ('corner_indices', (0, 4), np.int32),
        ('support_ids', (0, 48), np.int64),
    ):
        array = getattr(terrain, name)
        assert array.shape == shape
        assert array.dtype == dtype


def test_missing_neighbors_are_padded_not_invented():
    points = measured_grid(0, 1, step=0.25)
    terrain = reconstruct_terrain(
        points,
        np.arange(len(points), dtype=np.int64),
        GroundConfig(radius=0.6, nearest_support=0.6),
    )
    assert len(terrain.faces)
    assert (terrain.support_ids == -1).any()
    assert ((terrain.support_ids >= -1) & (terrain.support_ids < len(points))).all()


def test_continuous_ground_auditor_rejects_an_interior_hole():
    from dataclasses import replace

    from usim_env_pipeline.simulation_verify_ground import verify_ground

    # Given a complete continuous floor whose interior cell was removed.
    points = measured_grid()
    terrain = reconstruct(points)
    keep = ~(terrain.cells == [2, 2]).all(1)
    broken = replace(
        terrain,
        faces=terrain.faces[np.repeat(keep, 2)],
        cells=terrain.cells[keep],
        corner_indices=terrain.corner_indices[keep],
    )
    # When independently checking ground topology.
    # Then the missing floor is detected even when faces and cell metadata agree.
    with pytest.raises(PipelineInputError, match='without interior holes'):
        verify_ground(broken, points, np.arange(len(points)), GroundConfig())


def test_inferred_gap_nodes_are_explicit_and_satisfy_the_smoothing_audit():
    from usim_env_pipeline.simulation_verify_ground import verify_ground

    # Given measured ground with a two-meter sampling hole.
    points = measured_grid()
    points = points[~((points[:, :2] > 0) & (points[:, :2] < 2)).all(1)]
    # When generating and auditing the continuous floor.
    terrain = reconstruct(points)
    audit = verify_ground(terrain, points, np.arange(len(points)), GroundConfig())
    # Then inferred nodes carry no fabricated records and the exterior is one loop.
    inferred = ~(terrain.support_ids >= 0).any(1)
    assert inferred.any()
    assert (terrain.support_ids[inferred] == -1).all()
    assert audit.inferred_nodes == inferred.sum()
    assert audit.boundary_edges == 52


def test_smoothing_strength_increases_without_removing_steep_transition_cells():
    from usim_env_pipeline.simulation_verify_ground import verify_ground

    # Given two planar floor levels whose initial interpolation is too steep.
    points = measured_grid()
    points[:, 2] = np.where(points[:, 0] < 0.25, 0, 0.9)
    ids = np.arange(len(points), dtype=np.int64)
    # When smoothing against the complete-surface slope limit.
    terrain = reconstruct_terrain(points, ids)
    audit = verify_ground(terrain, points, ids, GroundConfig())
    # Then regularization increases while all cells, not just the flat ones, survive.
    assert terrain.smoothing_strength > 1
    assert audit.slope_max <= GroundConfig().max_slope + 1e-6
    assert len(terrain.cells) == 169
    assert np.isfinite(terrain_heights(np.array([[0.25, 1]]), terrain)).all()
