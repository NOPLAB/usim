"""Finite support and scoring eligibility must describe the same frozen patch."""

import numpy as np
from scipy.spatial import KDTree

from usim_env_pipeline.window_geometry_evidence import (
    cell_observation_weights,
    legacy_section_acceptance_mask,
    patch_observation_evidence,
    section_station_indices,
)


def test_common_pool_retains_junction_points_without_borrowing_a_depth_sheet() -> None:
    t, r = np.meshgrid(np.arange(0, 1.201, 0.05), [-0.04, 0.0, 0.04])
    front = np.column_stack((np.zeros(t.size), r.ravel(), t.ravel()))
    points = np.vstack((front, front + [0.12, 0, 0]))
    before = points.copy()
    component = np.flatnonzero(points[:, 1] < 0)

    result = patch_observation_evidence(
        points,
        np.array([[0, 0, 0], [0, 0, 1.2]], dtype=np.float64),
        np.array([1.0, 0, 0]),
        0.05,
        component,
        KDTree(points),
    )

    assert len(result.parts) == 1
    assert np.count_nonzero(result.component_mask) < len(result.rows)
    np.testing.assert_allclose(points[result.rows, 0], 0)
    assert np.any(points[result.rows, 1] > 0)
    np.testing.assert_array_equal(points, before)


def test_unsupported_middle_of_surface_receives_no_scoring_credit() -> None:
    t = np.r_[np.arange(0, 0.751, 0.05), np.arange(1.2, 2.001, 0.05)]
    measured = np.column_stack((np.zeros(len(t)), np.zeros(len(t)), t))
    middle = np.column_stack((np.zeros(8), np.full(8, 0.14), np.linspace(0.8, 1.15, 8)))
    points = np.vstack((measured, middle))

    result = patch_observation_evidence(
        points,
        np.array([[0, 0, 0], [0, 0, 2]], dtype=np.float64),
        np.array([1.0, 0, 0]),
        0.15,
        np.arange(len(points), dtype=np.int64),
        KDTree(points),
    )

    assert len(result.parts) == 2
    positions = points[result.rows, 2]
    assert np.all((positions <= 0.75 + 1e-12) | (positions >= 1.2 - 1e-12))


def test_residual_gain_is_measured_against_surface_not_distance_to_member_axis() -> None:
    t, r = np.meshgrid(np.arange(0, 1.201, 0.05), [-0.04, 0.0, 0.04])
    points = np.column_stack((np.full(t.size, 0.03), r.ravel(), t.ravel()))

    result = patch_observation_evidence(
        points,
        np.array([[0, 0, 0], [0, 0, 1.2]], dtype=np.float64),
        np.array([1.0, 0, 0]),
        0.05,
        np.arange(len(points), dtype=np.int64),
        KDTree(points),
    )

    np.testing.assert_allclose(result.gains, 0.75, atol=1e-12)
    assert len(result.parts) == 1


def test_no_valid_interval_has_no_observation_credit() -> None:
    points = np.array([[0, 0, 0], [0, 0, 0.1], [0, 0, 0.2]], dtype=np.float64)

    result = patch_observation_evidence(
        points,
        np.array([[0, 0, 0], [0, 0, 0.2]], dtype=np.float64),
        np.array([1.0, 0, 0]),
        0.05,
        np.arange(3, dtype=np.int64),
        KDTree(points),
    )

    assert not result.parts
    assert len(result.rows) == len(result.gains) == 0


def test_replication_within_fixed_source_cell_does_not_create_more_evidence() -> None:
    points = np.array([[0, 0, 0], [0.01, 0.01, 0.01], [0.1, 0, 0]])
    repeated = np.repeat(points, 7, axis=0)

    weights = cell_observation_weights(repeated)

    np.testing.assert_allclose(weights.reshape(3, 7).sum(axis=1), [0.5, 0.5, 1.0])
    assert np.isclose(weights.sum(), 2.0)


def test_short_oblique_strip_has_surface_support_but_not_longitudinal_evidence() -> None:
    t, r = np.meshgrid(np.arange(0, 3.001, 0.02), np.arange(-0.2, 0.201, 0.02))
    points = np.column_stack((np.zeros(t.size), r.ravel(), t.ravel()))
    ends = np.array([[0, -0.15, 1.2], [0, 0.15, 1.8]], dtype=np.float64)
    patch = patch_observation_evidence(
        points,
        ends,
        np.array([1.0, 0, 0]),
        0.15,
        np.arange(len(points), dtype=np.int64),
        KDTree(points),
    )

    longitudinal = legacy_section_acceptance_mask(points, ends, np.array([1.0, 0, 0]), patch.rows)

    assert patch.parts
    assert np.any(patch.gains > 0.9)
    assert not np.any(longitudinal)


def test_source_aligned_diagonal_retains_longitudinal_evidence_without_world_axes() -> None:
    t, r = np.meshgrid(np.arange(0.01, 3.001, 0.02), np.arange(-0.08, 0.081, 0.02))
    points = np.column_stack((np.zeros(t.size), r.ravel(), t.ravel()))
    angle = 0.61
    rotation = np.array(
        [[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]],
        dtype=np.float64,
    )
    ends = np.array([[0, 0, 0.21], [0, 0, 2.81]], dtype=np.float64) @ rotation.T
    rotated = points @ rotation.T
    patch = patch_observation_evidence(
        rotated,
        ends,
        np.array([1.0, 0, 0]),
        0.08,
        np.arange(len(points), dtype=np.int64),
        KDTree(rotated),
    )

    longitudinal = legacy_section_acceptance_mask(rotated, ends, np.array([1.0, 0, 0]), patch.rows)

    assert patch.parts
    assert float(longitudinal.mean()) > 0.8


def test_uncensored_wall_profile_cannot_earn_thin_member_credit() -> None:
    t, r = np.meshgrid(np.arange(0, 3.001, 0.025), np.arange(-1.0, 1.001, 0.025))
    points = np.column_stack((np.zeros(t.size), r.ravel(), t.ravel()))
    ends = np.array([[0, 0, 0.2], [0, 0, 2.8]], dtype=np.float64)
    patch = patch_observation_evidence(
        points,
        ends,
        np.array([1.0, 0, 0]),
        0.08,
        np.arange(len(points), dtype=np.int64),
        KDTree(points),
    )

    longitudinal = legacy_section_acceptance_mask(points, ends, np.array([1.0, 0, 0]), patch.rows)

    assert patch.parts
    assert np.any(patch.gains > 0.99)
    assert not np.any(longitudinal)


def test_terminal_station_is_included_without_clamping_out_of_span_points() -> None:
    positions = np.array([-0.01, 0, 0.149, 0.15, 0.74999, 0.75, 0.75001])

    bins = section_station_indices(positions, 0.75)

    np.testing.assert_array_equal(bins, [-1, 0, 0, 1, 4, 4, -1])
