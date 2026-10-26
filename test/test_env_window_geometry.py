"""Observed patch fitting must not turn crossing support into a new member."""

import numpy as np
from numpy.typing import NDArray

from usim_env_pipeline.window_geometry_fit import fit_member_patch, observed_surface_rows
from usim_env_pipeline.window_geometry_selection import select_patches
from usim_env_pipeline.window_geometry_evaluation import reference_capacity
from usim_env_pipeline.edge_quality import evaluate_edge_quality


def strip(width: float = 0.20) -> NDArray[np.float64]:
    t, r = np.meshgrid(np.arange(0, 3.001, 0.03), np.arange(-width, width + 0.001, 0.02))
    return np.column_stack((np.zeros(t.size), r.ravel(), t.ravel()))


def test_complete_strip_recentres_without_changing_original_points() -> None:
    points = strip()
    before = points.copy()
    seed = np.array([[0, 0.12, 0.2], [0, 0.12, 2.8]], dtype=np.float64)
    patch = fit_member_patch(points, seed, np.array([1.0, 0, 0]))
    assert patch is not None
    np.testing.assert_allclose(patch.endpoints[:, :2], 0, atol=0.015)
    assert patch.half_width_m > 0.15
    assert patch.centre_rms_m < 0.01
    np.testing.assert_array_equal(points, before)


def test_unbounded_wall_is_not_reported_as_a_member() -> None:
    points = strip(1.0)
    patch = fit_member_patch(
        points, np.array([[0, 0, 0.2], [0, 0, 2.8]], dtype=np.float64), np.array([1.0, 0, 0])
    )
    assert patch is None


def test_true_diagonal_is_preserved_without_manhattan_alignment() -> None:
    points = strip(0.08)
    angle = 0.61
    rotation = np.array(
        [[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]]
    )
    seed = np.array([[0, 0, 0.2], [0, 0, 2.8]], dtype=np.float64)
    patch = fit_member_patch(points @ rotation.T, seed @ rotation.T, np.array([1.0, 0, 0]))
    assert patch is not None
    direction = np.diff(patch.endpoints, axis=0)[0]
    direction /= np.linalg.norm(direction)
    np.testing.assert_allclose(direction, rotation[:, 2], atol=0.005)


def test_borrowed_diagonal_support_does_not_fit_a_new_longitudinal_member() -> None:
    points = strip(0.06)
    seed = np.array([[0, -0.4, 1.1], [0, 0.4, 1.9]], dtype=np.float64)
    patch = fit_member_patch(points, seed, np.array([1.0, 0, 0]))
    assert patch is None


def test_independent_nearby_depth_sheets_do_not_merge() -> None:
    points = strip(0.08)
    both = np.vstack((points, points + [0.12, 0, 0]))
    patch = fit_member_patch(
        both, np.array([[0.12, 0, 0.2], [0.12, 0, 2.8]]), np.array([1.0, 0, 0])
    )
    assert patch is not None
    np.testing.assert_allclose(both[patch.original_rows, 0], 0.12, atol=1e-12)
    np.testing.assert_allclose(patch.endpoints[:, 0], 0.12, atol=1e-12)


def test_rigid_rotation_and_translation_preserve_observed_fit() -> None:
    points = strip(0.08)
    seed = np.array([[0, 0, 0.2], [0, 0, 2.8]], dtype=np.float64)
    angle = 0.79
    rotation = np.array(
        [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]],
        dtype=np.float64,
    )
    shift = np.array([3.1, -2.7, 4.8])
    patch = fit_member_patch(
        points @ rotation.T + shift,
        seed @ rotation.T + shift,
        rotation @ np.array([1.0, 0, 0]),
    )
    assert patch is not None
    recovered = (patch.endpoints - shift) @ rotation
    np.testing.assert_allclose(recovered[:, :2], 0, atol=0.015)


def test_shared_coverage_discards_an_alias_without_deleting_the_member() -> None:
    incidence = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=np.bool_)
    result = select_patches(incidence, np.full(3, 0.25))
    assert result.optimal
    assert len(result.indices) == 2
    assert 2 in result.indices
    assert np.all(incidence[:, result.indices].any(axis=1))
    assert result.objective == -3.5


def test_borrowed_support_is_explained_by_the_longitudinal_members() -> None:
    incidence = np.array([[1, 0, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0]], dtype=np.bool_)
    result = select_patches(incidence, np.full(3, 0.25))
    np.testing.assert_array_equal(result.indices, [0, 1])


def test_independently_observed_diagonal_and_depth_sheet_are_not_explained_away() -> None:
    incidence = np.array(
        [
            [1, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
        ],
        dtype=np.bool_,
    )
    result = select_patches(incidence, np.full(4, 0.25))
    np.testing.assert_array_equal(result.indices, [0, 1, 2, 3])


def test_evidence_permutation_does_not_change_selection() -> None:
    incidence = np.array([[1, 0, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0]], dtype=np.bool_)
    result = select_patches(incidence[[3, 0, 2, 1]], np.full(3, 0.25))
    np.testing.assert_array_equal(result.indices, [0, 1])


def test_measured_surface_retains_observed_junction_without_borrowing_another_depth() -> None:
    points = strip(0.08)
    both = np.vstack((points, points + [0.12, 0, 0]))
    endpoints = np.array([[0, 0, 0.2], [0, 0, 2.8]], dtype=np.float64)
    rows = observed_surface_rows(both, endpoints, np.array([1.0, 0, 0]), 0.08)
    np.testing.assert_allclose(both[rows, 0], 0, atol=1e-12)
    assert np.any((both[rows, 2] >= 1.45) & (both[rows, 2] <= 1.55))
    assert np.min(both[rows, 2]) >= 0.2
    assert np.max(both[rows, 2]) <= 2.8


def test_nearby_parallel_aliases_cannot_both_receive_reference_credit() -> None:
    lines = np.array([[[0, 0, 0], [3, 0, 0]], [[0, 0.02, 0], [3, 0.02, 0]]], dtype=np.float64)
    reference = lines[:1]
    quality = evaluate_edge_quality(
        lines, np.ones(2, dtype=np.uint8), reference, np.ones(1, dtype=np.uint8)
    ).boundary
    result = reference_capacity(quality)
    assert result.precision == 0.5
    assert result.recall == 1.0
    assert result.uncredited_length_m == 3.0
    assert result.matched_prediction_surplus_m == 3.0


def test_verified_alias_conflicts_do_not_remove_an_independent_sheet() -> None:
    incidence = np.array([[1, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.bool_)
    result = select_patches(
        incidence, np.full(3, 0.25), conflicts=np.array([[0, 1]], dtype=np.int64)
    )
    assert len(result.indices) == 2
    assert 2 in result.indices
    assert not ({0, 1} <= set(result.indices))


def test_member_centre_is_not_shifted_by_uneven_transverse_sampling_density() -> None:
    transverse = np.r_[np.arange(-0.20, 0.001, 0.04), np.arange(0.01, 0.201, 0.01)]
    t, r = np.meshgrid(np.arange(0, 3.001, 0.03), transverse)
    points = np.column_stack((np.zeros(t.size), r.ravel(), t.ravel()))
    seed = np.array([[0, 0, 0.2], [0, 0, 2.8]], dtype=np.float64)

    patch = fit_member_patch(
        points, seed, np.array([1.0, 0, 0]), centre_rule='transverse_midpoint'
    )

    assert patch is not None
    np.testing.assert_allclose(patch.endpoints[:, :2], 0, atol=0.015)


def test_five_independent_observed_sections_can_fit_a_supported_short_member() -> None:
    t, r = np.meshgrid(np.arange(0, 0.801, 0.20), [-0.04, 0, 0.04])
    points = np.column_stack((np.zeros(t.size), r.ravel(), t.ravel()))
    seed = np.array([[0, 0, 0], [0, 0, 0.8]], dtype=np.float64)

    patch = fit_member_patch(points, seed, np.array([1.0, 0, 0]))

    assert patch is not None
    np.testing.assert_allclose(patch.endpoints[:, :2], 0, atol=0.015)
    np.testing.assert_allclose(patch.endpoints[:, 2], [0, 0.8], atol=1e-12)


def test_observed_front_surface_depth_is_not_midpoint_of_sparse_depth_outliers() -> None:
    points = strip(0.08)
    depth, t = np.meshgrid(np.arange(0.02, 0.101, 0.02), np.arange(0, 3.001, 0.10))
    outliers = np.column_stack((depth.ravel(), np.zeros(t.size), t.ravel()))
    observed = np.vstack((points, outliers))
    seed = np.array([[0, 0, 0.2], [0, 0, 2.8]], dtype=np.float64)

    patch = fit_member_patch(
        observed, seed, np.array([1.0, 0, 0]), centre_rule='transverse_midpoint'
    )

    assert patch is not None
    np.testing.assert_allclose(patch.endpoints[:, 0], 0, atol=0.015)


def test_depth_tilted_seed_can_recover_observed_axis_instead_of_adding_seed_slope() -> None:
    points = strip(0.08)
    seed = np.array([[0.024, 0, 0.2], [0.336, 0, 2.8]], dtype=np.float64)

    patch = fit_member_patch(points, seed, np.array([1.0, 0, 0]))

    assert patch is not None
    np.testing.assert_allclose(patch.endpoints[:, :2], 0, atol=0.015)


def test_reported_strip_normal_and_plane_residual_describe_the_exported_surface() -> None:
    points = strip(0.08)
    points[:, 0] = 0.02 * np.sin(3 * points[:, 2])
    seed = np.array([[0, 0, 0.2], [0, 0, 2.8]], dtype=np.float64)

    patch = fit_member_patch(points, seed, np.array([1.0, 0, 0]))

    assert patch is not None
    distances = (patch.corners - patch.endpoints[0]) @ patch.normal
    np.testing.assert_allclose(distances, 0, atol=1e-12)
    actual: NDArray[np.float64] = (points[patch.original_rows] - patch.endpoints[0]) @ patch.normal
    np.testing.assert_allclose(patch.plane_rms_m, np.sqrt(np.mean(actual**2)), atol=1e-12)
