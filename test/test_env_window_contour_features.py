"""Original-point multiscale contour evidence without density or depth collapse."""

from dataclasses import FrozenInstanceError

import numpy as np
from numpy.typing import NDArray
import pytest

from usim_env_pipeline.models import PipelineInputError
from usim_env_pipeline.window_contour_features import ContourFeatures, contour_features


def sheet(*, half: bool = False, thin: bool = False) -> NDArray[np.float64]:
    axis = np.arange(-20, 21, dtype=np.float64) * 0.025
    x, y = np.meshgrid(axis, axis)
    xyz = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    if half:
        xyz = xyz[xyz[:, 0] >= 0]
    if thin:
        xyz = xyz[np.abs(xyz[:, 1]) <= 0.026]
    return xyz


def origin_row(xyz: NDArray[np.float64]) -> NDArray[np.int64]:
    distances = (xyz * xyz) @ np.ones(3, dtype=np.float64)
    return np.array([np.argmin(distances)], dtype=np.int64)


def scalars(features: ContourFeatures) -> NDArray[np.float64]:
    return np.stack(
        (
            features.centroid_displacement_norm,
            features.linearity,
            features.planarity,
            features.scattering,
            features.surface_variation,
            features.angular_gap,
            features.boundary_evidence,
        )
    )


def test_plane_interior_and_half_plane_border() -> None:
    plane, border = sheet(), sheet(half=True)
    interior = contour_features(plane, origin_row(plane), (0.213, 0.413))
    boundary = contour_features(border, origin_row(border), (0.213, 0.413))
    assert interior.tangent_valid.all() and boundary.tangent_valid.all()
    assert np.all(interior.boundary_evidence < 0.06)
    np.testing.assert_allclose(boundary.angular_gap, np.pi, atol=1e-12)
    assert np.all(interior.centroid_displacement_norm < 1e-15)
    assert np.all(boundary.centroid_displacement_norm > 0.35)
    np.testing.assert_allclose(interior.planarity, 1, atol=1e-12)
    assert interior.neighbor_counts[0, 1] > 500  # No fixed-k truncation.


def test_rotated_translated_scalar_features_and_displacement() -> None:
    xyz = sheet(half=True)
    rotate = np.array([[0.36, 0.48, -0.8], [-0.8, 0.6, 0], [0.48, 0.64, 0.6]])
    shift = np.array([143.0, -89, 21])
    rows = origin_row(xyz)
    base = contour_features(xyz, rows, (0.213, 0.413))
    moved = contour_features(xyz @ rotate + shift, rows, (0.213, 0.413))
    np.testing.assert_array_equal(base.neighbor_counts, moved.neighbor_counts)
    np.testing.assert_allclose(scalars(base), scalars(moved), atol=1e-12)
    np.testing.assert_allclose(base.eigenvalues, moved.eigenvalues, atol=1e-12)
    np.testing.assert_allclose(
        base.centroid_displacement @ rotate, moved.centroid_displacement, atol=1e-12
    )


def test_noisy_thin_strip_is_line_like_at_large_scale() -> None:
    strip, broad = sheet(thin=True), sheet()
    rng = np.random.default_rng(412)
    strip[:, 2] = rng.normal(0, 0.001, len(strip))
    narrow = contour_features(strip, origin_row(strip), (0.073, 0.413))
    wide = contour_features(broad, origin_row(broad), (0.073, 0.413))
    assert narrow.linearity[0, 1] > 0.97
    assert narrow.linearity[0, 1] > narrow.linearity[0, 0]
    assert wide.planarity[0, 1] > 0.99
    assert narrow.boundary_evidence[0, 1] > wide.boundary_evidence[0, 1]


def test_separated_nearby_sheets_retain_depth_and_all_original_support() -> None:
    lower = sheet()
    upper = lower + [0, 0, 0.12]
    xyz = np.vstack((lower, upper))
    features = contour_features(xyz, origin_row(xyz), (0.091, 0.213))
    assert features.eigenvalues[0, 0, 2] == 0
    assert features.eigenvalues[0, 1, 2] > 0.002
    assert features.surface_variation[0, 1] > 0.1
    assert not features.tangent_valid[0, 1]
    assert np.isnan(features.angular_gap).sum() == 1
    q = xyz[origin_row(xyz)[0]]
    for j, radius in enumerate(features.radii):
        distances = ((xyz - q) ** 2) @ np.ones(3, dtype=np.float64)
        assert features.neighbor_counts[0, j] == np.count_nonzero(
            (distances > 0) & (distances <= radius**2)
        )


def test_duplicate_observations_do_not_inflate_support_or_features() -> None:
    xyz = sheet(half=True)
    duplicate = np.vstack((xyz, np.repeat(xyz[::3], 7, axis=0)))
    base = contour_features(xyz, origin_row(xyz), (0.213, 0.413))
    repeated = contour_features(duplicate, origin_row(duplicate), (0.213, 0.413))
    np.testing.assert_array_equal(base.neighbor_counts, repeated.neighbor_counts)
    np.testing.assert_array_equal(scalars(base), scalars(repeated))
    np.testing.assert_array_equal(base.eigenvalues, repeated.eigenvalues)
    cluster = np.repeat([[0.0, 0, 0], [0.1, 0, 0]], 30, axis=0)
    sparse = contour_features(cluster, np.array([0], dtype=np.int64), (0.2,))
    assert sparse.neighbor_counts[0, 0] == 1
    assert not sparse.tangent_valid.any()
    assert np.isnan(sparse.linearity).all()


def test_unchanged_inputs_query_order_batching_and_readonly_output() -> None:
    xyz = sheet()
    rows = np.array([841, 0, 841, 600], dtype=np.int64)
    before, rows_before = xyz.copy(), rows.copy()
    xyz.setflags(write=False)
    rows.setflags(write=False)
    small = contour_features(xyz, rows, (0.213, 0.413), batch_size=1)
    large = contour_features(xyz, rows, (0.213, 0.413), batch_size=8)
    np.testing.assert_array_equal(xyz, before)
    np.testing.assert_array_equal(rows, rows_before)
    np.testing.assert_array_equal(small.query_rows, rows)
    np.testing.assert_array_equal(scalars(small), scalars(large))
    np.testing.assert_array_equal(small.eigenframe, large.eigenframe)
    with pytest.raises(ValueError):
        small.neighbor_counts[0, 0] = 0
    with pytest.raises(FrozenInstanceError):
        setattr(small, 'radii', (0.5,))


def test_empty_isolated_and_collinear_neighborhoods_have_explicit_missing_evidence() -> None:
    xyz = np.column_stack((np.arange(11, dtype=np.float64), np.zeros((11, 2))))
    rows = np.array([5], dtype=np.int64)
    features = contour_features(xyz, rows, (0.1, 4.1))
    np.testing.assert_array_equal(features.neighbor_counts, [[0, 8]])
    assert np.isnan(features.eigenvalues).sum() == 3
    assert np.isnan(features.centroid_displacement).sum() == 3
    assert features.linearity[0, 1] == 1
    assert not features.tangent_valid.any()
    assert np.isnan(features.angular_gap).all()
    empty = contour_features(xyz, np.array([], dtype=np.int64), (0.1,))
    assert empty.eigenframe.shape == (0, 1, 3, 3)


@pytest.mark.parametrize('radii', [(), (0.0,), (np.inf,), (0.3, 0.2), (0.2, 0.2)])
def test_invalid_radii_are_rejected(radii: tuple[float, ...]) -> None:
    xyz = sheet()
    with pytest.raises(PipelineInputError):
        _ = contour_features(xyz, origin_row(xyz), radii)


def test_invalid_coordinates_rows_and_batch_size_are_rejected() -> None:
    xyz = sheet()
    with pytest.raises(PipelineInputError):
        _ = contour_features(xyz, np.array([-1], dtype=np.int64), (0.2,))
    with pytest.raises(PipelineInputError):
        _ = contour_features(xyz, origin_row(xyz), (0.2,), batch_size=0)
    xyz[0, 0] = np.nan
    with pytest.raises(PipelineInputError):
        _ = contour_features(xyz, origin_row(xyz), (0.2,))
