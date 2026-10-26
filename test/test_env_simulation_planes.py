"""Residual planes use measured occupancy, not repaired surface extents."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pytest

pytest.importorskip('scipy')

from usim_env_pipeline.plane_types import plane_basis
from usim_env_pipeline.simulation_planes import PlanarProxy, reconstruct_proxies


def archive(folder: Path, number: int, normal, offset: float, ids) -> None:
    np.savez(
        folder / f'plane_{number:04}.npz',
        equation=np.append(normal, offset),
        ids=np.asarray(ids, dtype=np.int64),
        # Deliberately unsafe inferred data must have no effect on reconstruction.
        mask=np.ones((100, 100), dtype=bool),
        positions=np.array([[-100, -100, -100], [100, 100, 100]], dtype=np.float32),
    )


def samples(
    keys, normal, offset: float = 0.0, noise: float | list[float] = 0.0
) -> NDArray[np.float32]:
    normal = np.asarray(normal, dtype=np.float64)
    u, v = plane_basis(normal)
    uv = (np.asarray(keys) + 0.5) * 0.4
    return (
        uv[:, :1] * u + uv[:, 1:2] * v + (np.asarray(noise) - offset)[..., None] * normal
    ).astype(np.float32)


def closed_volume(proxy: PlanarProxy) -> float:
    edges = np.concatenate(
        (proxy.faces[:, [0, 1]], proxy.faces[:, [1, 2]], proxy.faces[:, [2, 0]])
    )
    _, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    assert (counts == 2).all()
    # Translate before summing to avoid cancellation at large meter coordinates.
    triangles = proxy.positions[proxy.faces].astype(np.float64)
    triangles -= proxy.positions.mean(axis=0, dtype=np.float64)
    return float(
        np.einsum('ij,ij->i', triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum()
        / 6
    )


def contains(proxy: PlanarProxy, point: NDArray[np.float32]) -> bool:
    # The closed mesh's solid angle checks actual volume, not only its key metadata.
    a, b, c = (proxy.positions[proxy.faces].astype(np.float64) - point).transpose(1, 0, 2)
    la, lb, lc = (np.linalg.norm(x, axis=1) for x in (a, b, c))
    numerator = np.einsum('ij,ij->i', a, np.cross(b, c))
    denominator = (
        la * lb * lc
        + np.einsum('ij,ij->i', a, b) * lc
        + np.einsum('ij,ij->i', b, c) * la
        + np.einsum('ij,ij->i', c, a) * lb
    )
    return bool(abs((2 * np.arctan2(numerator, denominator)).sum()) > 2 * np.pi)


def test_noisy_wall_is_flat_closed_and_contains_only_in_thickness_support(tmp_path):
    # Given wall samples scattered through its thickness, plus one outlying return.
    keys = [[0, 0], [0, 0], [1, 0], [1, 1], [0, 1], [0, 1]]
    points = samples(keys, [1, 0, 0], -2, [-0.079, 0.079, -0.04, 0.04, 0, 0.081])
    archive(tmp_path, 7, [1, 0, 0], -2, np.arange(len(points)))
    # When using the measured plane rather than source point voxel depths.
    (proxy,) = reconstruct_proxies(
        points, np.ones(len(points), dtype=bool), np.empty(0, np.int64), tmp_path
    )
    # Then the volume is flat, outward, closed and contains all claimed measurements.
    assert proxy.source_plane == 7
    np.testing.assert_array_equal(proxy.support_ids, np.arange(5))
    np.testing.assert_allclose(np.unique(proxy.positions[:, 0]), [1.92, 2.08], atol=1e-6)
    assert closed_volume(proxy) == pytest.approx(4 * 0.4**2 * 0.16, rel=1e-6)
    assert all(contains(proxy, point) for point in points[proxy.support_ids])
    assert not contains(proxy, points[-1])
    assert proxy.positions.dtype == np.float32
    assert proxy.faces.dtype == np.int32
    assert proxy.keys.dtype == proxy.support_ids.dtype == np.int64
    assert proxy.normal.dtype == proxy.u.dtype == proxy.v.dtype == np.float64


@pytest.mark.parametrize(
    'keys,empty',
    [
        ([[0, 0]], [[1, 0], [-1, 0], [0, 1]]),
        ([[x, y] for x in range(3) for y in range(3) if (x, y) != (1, 1)], [[1, 1]]),
        ([[x, y] for x in (0, 2) for y in range(3)], [[1, y] for y in range(3)]),
    ],
)
def test_sparse_cells_preserve_single_return_holes_and_passage(tmp_path, keys, empty):
    # Given measured cells separated by a hole or a missing passage.
    points = samples(keys, [1, 0, 0])
    archive(tmp_path, 0, [1, 0, 0], 0, np.arange(len(points)))
    # When building only the measured cell union.
    (proxy,) = reconstruct_proxies(
        points, np.ones(len(points), bool), np.empty(0, np.int64), tmp_path
    )
    # Then the actual closed mesh retains observations and excludes unmeasured cells.
    np.testing.assert_array_equal(proxy.keys, np.unique(keys, axis=0))
    assert closed_volume(proxy) == pytest.approx(len(keys) * 0.4**2 * 0.16, rel=1e-6)
    assert all(contains(proxy, point) for point in points)
    assert not any(contains(proxy, point) for point in samples(empty, [1, 0, 0]))


@pytest.mark.parametrize('normal', [[0.6, 0.8, 0], [-0.6, -0.8, 0], [0.8, 0, 0.6]])
def test_rotated_and_sloping_slabs_have_positive_volume_at_meter_coordinates(tmp_path, normal):
    # Given a distant, rotated source plane with a sparse observed strip.
    keys = [[2500, -1800], [2501, -1800], [2502, -1800]]
    points = samples(keys, normal, -1200, [0.02, -0.02, 0])
    archive(tmp_path, 0, normal, -1200, np.arange(3))
    # When extruding its right-handed planar basis.
    (proxy,) = reconstruct_proxies(points, np.ones(3, bool), np.empty(0, np.int64), tmp_path)
    # Then winding remains outward and geometry retains its world meter coordinates.
    assert closed_volume(proxy) == pytest.approx(3 * 0.4**2 * 0.16, rel=0.002)
    assert all(contains(proxy, point) for point in points)
    np.testing.assert_allclose(proxy.normal, normal)
    np.testing.assert_allclose(np.cross(proxy.u, proxy.v), normal)
    distances = proxy.positions.astype(np.float64) @ proxy.normal + proxy.offset
    np.testing.assert_allclose(np.abs(distances), 0.08, atol=0.0002)


@pytest.mark.parametrize(
    'height,ground_xy,with_ground,accepted',
    [
        (0, 0, True, False),
        (1, 0, True, False),
        (3, 0, True, True),
        (3, 11, True, False),
        (3, 0, False, False),
        (3, 10, True, True),
    ],
)
def test_horizontal_planes_need_nearby_measured_ground_and_clearance(
    tmp_path, height, ground_xy, with_ground, accepted
):
    # Given a horizontal sample and measured ground at a controlled XY distance.
    points = np.array([[0, 0, height], [ground_xy, 0, 0]], dtype=np.float32)
    archive(tmp_path, 0, [0, 0, 1], -height, [0])
    ground_ids = np.array([1] if with_ground else [], dtype=np.int64)
    # When checking whether the plane is elevated rather than ground-like.
    proxies = reconstruct_proxies(points, np.ones(2, bool), ground_ids, tmp_path)
    # Then only elevated planes with a local measured reference become obstacles.
    assert bool(proxies) == accepted
    if accepted:
        assert closed_volume(proxies[0]) == pytest.approx(0.4**2 * 0.16, rel=1e-6)


@pytest.mark.parametrize('near_count,accepted', [(1, False), (2, True)])
def test_horizontal_reference_must_cover_at_least_half_the_assigned_samples(
    tmp_path, near_count, accepted
):
    # Given four roof samples, only some within ten meters of measured ground.
    points = np.array(
        [[x, 0, 3] for x in [0] * near_count + [30] * (4 - near_count)] + [[0, 0, 0]],
        dtype=np.float32,
    )
    archive(tmp_path, 0, [0, 0, 1], -3, np.arange(4))
    # When classifying the whole source-backed horizontal plane.
    proxies = reconstruct_proxies(points, np.ones(5, bool), np.array([4]), tmp_path)
    # Then half is sufficient, but less than half cannot establish a roof.
    assert bool(proxies) == accepted


def test_overlapping_archives_claim_available_support_once_without_mutating_inputs(tmp_path):
    # Given archives created in reverse order with duplicate and overlapping source IDs.
    points = samples([[0, 0], [1, 0], [2, 0], [3, 0]], [1, 0, 0])
    available = np.array([True, False, True, True])
    ground_ids = np.array([1], dtype=np.int64)
    original_points, original_available, original_ground = (
        points.copy(),
        available.copy(),
        ground_ids.copy(),
    )
    archive(tmp_path, 9, [1, 0, 0], 0, [3, 2, 0])
    archive(tmp_path, 2, [1, 0, 0], 0, [2, 1, 0, 0])
    # When processing all archives deterministically.
    proxies = reconstruct_proxies(points, available, ground_ids, tmp_path)
    # Then the first sorted archive owns overlaps and the caller's inputs are untouched.
    assert [proxy.source_plane for proxy in proxies] == [2, 9]
    np.testing.assert_array_equal(proxies[0].support_ids, [0, 2])
    np.testing.assert_array_equal(proxies[1].support_ids, [3])
    claimed = np.concatenate([proxy.support_ids for proxy in proxies])
    assert len(np.unique(claimed)) == len(claimed)
    np.testing.assert_array_equal(points, original_points)
    np.testing.assert_array_equal(available, original_available)
    np.testing.assert_array_equal(ground_ids, original_ground)


def test_empty_available_support_produces_no_proxy(tmp_path):
    # Given a measured plane whose only point is already owned.
    points = samples([[0, 0]], [1, 0, 0])
    archive(tmp_path, 0, [1, 0, 0], 0, [0])
    # When reconstructing the residual set.
    proxies = reconstruct_proxies(points, np.zeros(1, bool), np.empty(0, np.int64), tmp_path)
    # Then no geometry is fabricated from the archive's inferred surface.
    assert proxies == []
