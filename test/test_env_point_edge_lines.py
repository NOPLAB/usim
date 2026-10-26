"""Straight structural lines require original plane evidence, not candidate scatter."""

from dataclasses import replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pytest

from test.test_env_live_lof import save_records
from usim_env_pipeline.edge_line_geometry import EdgeLineConfig, extract_edge_lines, region_lines
from usim_env_pipeline.edge_line_support import LineSupport
from usim_env_pipeline.pcd import sha256
from usim_env_pipeline.point_edge_lines import compute_edge_lines


def corner() -> NDArray[np.float64]:
    x, y = np.meshgrid(np.arange(1, 5.001, 0.05), np.arange(0, 3.001, 0.05))
    floor = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    x, z = np.meshgrid(np.arange(1, 5.001, 0.05), np.arange(0.05, 3.001, 0.05))
    wall = np.column_stack((x.ravel(), np.zeros(x.size), z.ravel()))
    return np.vstack((floor, wall)).astype(np.float64)


def test_actual_corner_has_a_supported_line_not_a_wireframe():
    xyz = corner()
    before = xyz.copy()
    lines = region_lines(xyz, np.zeros(len(xyz), dtype=np.bool_))
    assert lines
    for edge in lines:
        assert edge.kind == 2
        np.testing.assert_allclose(edge.line.endpoints[:, 1:], 0, atol=1e-5)
        assert edge.line.endpoints[:, 0].min() >= 1 - 1e-5
        assert edge.line.endpoints[:, 0].max() <= 5 + 1e-5
        assert len(edge.line.witnesses) == 2
        assert not np.intersect1d(*edge.line.witnesses).size
    np.testing.assert_array_equal(xyz, before)


def test_volume_random_points_do_not_create_ransac_plane_wireframe():
    xyz = np.random.default_rng(123).uniform(0, 4, (1800, 3))
    lines = region_lines(xyz, np.ones(len(xyz), dtype=np.bool_))
    assert not lines


def test_noisy_registered_corner_uses_physical_surface_scale():
    xyz = corner() + np.random.default_rng(19).normal(0, 0.035, corner().shape)
    lines = region_lines(xyz, np.zeros(len(xyz), dtype=np.bool_))
    assert any(edge.kind == 2 for edge in lines)


def test_declared_core_limits_physical_normal_probes(monkeypatch: pytest.MonkeyPatch):
    from scipy.spatial import KDTree
    from usim_env_pipeline import edge_line_geometry

    xyz = corner()
    xyz[:, 0] -= 1.5
    low, high = np.zeros(3), np.array([3.0, 4.0, 4.0])
    probes = []

    def tracked_tree(points: NDArray[np.float64], leafsize: int) -> KDTree:
        tree = KDTree(points, leafsize=leafsize)
        original = tree.query_ball_point

        def tracked_query(centers: NDArray[np.float64], radius: float):
            probes.append(centers.copy())
            return original(centers, radius)

        monkeypatch.setattr(tree, 'query_ball_point', tracked_query)
        return tree

    monkeypatch.setattr(edge_line_geometry, 'KDTree', tracked_tree)
    region_lines(xyz, np.zeros(len(xyz), dtype=np.bool_), core_bounds=(low, high))
    assert probes
    for centers in probes:
        assert ((centers >= low) & (centers < high)).all()


def test_continuous_plane_crossing_tile_seams_creates_no_seam_line():
    x, y = np.meshgrid(np.arange(0.1, 12.01, 0.1), np.arange(0.1, 4.01, 0.1))
    xyz = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 1.0)))
    config = EdgeLineConfig(tile_size=4)
    lines = list(extract_edge_lines(xyz, np.zeros(len(xyz), dtype=np.bool_), config))
    assert not lines


@pytest.mark.parametrize('length', [0.8, 1.8])
def test_supported_corner_crossing_core_retains_its_full_length(length: float):
    xyz = corner()
    low, high = 4 - length / 2, 4 + length / 2
    xyz = xyz[(xyz[:, 0] >= low - 1e-8) & (xyz[:, 0] <= high + 1e-8)]
    config = EdgeLineConfig(tile_size=4)
    before = xyz.copy()
    lines = list(extract_edge_lines(xyz, np.zeros(len(xyz), dtype=np.bool_), config))
    assert len(lines) == 1
    np.testing.assert_allclose(lines[0].line.endpoints[:, 0], [low, high], atol=1e-5)
    np.testing.assert_allclose(lines[0].line.endpoints[:, 1:], 0, atol=1e-5)
    np.testing.assert_array_equal(xyz, before)


def test_shifted_core_origin_preserves_the_same_supported_corner():
    xyz = corner()
    xyz = xyz[(xyz[:, 0] >= 3.1 - 1e-8) & (xyz[:, 0] <= 4.9 + 1e-8)]
    support = LineSupport(min_length=1.5)
    base = EdgeLineConfig(support=support)
    moved = replace(base, tile_origin=(4.0, 4.0, 4.0))
    flags = np.zeros(len(xyz), dtype=np.bool_)
    original = list(extract_edge_lines(xyz, flags, base))
    shifted = list(extract_edge_lines(xyz, flags, moved))
    assert len(original) == len(shifted) == 1
    np.testing.assert_allclose(original[0].line.endpoints, shifted[0].line.endpoints, atol=1e-5)


def test_shifted_ownership_grid_does_not_change_noisy_face_discovery():
    xyz = corner() + np.random.default_rng(19).normal(0, 0.035, corner().shape)
    config = EdgeLineConfig()
    flags = np.zeros(len(xyz), dtype=np.bool_)
    original = list(extract_edge_lines(xyz, flags, config))
    shifted = list(extract_edge_lines(xyz, flags, replace(config, tile_origin=(4.0, 4.0, 4.0))))
    assert original and len(original) == len(shifted)
    for first, second in zip(original, shifted, strict=True):
        np.testing.assert_allclose(first.line.endpoints, second.line.endpoints, atol=1e-10)


def test_corner_missing_one_face_interval_does_not_bridge_gap():
    xyz = corner()
    xyz = xyz[~((xyz[:, 2] > 0) & (xyz[:, 0] > 2.4) & (xyz[:, 0] < 3.6))]
    config = EdgeLineConfig(support=LineSupport(min_length=0.75))
    lines = region_lines(xyz, np.zeros(len(xyz), dtype=np.bool_), config)
    assert len(lines) >= 2
    for edge in lines:
        low, high = edge.line.endpoints[:, 0].min(), edge.line.endpoints[:, 0].max()
        assert high <= 2.45 or low >= 3.55


def test_finite_cache_uses_original_ids_and_preserves_attributes(tmp_path: Path):
    original = corner()
    xyz = np.vstack(([[-10.0, -10, -10]], original))
    source = save_records(tmp_path / 'records.npy', xyz)
    ids = np.arange(1, len(xyz), dtype=np.int64)
    np.save(tmp_path / 'ids.npy', ids)
    np.save(tmp_path / 'flags.npy', np.zeros(len(ids), dtype=np.uint8))
    before = sha256(source)
    result = compute_edge_lines(
        source,
        tmp_path / 'ids.npy',
        tmp_path / 'cache',
        boundary_flags_path=tmp_path / 'flags.npy',
    )
    assert len(result.segments) > 0 and not result.cached
    np.testing.assert_array_equal(result.source_ids, ids)
    np.testing.assert_array_equal(result.xyz, original)
    assert np.isin(result.witness_ids, ids).all()
    assert not result.source_ids.flags.writeable
    assert not result.segments.flags.writeable
    assert not result.bin_origins.flags.writeable
    assert sha256(source) == before
    repeated = compute_edge_lines(
        source,
        tmp_path / 'ids.npy',
        tmp_path / 'cache',
        boundary_flags_path=tmp_path / 'flags.npy',
    )
    assert repeated.cached
    np.testing.assert_array_equal(repeated.segments, result.segments)
    np.testing.assert_array_equal(repeated.bin_origins, result.bin_origins)


def test_line_support_change_uses_new_cache(tmp_path: Path):
    xyz = corner()
    source = save_records(tmp_path / 'records.npy', xyz)
    np.save(tmp_path / 'ids.npy', np.arange(len(xyz), dtype=np.int64))
    np.save(tmp_path / 'flags.npy', np.zeros(len(xyz), dtype=np.uint8))
    first = compute_edge_lines(
        source,
        tmp_path / 'ids.npy',
        tmp_path / 'cache',
        boundary_flags_path=tmp_path / 'flags.npy',
    )
    second = compute_edge_lines(
        source,
        tmp_path / 'ids.npy',
        tmp_path / 'cache',
        boundary_flags_path=tmp_path / 'flags.npy',
        config=replace(EdgeLineConfig(), support=LineSupport(min_length=2)),
    )
    assert first.cache_dir != second.cache_dir and not second.cached
