"""Density cleanup distinguishes isolated returns from disconnected measured foliage."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pytest

pytest.importorskip('scipy')

from usim_env_pipeline.simulation_cleanup import density_cleanup
from usim_env_pipeline.simulation_obstacles import (
    ObstacleConfig,
    occupied_obstacles,
    voxel_surface,
)


def test_disconnected_foliage_is_kept_but_repeated_isolated_returns_are_removed():
    # Given separated leaves in a dense neighborhood and many returns in one distant cell.
    keys = np.array([[x, y, 20] for x in (0, 2, 4) for y in (0, 2, 4)])
    leaves = ((keys + 0.5) * 0.1).astype(np.float32)
    noise = np.repeat([[10.05, 0.05, 2.05]], 1000, axis=0).astype(np.float32)
    points = np.concatenate((leaves, noise))
    original = points.copy()
    config = ObstacleConfig(cell=0.1, cleanup_radius=0.31, cleanup_min_cells=4)
    occupied = occupied_obstacles(points, np.arange(len(points)), config)
    # When counting distinct nearby occupied cells, rather than returns or connectivity.
    cleaned = density_cleanup(occupied, config)
    kept = occupied.ids[cleaned.keep[occupied.groups]]
    # Then all leaves remain, while repetition cannot make an isolated cell legitimate.
    np.testing.assert_array_equal(kept, np.arange(len(leaves)))
    np.testing.assert_array_equal(points, original)


def test_thin_connected_column_keeps_its_endpoints_without_filling_surrounding_space():
    # Given a thin column with one measured return per fine cell.
    keys = np.array([[0, 0, z] for z in range(10)], dtype=np.int64)
    points = ((keys + 0.5) * 0.1).astype(np.float32)
    config = ObstacleConfig(cell=0.1, cleanup_radius=0.31, cleanup_min_cells=4)
    occupied = occupied_obstacles(points, np.arange(len(points)), config)
    # When cleaning its spatially supported occupancy and producing the actual boundary.
    cleaned = density_cleanup(occupied, config)
    positions, _faces, _owners = voxel_surface(occupied.keys[cleaned.keep], config.cell)
    # Then the column keeps its original extent and gains no surrounding voxels.
    assert cleaned.keep.all()
    np.testing.assert_allclose(positions.min(0), [0, 0, 0])
    np.testing.assert_allclose(positions.max(0), [0.1, 0.1, 1])


def test_cleanup_is_independent_of_source_record_order_and_repeated_scan_density():
    # Given the same occupied cells scanned in different orders and repetition counts.
    keys = np.array([[x, y, 1] for x in range(3) for y in range(3)] + [[40, 0, 1]])
    points = ((keys + 0.5) * 0.1).astype(np.float32)
    repeated = np.repeat(points[::-1], np.arange(1, len(points) + 1), axis=0)
    config = ObstacleConfig(cell=0.1, cleanup_radius=0.31, cleanup_min_cells=4)
    first = occupied_obstacles(points, np.arange(len(points)), config)
    second = occupied_obstacles(repeated, np.arange(len(repeated)), config)
    # When applying the same spatial density rule to both scans.
    first_result, second_result = density_cleanup(first, config), density_cleanup(second, config)
    # Then the removal decision depends only on measured occupied locations.
    np.testing.assert_array_equal(first.keys[first_result.keep], second.keys[second_result.keep])
    np.testing.assert_array_equal(first_result.support_counts, second_result.support_counts)


def test_empty_cleanup_has_no_fabricated_neighbors():
    # Given an empty obstacle partition.
    config = ObstacleConfig()
    occupied = occupied_obstacles(np.empty((0, 3), np.float32), np.empty(0, np.int64), config)
    # When cleaning the partition.
    cleaned = density_cleanup(occupied, config)
    # Then no cells or neighbor evidence are invented.
    assert cleaned.keep.shape == cleaned.support_counts.shape == (0,)
    assert cleaned.keep.dtype == np.bool_
    assert cleaned.support_counts.dtype == np.int64


def test_fine_voxels_bound_surface_overhang_to_a_tenth_of_a_meter_per_axis():
    # Given measured returns whose surfaces formerly extended by a coarse voxel width.
    points = np.array([[0.13, 0.17, 0.22], [0.24, 0.17, 0.22]], dtype=np.float32)
    config = ObstacleConfig()
    occupied = occupied_obstacles(points, np.arange(len(points)), config)
    # When generating the default measured occupied boundary.
    positions, _faces, _owners = voxel_surface(occupied.keys, config.cell)
    # Then the stair increments and all source-to-boundary extensions are at most 0.1 m.
    assert np.max(np.diff(np.unique(positions[:, 0]))) <= 0.100001
    assert (positions.min(0) >= points.min(0) - 0.100001).all()
    assert (positions.max(0) <= points.max(0) + 0.100001).all()


def cleanup_fixture(
    folder: Path, retain_noise: bool = False
) -> tuple[NDArray[np.float32], ObstacleConfig]:
    keys = np.array([[x, y, 20] for x in (0, 2, 4) for y in (0, 2, 4)] + [[100, 0, 20]])
    points = ((keys + 0.5) * 0.1).astype(np.float32)
    config = ObstacleConfig(cell=0.1, cleanup_radius=0.31, cleanup_min_cells=4)
    output = folder / 'simulation-models'
    output.mkdir()
    keep = np.array([True] * 9 + [retain_noise])
    np.savez(output / 'source-partition.npz', labels=np.where(keep, 3, 5).astype(np.uint8))
    np.savez(
        output / 'cleanup.npz',
        keys=keys,
        keep=keep,
        support_counts=np.array([4, 6, 4, 6, 9, 6, 4, 6, 4, 1], dtype=np.int64),
    )
    return points, config


def test_independent_cleanup_audit_checks_measured_density_and_removed_record_ownership(tmp_path):
    from usim_env_pipeline.simulation_verify_cleanup import verify_cleanup

    # Given explicit density evidence and an independently assigned isolated source return.
    points, config = cleanup_fixture(tmp_path)
    # When auditing against original source locations.
    removed = verify_cleanup(tmp_path, points, config)
    # Then exactly the isolated cell is accounted as removed.
    assert removed == 1


def test_independent_audit_rejects_noise_retention_even_if_source_labels_are_also_forged(tmp_path):
    from usim_env_pipeline.models import PipelineInputError
    from usim_env_pipeline.simulation_verify_cleanup import verify_cleanup

    # Given consistent but forged retention flags and source labels for an isolated return.
    points, config = cleanup_fixture(tmp_path, retain_noise=True)
    # When independently computing density rather than trusting the saved flags.
    with pytest.raises(PipelineInputError):
        # Then source-label consistency cannot certify physically unsupported noise.
        verify_cleanup(tmp_path, points, config)


def test_removed_noise_is_not_restored_by_retained_original_triangles(tmp_path):
    pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom

    from usim_env_pipeline.simulation_ground import GroundConfig, Terrain
    from usim_env_pipeline.simulation_output import publish
    from usim_env_pipeline.simulation_residual import publish_residual
    from usim_env_pipeline.simulation_verify_faces import verify_faces
    from usim_env_pipeline.storage import write_json
    from usim_env_pipeline.usd import Geometry, LayerMetadata, author_geometry, stage_base

    # Given one genuine unresolved triangle and another reaching a removed noisy vertex.
    points = np.array(
        [[0.05, 0.05, 0.05], [0.15, 0.05, 0.05], [0.05, 0.15, 0.05], [5.05, 0.05, 0.05]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 3, 2]], dtype=np.int32)
    metadata = LayerMetadata('CC-BY-4.0', 'fixture', '', 'XYZ', 'noise fixture')
    source = tmp_path / 'source.usdc'
    author_geometry(
        source, Geometry(points, np.arange(4), np.full((4, 3), 0.5, np.float32), faces), metadata
    )
    original_bytes = source.read_bytes()
    write_json(
        tmp_path / 'authored-index.json',
        [{'path': '/World/Regions/mixed', 'payload': 'source.usdc'}],
    )
    buildings = stage_base(tmp_path / 'buildings.usda', '/World', metadata)
    UsdGeom.Xform.Define(buildings, '/World/BuildingModels')
    buildings.GetRootLayer().Save()
    (tmp_path / 'building-models').mkdir()
    write_json(tmp_path / 'building-models/models.json', [])
    terrain = Terrain(
        np.empty((0, 3), np.float32),
        np.empty((0, 3), np.int32),
        np.empty((0, 2), np.int64),
        np.empty((0, 2), np.int64),
        np.empty((0, 4), np.int32),
        np.empty((0, 48), np.int64),
    )
    config = ObstacleConfig()
    occupied = occupied_obstacles(points, np.arange(3), config)
    noise_ids = np.array([3], dtype=np.int64)
    # When composing retained faces after explicitly removing the isolated noisy record.
    retained, total, replaced, discarded = publish_residual(
        tmp_path, terrain, occupied, GroundConfig(), config, metadata, noise_ids
    )
    tiles = publish(tmp_path, terrain, occupied, config.cell, metadata)
    stage = Usd.Stage.Open(str(tmp_path / 'simulation.usda'))
    for row in retained:
        UsdGeom.Xform.Define(stage, row['path']).GetPrim().GetPayloads().AddPayload(
            row['payload'], '/Tile'
        )
    stage.GetRootLayer().Save()
    write_json(tmp_path / 'simulation-models/index.json', tiles + retained)
    # Then noise-bearing triangles stay absent and the independent face audit agrees.
    assert (total, replaced, discarded) == (2, 0, 1)
    with np.load(tmp_path / 'simulation-models/residual/mixed.npz') as residual:
        np.testing.assert_array_equal(residual['source_face_indices'], [0])
    assert verify_faces(tmp_path, stage, noise_ids) == (2, 0, 1, 1)
    assert source.read_bytes() == original_bytes
