"""Supported primitives fill sampling defects without guessing large missing objects."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('scipy')

from usim_env_pipeline.simulation_planes import reconstruct_proxies
from usim_env_pipeline.simulation_primitive_detection import detect_primitives
from test.test_env_simulation_planes import archive, closed_volume


def wall_points(keys):
    uv = (np.asarray(keys) + 0.5) * 0.1
    return np.column_stack((np.full(len(uv), 2), uv)).astype(np.float32)


def test_small_sampling_hole_is_completed_but_large_opening_survives(tmp_path):
    # Given a planar wall with a small missing sample and a wide real opening.
    keys = [
        [y, z]
        for y in range(30)
        for z in range(20)
        if (y, z) != (5, 5) and not (15 <= y < 23 and 5 <= z < 13)
    ]
    points = wall_points(keys)
    archive(tmp_path, 0, [1, 0, 0], -2, np.arange(len(points)))
    # When fitting a bounded completed slab from the actual measurements.
    (proxy,) = reconstruct_proxies(
        points,
        np.ones(len(points), bool),
        np.empty(0, np.int64),
        tmp_path,
        cell=0.1,
        completion=True,
    )
    # Then missing sampling closes while the unsupported center remains empty.
    assert (proxy.keys == [5, 5]).all(1).any()
    assert not (proxy.keys == [19, 9]).all(1).any()
    np.testing.assert_array_equal(proxy.observed_keys, np.asarray(keys))
    assert proxy.completed
    assert closed_volume(proxy) == pytest.approx(len(proxy.keys) * 0.1**2 * 0.16, rel=1e-5)


def test_observed_recess_protects_a_small_wall_opening(tmp_path):
    # Given a missing wall cell with three original returns behind it.
    keys = [[y, z] for y in range(20) for z in range(20) if (y, z) != (10, 10)]
    wall = wall_points(keys)
    recess = wall_points([[10, 10]] * 3)
    recess[:, 0] -= 0.5
    points = np.concatenate((wall, recess))
    archive(tmp_path, 0, [1, 0, 0], -2, np.arange(len(wall)))
    # When repairing the observed support with bounded original depth evidence.
    (proxy,) = reconstruct_proxies(
        points,
        np.ones(len(points), bool),
        np.empty(0, np.int64),
        tmp_path,
        cell=0.1,
        completion=True,
    )
    # Then the supported recess remains an opening, not a solid patch.
    assert not (proxy.keys == [10, 10]).all(1).any()
    assert not np.isin(proxy.support_ids, np.arange(len(wall), len(points))).any()


def test_low_horizontal_structure_is_supported_without_becoming_ground(tmp_path):
    # Given a low planar top with nearby measured ground well below it.
    x, y = np.meshgrid(np.arange(0.05, 1.0, 0.1), np.arange(0.05, 1.0, 0.1))
    top = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 0.5))).astype(np.float32)
    points = np.concatenate((top, top - [0, 0, 0.5])).astype(np.float32)
    archive(tmp_path, 0, [0, 0, 1], -0.5, np.arange(len(top)))
    # When using the bounded low-structure approximation rather than roof-only rules.
    (proxy,) = reconstruct_proxies(
        points,
        np.r_[np.ones(len(top), bool), np.zeros(len(top), bool)],
        np.arange(len(top), len(points)),
        tmp_path,
        cell=0.1,
        completion=True,
        minimum_clearance=0.25,
    )
    # Then the top becomes a thin closed solid with its original support IDs.
    np.testing.assert_array_equal(proxy.support_ids, np.arange(len(top)))
    np.testing.assert_allclose(np.unique(proxy.positions[:, 2]), [0.42, 0.58], atol=1e-6)


def structural_points():
    x, z = np.meshgrid(np.arange(0.05, 1.05, 0.05), np.arange(0.35, 1.55, 0.05))
    first = np.column_stack((x.ravel(), np.full(x.size, 2), z.ravel()))
    second = first + [2, 0, 0]
    return np.concatenate((first, second)).astype(np.float32)


def test_additional_detection_finds_two_low_walls_without_bridging_the_gap(tmp_path):
    pytest.importorskip('open3d')
    # Given two low connected planar regions missed by the building size gates.
    points = structural_points()
    original = points.copy()
    # When extracting normal-supported connected primitive hypotheses.
    count = detect_primitives(points, np.ones(len(points), bool), tmp_path)
    # Then both regions remain separate and use only original record IDs.
    assert count == 2
    memberships = []
    for number in range(count):
        with np.load(tmp_path / f'plane_{number:04}.npz') as plane:
            memberships.append(plane['ids'])
            assert np.ptp(points[plane['ids'], 0]) < 1.1
            assert (
                np.abs(points[plane['ids']] @ plane['equation'][:3] + plane['equation'][3]) < 0.04
            ).all()
    assert not np.intersect1d(*memberships).size
    np.testing.assert_array_equal(points, original)


def test_volumetric_scatter_is_not_forced_into_planar_solids(tmp_path):
    pytest.importorskip('open3d')
    # Given isotropic dense irregular samples, not a measured planar sheet.
    axis = np.arange(0.05, 1.1, 0.1)
    points = np.stack(np.meshgrid(axis, axis, axis), axis=-1).reshape(-1, 3).astype(np.float32)
    # When proposing finite planar primitives from the residual cloud.
    count = detect_primitives(points, np.ones(len(points), bool), tmp_path)
    # Then normal/planarity evidence rejects accidental RANSAC slices.
    assert count == 0


def test_generated_archive_manifest_excludes_stale_plane_files(tmp_path):
    # Given one active archive and an unrelated stale archive in the same folder.
    points = wall_points([[0, 0], [1, 0]])
    archive(tmp_path, 0, [1, 0, 0], -2, [0])
    archive(tmp_path, 99, [1, 0, 0], -2, [1])
    # When passing the active generation's explicit source paths.
    proxies = reconstruct_proxies(
        points,
        np.ones(2, bool),
        np.empty(0, np.int64),
        tmp_path,
        cell=0.1,
        source_paths=[tmp_path / 'plane_0000.npz'],
    )
    # Then stale payloads do not silently reclaim current source points.
    assert len(proxies) == 1
    np.testing.assert_array_equal(proxies[0].support_ids, [0])
    assert proxies[0].source_path == Path(tmp_path / 'plane_0000.npz')
