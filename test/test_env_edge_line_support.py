"""Fitted intersection lines require finite evidence from both unchanged faces."""

import numpy as np
from numpy.typing import NDArray
import pytest

from usim_env_pipeline.edge_line_support import (
    LineSupport,
    plane_intersection,
    supported_intervals,
)
from usim_env_pipeline.models import PipelineInputError


def strips(*, gap=False, missing_face=False):
    t = np.arange(0, 4.01, 0.05)
    if gap:
        t = t[(t < 1.4) | (t > 2.6)]
    floor = np.column_stack((t, np.full(len(t), 0.05), np.zeros(len(t))))
    wall = np.column_stack((t, np.zeros(len(t)), np.full(len(t), 1.0 if missing_face else 0.05)))
    xyz = np.vstack((floor, wall)).astype(np.float64)
    groups = (np.arange(len(t), dtype=np.int64), np.arange(len(t), 2 * len(t), dtype=np.int64))
    return xyz, groups


def test_perpendicular_planes_have_actual_intersection_not_box_edge():
    line = plane_intersection(np.array([0.0, 0, 1, -2]), np.array([0.0, 1, 0, -3]))
    assert line is not None
    origin, direction = line
    np.testing.assert_allclose(origin, [0, 3, 2])
    np.testing.assert_allclose(direction, [1, 0, 0])


def test_parallel_surfaces_do_not_create_feature_line():
    assert plane_intersection(np.array([0.0, 0, 1, 0]), np.array([0.0, 0, 1, -2])) is None


def test_finite_line_stays_within_both_faces_and_preserves_xyz():
    xyz, groups = strips()
    before = xyz.copy()
    lines = supported_intervals(xyz, groups, np.zeros(3), np.array([1.0, 0, 0]))
    assert len(lines) == 1
    np.testing.assert_allclose(lines[0].endpoints, [[0, 0, 0], [4, 0, 0]])
    np.testing.assert_array_equal(xyz, before)
    assert all(len(ids) >= 12 for ids in lines[0].witnesses)


def test_intersection_without_nearby_second_face_is_rejected():
    xyz, groups = strips(missing_face=True)
    lines = supported_intervals(xyz, groups, np.zeros(3), np.array([1.0, 0, 0]))
    assert not lines


def test_unobserved_gap_splits_line_instead_of_bridge():
    xyz, groups = strips(gap=True)
    lines = supported_intervals(
        xyz,
        groups,
        np.zeros(3),
        np.array([1.0, 0, 0]),
        LineSupport(min_length=1),
    )
    assert len(lines) == 2
    assert lines[0].endpoints[1, 0] < 1.4
    assert lines[1].endpoints[0, 0] > 2.6


def test_tiny_colocated_cluster_cannot_back_extended_line():
    xyz = np.tile([0.0, 0.05, 0], (100, 1))
    lines = supported_intervals(
        xyz, (np.arange(100, dtype=np.int64),), np.zeros(3), np.array([1.0, 0, 0])
    )
    assert not lines


def test_rotated_translated_support_produces_rotated_translated_line():
    xyz, groups = strips()
    rotate: NDArray[np.float64] = np.array(
        [[0.36, 0.48, -0.8], [-0.8, 0.6, 0], [0.48, 0.64, 0.6]], dtype=np.float64
    )
    shift = np.array([143, -89, 21])
    expected = supported_intervals(xyz, groups, np.zeros(3), np.array([1.0, 0, 0]))
    actual = supported_intervals(xyz @ rotate + shift, groups, shift, rotate[0])
    assert len(actual) == 1
    np.testing.assert_allclose(actual[0].endpoints, expected[0].endpoints @ rotate + shift)


@pytest.mark.parametrize('seed', range(10))
def test_sparse_witness_roundtrip_retains_the_original_bin_phase(seed: int):
    rng = np.random.default_rng(seed)
    first = np.column_stack((np.sort(rng.uniform(0, 3, 120)), np.full(120, 0.05), np.zeros(120)))
    second = np.column_stack((np.sort(rng.uniform(0, 3, 120)), np.zeros(120), np.full(120, 0.05)))
    xyz = np.vstack((first, second))
    groups = (np.arange(120, dtype=np.int64), np.arange(120, 240, dtype=np.int64))
    direction = np.array([1.0, 0, 0])
    lines = supported_intervals(xyz, groups, np.zeros(3), direction)
    assert lines
    for line in lines:
        fresh = supported_intervals(
            xyz, line.witnesses, line.endpoints[0], direction, bin_origin=line.bin_origin
        )
        assert len(fresh) == 1
        np.testing.assert_allclose(fresh[0].endpoints, line.endpoints, atol=1e-10, rtol=0)
        assert fresh[0].coverage == pytest.approx(line.coverage)
        assert fresh[0].max_unsupported_gap == pytest.approx(line.max_unsupported_gap)


def test_invalid_support_parameters_rejected():
    with pytest.raises(PipelineInputError):
        LineSupport(max_gap=0.01)
