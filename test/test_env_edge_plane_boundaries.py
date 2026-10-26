"""Observed face neighborhoods, not seed clouds or hulls, support boundary lines."""

import numpy as np
from numpy.typing import NDArray
import pytest

from usim_env_pipeline.edge_line_support import LineSupport
from usim_env_pipeline.edge_plane_boundaries import observed_boundary_lines
from usim_env_pipeline.models import PipelineInputError
from usim_env_pipeline.plane_types import Plane


def rectangle(
    *, gap: bool = False
) -> tuple[NDArray[np.float64], Plane, NDArray[np.int64], NDArray[np.float64]]:
    x, y = np.meshgrid(np.arange(81) / 20, np.arange(61) / 20)
    local = np.column_stack((x.ravel(), y.ravel()))
    if gap:
        local = local[~((local[:, 1] == 0) & (local[:, 0] >= 1.4) & (local[:, 0] <= 2.6))]
    basis = np.array([[0.8, -0.36], [0.6, 0.48], [0.0, 0.8]])
    xyz = np.asarray(local @ basis.T + [2, -3, 1], dtype=np.float64)
    normal = np.cross(basis[:, 0], basis[:, 1])
    ids = np.arange(len(xyz), dtype=np.int64)
    seeds = ids[(local[:, 0] == 0) | (local[:, 0] == 4) | (local[:, 1] == 0) | (local[:, 1] == 3)]
    return xyz, Plane(normal, -float(xyz[0] @ normal), ids), seeds, basis


def test_oblique_face_has_only_observed_borders_not_diagonals_or_box_edges():
    xyz, plane, seeds, basis = rectangle()
    before = xyz.copy()
    lines = observed_boundary_lines(xyz, plane, seeds)
    assert len(lines) == 4
    for line in lines:
        endpoints = (line.endpoints - xyz[0]) @ basis
        delta = np.abs(endpoints[1] - endpoints[0])
        assert np.max(delta) > 2.4
        assert np.min(delta) < 1e-10
        fixed = int(np.argmin(delta))
        assert np.isclose(endpoints[0, fixed], 0) or np.isclose(endpoints[0, fixed], [4, 3][fixed])
        assert np.all(endpoints >= -1e-10)
        assert np.all(endpoints <= [4 + 1e-10, 3 + 1e-10])
        assert len(line.witnesses) == 1
        assert np.isin(line.witnesses[0], seeds).all()
        assert line.coverage >= LineSupport().min_coverage
        assert line.max_unsupported_gap <= LineSupport().max_gap
        assert not line.endpoints.flags.writeable
        assert not line.witnesses[0].flags.writeable
    np.testing.assert_array_equal(xyz, before)
    repeated = observed_boundary_lines(xyz, plane, seeds)
    for first, second in zip(lines, repeated):
        np.testing.assert_array_equal(first.endpoints, second.endpoints)
        np.testing.assert_array_equal(first.witnesses[0], second.witnesses[0])


def test_interior_only_fake_seeds_do_not_create_a_line_on_a_continuous_face():
    xyz, plane, _, basis = rectangle()
    local = (xyz - xyz[0]) @ basis
    seeds = np.flatnonzero(
        np.isclose(local[:, 1], 1.5) & (local[:, 0] > 0.5) & (local[:, 0] < 3.5)
    )
    assert len(seeds) > 40
    assert observed_boundary_lines(xyz, plane, seeds) == []


def test_missing_boundary_observations_split_instead_of_bridge():
    xyz, plane, seeds, basis = rectangle(gap=True)
    lines = observed_boundary_lines(xyz, plane, seeds, LineSupport(min_length=1))
    bottom = [
        (line.endpoints - xyz[0]) @ basis
        for line in lines
        if np.allclose(((line.endpoints - xyz[0]) @ basis)[:, 1], 0, atol=1e-10)
    ]
    assert len(bottom) == 2
    bottom.sort(key=lambda points: float(points[:, 0].min()))
    assert bottom[0][:, 0].max() < 1.4
    assert bottom[1][:, 0].min() > 2.6


def test_rotation_translation_preserve_supported_boundaries_and_original_rows():
    xyz, plane, seeds, _ = rectangle()
    rotate = np.array([[0.36, 0.48, -0.8], [-0.8, 0.6, 0], [0.48, 0.64, 0.6]])
    shift = np.array([143, -89, 21])
    moved = xyz @ rotate + shift
    normal = plane.normal @ rotate
    transformed = Plane(normal, plane.offset - float(shift @ normal), plane.ids)
    expected = observed_boundary_lines(xyz, plane, seeds)
    actual = observed_boundary_lines(moved, transformed, seeds)
    assert len(actual) == len(expected) == 4
    for line in expected:
        endpoints = line.endpoints @ rotate + shift
        matches = [
            candidate
            for candidate in actual
            if np.allclose(candidate.endpoints, endpoints)
            or np.allclose(candidate.endpoints[::-1], endpoints)
        ]
        assert len(matches) == 1
        np.testing.assert_array_equal(np.sort(matches[0].witnesses[0]), np.sort(line.witnesses[0]))
    np.testing.assert_array_equal(moved, xyz @ rotate + shift)


def test_coincident_observations_do_not_supply_eight_boundary_directions():
    xyz = np.repeat(np.array([[0.0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]]), 20, axis=0)
    ids = np.arange(len(xyz), dtype=np.int64)
    plane = Plane(np.array([0.0, 0, 1]), 0.0, ids)
    assert observed_boundary_lines(xyz, plane, ids, LineSupport(min_length=0.1)) == []


@pytest.mark.parametrize(
    'parameters',
    [
        {'neighborhood_radius': 0},
        {'neighborhood_radius': True},
        {'angle_threshold': float('nan')},
        {'angle_threshold': 361},
        {'tangent_tolerance': 91},
        {'fit_tolerance': -1},
        {'iterations': 0},
        {'iterations': 2.5},
        {'max_lines': False},
        {'seed': -1},
    ],
)
def test_invalid_parameters_raise_pipeline_input_error(parameters):
    xyz, plane, seeds, _ = rectangle()
    with pytest.raises(PipelineInputError):
        observed_boundary_lines(xyz, plane, seeds, **parameters)


def test_out_of_range_original_row_ids_are_rejected():
    xyz, plane, _, _ = rectangle()
    with pytest.raises(PipelineInputError):
        observed_boundary_lines(xyz, plane, np.array([len(xyz)], dtype=np.int64))
