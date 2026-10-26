"""Implicit shape fitting generates genuine curved envelopes with source-supported gaps."""

import numpy as np
import pytest

pytest.importorskip('scipy')
pytest.importorskip('skimage')

from usim_env_pipeline.simulation_curve_surface import (
    curve_surface,
    quantized_surface,
    supported_surface,
)
from usim_env_pipeline.simulation_metrics import signed_volume
from scipy.spatial import cKDTree


def test_concave_smoothing_overshoot_is_bounded_before_publication():
    # Given the support overshoot measured by the full-map independent audit.
    positions = np.array([[0.307854, 0, 0], [0, 0.1, 0], [0, 0, 0.1], [0, 0, 0]], np.float32)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], np.int32)
    support = cKDTree([[0, 0, 0]])
    assert support.query(quantized_surface(positions, faces).positions)[0].max() > 0.3
    # When fitting is bounded before float32 publication.
    surface = supported_surface(positions, faces, support)
    # Then the actual mesh is supported, closed and nondegenerate.
    assert support.query(surface.positions)[0].max() <= 0.3
    assert len(surface.faces) == 4
    assert abs(signed_volume(surface.positions, surface.faces)) > 0


def test_smoothed_cavity_stays_within_measured_support():
    # Given a dense object with an unsupported cavity inside it.
    indices = np.indices((19, 19, 19)).reshape(3, -1).T
    cavity = np.linalg.norm(indices - 9, axis=1) < 3.5
    keys = indices[~cavity].astype(np.int64)
    # When smoothing the curved cavity boundary.
    surface = curve_surface(keys, (0, 0), 0.1)
    # Then every actual float32 vertex remains within 0.3 m of measured support.
    distance = cKDTree((keys + 0.5) * 0.1).query(surface.positions)[0]
    assert distance.max() <= 0.3


def test_support_projection_agrees_when_neighbor_trees_order_ties_differently():
    # Given a seam vertex equally supported by two occupied cells.
    positions = np.array([[0, 0.3, 0], [0, 0.1, 0], [0, 0, 0.1]], np.float32)
    faces = np.array([[0, 1, 2]], np.int32)
    centers = np.array([[-0.1, 0, 0], [0.1, 0, 0]])
    # When different tile-local trees choose their nearest tied cells.
    first = supported_surface(positions, faces, cKDTree(centers))
    second = supported_surface(positions, faces, cKDTree(centers[::-1]))
    # Then publication cannot introduce cracks at their shared vertex.
    np.testing.assert_array_equal(first.positions, second.positions)


@pytest.mark.parametrize('origin', [[0, 0, 0], [-199.25, -48.45, 13.75]])
def test_rounding_a_split_edge_does_not_leave_zero_area_triangles(origin):
    # Given a closed tetrahedron with a tiny subdivision on one edge.
    positions = (
        np.array(
            [[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1], [1e-7, 0, 0]],
            np.float64,
        )
        + origin
    )
    faces = np.array([[0, 2, 4], [4, 2, 1], [0, 4, 3], [4, 1, 3], [0, 3, 2], [1, 2, 3]], np.int32)
    # When converting into the actual float32 world coordinates used for USD.
    surface = quantized_surface(positions, faces)
    # Then every triangle remains positive-area and the surface stays closed.
    triangles = surface.positions[surface.faces].astype(np.float64)
    area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    assert (area > 0).all()
    edges = np.sort(
        np.concatenate(
            (surface.faces[:, [0, 1]], surface.faces[:, [1, 2]], surface.faces[:, [2, 0]])
        ),
        axis=1,
    )
    assert (np.unique(edges, axis=0, return_counts=True)[1] == 2).all()
    assert signed_volume(surface.positions, surface.faces) > 0


def test_voxelized_sphere_becomes_a_continuous_curved_surface():
    # Given a quantized spherical object with stair-step voxel boundaries.
    indices = np.indices((21, 21, 21)).reshape(3, -1).T
    keys = indices[np.linalg.norm(indices - 10, axis=1) <= 7].astype(np.int64)
    # When fitting a smooth implicit envelope instead of moving cube vertices.
    surface = curve_surface(keys, (0, 0), 0.1)
    # Then the closed curved envelope has varied normals and no axis-aligned box shell.
    triangles = surface.positions[surface.faces].astype(np.float64)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    assert ((np.abs(normals).max(1) < 0.98).mean()) > 0.75
    assert signed_volume(surface.positions, surface.faces) > 0
    radius = np.linalg.norm(surface.positions - 1.05, axis=1)
    assert radius.std() < 0.04
    edges = np.sort(
        np.concatenate(
            (surface.faces[:, [0, 1]], surface.faces[:, [1, 2]], surface.faces[:, [2, 0]])
        ),
        axis=1,
    )
    assert (np.unique(edges, axis=0, return_counts=True)[1] == 2).all()


def test_large_opening_and_disconnected_objects_remain_separate():
    # Given two measured blocks separated by an unsupported one-metre passage.
    block = np.indices((5, 5, 5)).reshape(3, -1).T + [10, 10, 10]
    keys = np.concatenate((block, block + [20, 0, 0])).astype(np.int64)
    original = keys.copy()
    # When reconstructing their local curved envelopes.
    surface = curve_surface(keys, (0, 0), 0.1)
    # Then no face occupies the central passage and measurements are untouched.
    centers = surface.positions[surface.faces].mean(1)
    assert not ((centers[:, 0] > 1.9) & (centers[:, 0] < 2.5)).any()
    np.testing.assert_array_equal(keys, original)


def test_neighboring_tiles_share_bit_identical_boundary_vertices():
    # Given one supported object spanning a tile boundary.
    keys = (np.indices((15, 15, 15)).reshape(3, -1).T + [143, 10, 10]).astype(np.int64)
    # When fitting both core tiles with the same measured neighborhood.
    first, second = curve_surface(keys, (0, 0), 0.1), curve_surface(keys, (1, 0), 0.1)
    # Then vertices on their interface agree exactly without cracks.
    a = first.positions[np.isclose(first.positions[:, 0], 15.05)]
    b = second.positions[np.isclose(second.positions[:, 0], 15.05)]
    assert len(a) > 0
    np.testing.assert_array_equal(np.unique(a, axis=0), np.unique(b, axis=0))
