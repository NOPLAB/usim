"""Smoothing changes positions, never voxel topology or independently tiled seams."""

import numpy as np
import pytest

pytest.importorskip('scipy')

from usim_env_pipeline.simulation_obstacles import voxel_surface
from usim_env_pipeline.models import PipelineInputError
from usim_env_pipeline.simulation_surface_smoothing import smooth_surface


def block(size=4, cell=0.1):
    keys = np.indices((size, size, size)).reshape(3, -1).T.astype(np.int64)
    positions, faces, _ = voxel_surface(keys, cell)
    return positions, faces


def face_vectors(positions, faces):
    triangles = positions[faces].astype(np.float64)
    return np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])


def pinned_vertices(faces):
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    return np.unique(edges[counts != 2])


def test_block_rounds_corners_but_retains_planar_interior_and_displacement_bound():
    positions, faces = block(20)
    result = smooth_surface(positions, faces, 0.1)
    corner = np.flatnonzero((positions == 0).all(axis=1))[0]
    assert (result[corner] > positions[corner]).all()
    assert np.linalg.norm(result[corner] - positions[corner]) > 0.001
    center = np.flatnonzero(
        np.isclose(positions[:, 0], 0)
        & np.isclose(positions[:, 1], 1.0)
        & np.isclose(positions[:, 2], 1.0)
    )[0]
    np.testing.assert_allclose(result[center], positions[center], atol=1e-7)
    distance = np.linalg.norm(result.astype(np.float64) - positions, axis=1)
    assert distance.max() <= 0.02
    assert result.dtype == np.float32
    assert result.shape == positions.shape


def test_planar_open_tile_is_unchanged_including_all_boundary_vertices():
    keys = np.indices((7, 7, 1)).reshape(3, -1).T.astype(np.int64)
    positions, faces, _ = voxel_surface(keys, 0.1)
    top = np.isclose(positions[faces, 2], 0.1).all(axis=1)
    faces = faces[top]
    result = smooth_surface(positions, faces, 0.1)
    np.testing.assert_allclose(result, positions, atol=1e-7)
    boundary = pinned_vertices(faces)
    assert len(boundary) > 0
    np.testing.assert_array_equal(result[boundary], positions[boundary])


def test_bent_tile_moves_interior_but_pins_seams_bit_for_bit():
    keys = np.indices((7, 7, 1)).reshape(3, -1).T.astype(np.int64)
    positions, faces, _ = voxel_surface(keys, 0.1)
    faces = faces[np.isclose(positions[faces, 2], 0.1).all(axis=1)]
    center = np.isclose(positions[:, 0], 0.3) & np.isclose(positions[:, 1], 0.3)
    positions[center, 2] += 0.04
    boundary = pinned_vertices(faces)
    result = smooth_surface(positions, faces, 0.1)
    np.testing.assert_array_equal(result[boundary], positions[boundary])
    assert np.any(result[center] != positions[center])


def test_nonmanifold_edge_contacts_are_fixed():
    keys = np.array([[0, 0, 0], [1, 1, 0]], dtype=np.int64)
    positions, faces, _ = voxel_surface(keys, 0.1)
    boundary = pinned_vertices(faces)
    assert len(boundary) == 2
    result = smooth_surface(positions, faces, 0.1)
    np.testing.assert_array_equal(result[boundary], positions[boundary])
    assert np.any(result != positions)


def test_hole_and_separate_object_keep_empty_space_and_original_topology():
    ring = [[x, y, 0] for x in range(5) for y in range(5) if x != 2 or y != 2]
    keys = np.array(ring + [[9, 0, 0]], dtype=np.int64)
    positions, faces, _ = voxel_surface(keys, 0.1)
    original_faces = faces.copy()
    result = smooth_surface(positions, faces, 0.1)
    np.testing.assert_array_equal(faces, original_faces)
    assert len(result) == len(positions)
    # Even after the displacement cap, the separate cube cannot cross the street.
    left = positions[:, 0] <= 0.5
    assert result[left, 0].max() < result[~left, 0].min() - 0.3
    # The hole-facing triangles cannot reach the hole's central vertical axis.
    centers = result[faces].mean(axis=1)
    assert not (
        (np.abs(centers[:, 0] - 0.25) < 0.025) & (np.abs(centers[:, 1] - 0.25) < 0.025)
    ).any()


@pytest.mark.parametrize('sliver', [False, True])
def test_output_is_finite_nondegenerate_and_retains_original_face_orientation(sliver):
    positions, faces = block()
    if sliver:
        positions[1] = positions[0] + np.array([0, 0, 1e-6], dtype=np.float32)
    before = face_vectors(positions, faces)
    result = smooth_surface(positions, faces, 0.1)
    after = face_vectors(result, faces)
    assert np.isfinite(result).all()
    assert (np.linalg.norm(after, axis=1) > 0).all()
    assert (np.einsum('ij,ij->i', before, after) > 0).all()
    upward = (before[:, 2] > 0) & np.isclose(before[:, :2], 0).all(axis=1)
    assert upward.any()
    assert (after[upward, 2] > 0).all()
    assert np.linalg.norm(result.astype(np.float64) - positions, axis=1).max() <= 0.02


def test_inputs_are_immutable_and_result_is_deterministic():
    positions, faces = block()
    original_positions, original_faces = positions.copy(), faces.copy()
    positions.flags.writeable = faces.flags.writeable = False
    first = smooth_surface(positions, faces, 0.1)
    second = smooth_surface(positions, faces, 0.1)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(positions, original_positions)
    np.testing.assert_array_equal(faces, original_faces)
    assert not np.shares_memory(first, positions)


def test_empty_faces_and_unreferenced_vertices_are_preserved():
    positions = np.array([[1, 2, 3]], dtype=np.float32)
    faces = np.empty((0, 3), dtype=np.int32)
    np.testing.assert_array_equal(smooth_surface(positions, faces, 0.1), positions)
    empty = np.empty((0, 3), dtype=np.float32)
    assert smooth_surface(empty, faces, 0.1).shape == (0, 3)


def test_float32_rounding_at_large_coordinates_still_obeys_displacement_cap():
    positions, faces = block()
    positions += np.float32(100000)
    result = smooth_surface(positions, faces, 0.1)
    assert np.linalg.norm(result.astype(np.float64) - positions, axis=1).max() <= 0.02
    dot = np.einsum('ij,ij->i', face_vectors(positions, faces), face_vectors(result, faces))
    assert (dot > 0).all()


def test_degenerate_input_is_rejected_explicitly():
    positions = np.zeros((3, 3), dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    with pytest.raises(PipelineInputError, match='nonzero'):
        smooth_surface(positions, faces, 0.1)
