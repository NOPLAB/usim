"""Actual geometry reduction preserves seams, contact patches and disconnected sheets."""

import numpy as np
import pytest

pytest.importorskip('scipy')
pytest.importorskip('skimage')
pytest.importorskip('pymeshlab')
o3d = pytest.importorskip('open3d')
poly = pytest.importorskip('usim_env_pipeline.simulation_polyhedron')


def sphere():
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    return np.asarray(mesh.vertices).astype(np.float32), np.asarray(mesh.triangles).astype(
        np.int32
    )


def test_closed_sphere_is_reduced_to_actual_planar_faces_without_losing_topology():
    # Given a finely sampled closed curved surface.
    positions, faces = sphere()
    # When decimating the connected interior without splitting its topology.
    result = poly.polyhedron(positions, faces, np.zeros(len(faces), bool), 0.35)
    # Then geometry itself is smaller, remains closed, and gains no nonmanifold edges.
    assert len(result.surface.faces) < len(faces) / 2
    assert poly.topology(result.surface) == (1, 2, 0)
    assert len(poly.boundary_edges(result.surface.faces)) == 0


def test_contact_triangles_are_bit_identical_after_reduction():
    # Given a contact patch at the bottom of a closed body.
    positions, faces = sphere()
    protected = positions[faces].min(1)[:, 2] < -0.9
    # When simplifying the body while protecting complete contact triangles.
    result = poly.polyhedron(positions, faces, protected, 0.35)
    # Then the exact triangles witnessing support remain in the reduced body.
    actual = set(map(tuple, result.surface.positions[result.surface.faces].reshape(-1, 9)))
    expected = set(map(tuple, positions[faces[protected]].reshape(-1, 9)))
    assert expected <= actual
    assert len(result.surface.faces) < len(faces)


def test_open_tile_boundary_edges_remain_exact_not_merely_nearby():
    # Given a hemisphere with an actual open interface edge ring.
    positions, faces = sphere()
    faces = faces[(positions[faces][:, :, 0] >= -0.00001).all(1)]
    reference = poly.CurveSurface(positions, faces)
    # When explicitly forbidding boundary vertex deletion.
    result = poly.polyhedron(positions, faces, np.zeros(len(faces), bool), 0.35)
    # Then complete boundary edge coordinates and their segmentation are retained.
    assert poly.boundary_coordinates(reference) == poly.boundary_coordinates(result.surface)
    assert len(result.surface.faces) < len(faces)


def test_nearby_disconnected_sheets_do_not_merge_within_one_bin():
    # Given two disconnected closed bodies occupying some of the same coarse bins.
    positions, faces = sphere()
    positions *= 0.1
    second = positions + [0.25, 0, 0]
    combined = np.concatenate((positions, second)).astype(np.float32)
    triangles = np.concatenate((faces, faces + len(positions)))
    # When simplifying each original component independently.
    result = poly.polyhedron(combined, triangles, np.zeros(len(triangles), bool), 0.35)
    # Then neither body disappears or connects to the other.
    assert poly.topology(result.surface) == (2, 4, 0)


def test_torus_passage_stays_open_in_actual_triangle_raycast():
    # Given a closed body with a real tunnel through its center.
    mesh = o3d.geometry.TriangleMesh.create_torus(
        torus_radius=1.0, tube_radius=0.3, radial_resolution=40, tubular_resolution=20
    )
    positions = np.asarray(mesh.vertices).astype(np.float32)
    faces = np.asarray(mesh.triangles).astype(np.int32)
    # When reducing the geometry, not replacing it with a global convex envelope.
    result = poly.polyhedron(positions, faces, np.zeros(len(faces), bool), 0.3)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(result.surface.positions),
        o3d.core.Tensor(result.surface.faces.astype(np.uint32)),
    )
    hits = scene.cast_rays(
        o3d.core.Tensor([[0, 0, 2, 0, 0, -1], [1, 0, 2, 0, 0, -1]], o3d.core.Dtype.Float32)
    )['t_hit'].numpy()
    # Then the tunnel is empty, its surrounding material remains, and topology is unchanged.
    assert np.isinf(hits[0]) and np.isfinite(hits[1])
    assert poly.topology(result.surface) == (1, 0, 0)
    assert len(result.surface.faces) < len(faces)


def test_irregular_closed_body_is_coarsened_without_discarding_safe_reduction():
    # Given a dense, irregular surface like the residual measurement envelopes.
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=40)
    positions = np.asarray(mesh.vertices).astype(np.float32)
    positions *= (1 + 0.15 * np.sin(positions[:, 0] * 13) * np.cos(positions[:, 2] * 17))[:, None]
    faces = np.asarray(mesh.triangles).astype(np.int32)
    # When reducing the interior while preserving the closed body's topology.
    result = poly.polyhedron(positions, faces, np.zeros(len(faces), bool), 0.3)
    # Then a coarse body replaces the dense surface instead of reverting it wholesale.
    assert len(result.surface.faces) < len(faces) / 4
    assert poly.topology(result.surface) == (1, 2, 0)


def test_original_nonmanifold_seam_is_fixed_while_regular_regions_are_coarsened():
    from usim_env_pipeline.simulation_curve_surface import quantized_surface

    # Given two closed bodies sharing one edge, like an existing pinched source seam.
    positions, faces = sphere()
    second = positions + [3, 0, 0]
    second[[0, 2]] = positions[[0, 2]]
    reference = quantized_surface(
        np.concatenate((positions, second)).astype(np.float32),
        np.concatenate((faces, faces + len(positions))),
    )
    protected = reference.positions[reference.faces].min(axis=1)[:, 2] < -0.9
    # When freezing nonmanifold neighborhoods before simplifying the remaining patches.
    result = poly.polyhedron(reference.positions, reference.faces, protected, 0.35)
    # Then the original exceptional seam is unchanged and safe regions actually become coarse.
    assert len(result.surface.faces) < len(reference.faces) / 2
    assert poly.topology(result.surface) == poly.topology(reference)
    assert poly.nonmanifold_edges(result.surface) == poly.nonmanifold_edges(reference)


@pytest.mark.parametrize('cell', [0, -1, float('nan'), float('inf')])
def test_invalid_clustering_width_is_rejected_at_the_boundary(cell):
    from usim_env_pipeline.models import PipelineInputError

    positions, faces = sphere()
    with pytest.raises(PipelineInputError):
        poly.polyhedron(positions, faces, np.zeros(len(faces), bool), cell)
