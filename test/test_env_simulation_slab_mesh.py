"""Planar solids retain exact footprints with compact, conforming polygon faces."""

import numpy as np

from usim_env_pipeline.simulation_slab_mesh import slab_surface
from usim_env_pipeline.simulation_metrics import signed_volume


def volume(positions, faces):
    triangles = positions[faces] - positions.mean(0)
    return (
        np.einsum('ij,ij->i', triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum()
        / 6
    )


def test_dense_rectangle_becomes_sixteen_triangles_without_volume_loss():
    # Given ten thousand occupied planar cells.
    keys = np.indices((100, 100)).reshape(2, -1).T.astype(np.int64)
    # When creating a closed conforming slab rather than per-cell cube faces.
    positions, faces = slab_surface(keys, 0.1, 0.16)
    # Then polygon assembly compresses the mesh while retaining exact footprint volume.
    assert len(faces) == 16
    np.testing.assert_allclose(volume(positions, faces), 16, atol=1e-10)
    np.testing.assert_allclose(positions.min(0), [0, 0, -0.08])
    np.testing.assert_allclose(positions.max(0), [10, 10, 0.08])


def test_l_shape_splits_rectangle_junctions_into_closed_shared_edges():
    # Given coplanar runs with a T-junction and an unoccupied exterior notch.
    keys = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [0, 2]], np.int64) - [20, 30]
    # When merging polygons and extruding only exposed boundaries.
    positions, faces = slab_surface(keys, 0.1, 0.16)
    # Then every geometric edge is shared exactly twice and the notch adds no volume.
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert (counts == 2).all()
    np.testing.assert_allclose(volume(positions, faces), len(keys) * 0.1**2 * 0.16)
    triangles = positions[faces]
    assert (
        np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
            axis=1,
        )
        > 0
    ).all()


def test_inner_hole_and_separate_patch_are_not_filled_by_polygon_assembly():
    # Given a ring and a disconnected measured patch.
    keys = np.array(
        [[x, y] for x in range(5) for y in range(5) if (x, y) != (2, 2)] + [[9, 0]],
        np.int64,
    )
    # When emitting the closed slab's simplified planar surfaces.
    positions, faces = slab_surface(keys, 0.1, 0.16)
    # Then the exact occupied-cell volume is retained without filling the hole or gap.
    np.testing.assert_allclose(volume(positions, faces), len(keys) * 0.1**2 * 0.16)
    centers = positions[faces].mean(1)
    assert not ((centers[:, 0] > 0.55) & (centers[:, 0] < 0.85)).any()


def test_open_mesh_tile_volumes_share_one_global_origin():
    # Given a translated closed slab published as three individually open mesh parts.
    positions, faces = slab_surface(np.array([[0, 0]], np.int64), 0.1, 0.16)
    positions += [2, 3, 4]
    # When summing signed tile contributions without recentering each open tile.
    total = sum(signed_volume(positions, part) for part in np.array_split(faces, 3))
    # Then the contributions recover the closed mesh's exact declared volume.
    np.testing.assert_allclose(total, 0.1**2 * 0.16, atol=1e-10)
