"""Native patch reduction with monotonic constraints on exceptional source connections."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .simulation_curve_surface import CurveSurface, quantized_surface


def nonmanifold_edges(surface: CurveSurface) -> set[tuple[float, ...]]:
    """Identify exceptional edges by their exact published coordinates."""
    edges = np.sort(surface.faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    return {
        tuple(value for point in sorted((tuple(first), tuple(second))) for value in point)
        for first, second in surface.positions[edges[counts > 2]]
    }


def decimate(
    reference: CurveSurface, protected_faces: NDArray[np.bool_], cell: float
) -> CurveSurface:
    """Freeze each newly exceptional neighborhood before attempting further reduction."""
    import open3d as o3d
    import pymeshlab

    fixed = protected_faces.copy()
    native_points = o3d.utility.Vector3dVector(reference.positions.astype(np.float64))
    native = o3d.geometry.TriangleMesh(native_points, o3d.utility.Vector3iVector(reference.faces))
    irregular = np.asarray(native.get_non_manifold_vertices(), np.int64)
    fixed |= np.isin(reference.faces, irregular).any(axis=1)
    original_exceptions = nonmanifold_edges(reference)
    while (~fixed).any():
        movable = reference.faces[~fixed]
        native = o3d.geometry.TriangleMesh(native_points, o3d.utility.Vector3iVector(movable))
        irregular = np.asarray(native.get_non_manifold_vertices(), np.int64)
        expanded = np.isin(reference.faces, irregular).any(axis=1)
        if (expanded & ~fixed).any():
            fixed |= expanded
            continue
        used, inverse = np.unique(movable, return_inverse=True)
        triangles = reference.positions[movable].astype(np.float64)
        area = (
            np.linalg.norm(
                np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
                axis=1,
            ).sum()
            * 0.5
        )
        meshes = pymeshlab.MeshSet()
        meshes.add_mesh(
            pymeshlab.Mesh(
                vertex_matrix=reference.positions[used].astype(np.float64),
                face_matrix=inverse.reshape(-1, 3).astype(np.int32),
            )
        )
        meshes.meshing_decimation_quadric_edge_collapse(
            targetfacenum=max(4, int(area / (cell * cell * 0.5))),
            preserveboundary=True,
            preservetopology=True,
            preservenormal=True,
            optimalplacement=False,
            planarquadric=True,
        )
        coarse = meshes.current_mesh()
        centers = np.asarray(coarse.vertex_matrix(), np.float32)
        candidate = quantized_surface(
            np.concatenate((centers, reference.positions)),
            np.concatenate(
                (
                    np.asarray(coarse.face_matrix(), np.int32),
                    reference.faces[fixed] + len(centers),
                )
            ),
        )
        new_exceptions = nonmanifold_edges(candidate) - original_exceptions
        if not new_exceptions:
            return candidate
        # Native topology preservation can still pinch a narrow measured connection.
        # Lock original neighborhoods at the offending coordinates rather than reverting
        # all safe reductions in this potentially very large component.
        faulty_points = np.asarray(sorted(new_exceptions), np.float32).reshape(-1, 3)
        _, labels = np.unique(
            np.concatenate((reference.positions, faulty_points)), axis=0, return_inverse=True
        )
        faulty = np.isin(labels[: len(reference.positions)], labels[len(reference.positions) :])
        neighbors = np.unique(reference.faces[faulty[reference.faces].any(axis=1)])
        expanded = np.isin(reference.faces, neighbors).any(axis=1)
        if not (expanded & ~fixed).any():
            return reference
        fixed |= expanded
    return reference
