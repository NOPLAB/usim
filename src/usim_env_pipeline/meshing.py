"""Source-vertex ball pivoting with measured edge/tangent/interior support gates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .models import MapConfig, PipelineInputError
from .pcd import normals, xyz


@dataclass(frozen=True, slots=True)
class Neighborhood:
    records: NDArray[np.void]
    ids: NDArray[np.int64]
    key: tuple[int, int]


@dataclass(frozen=True, slots=True)
class Surface:
    positions: NDArray[np.float32]
    ids: NDArray[np.int64]
    faces: NDArray[np.int32]
    record_indices: NDArray[np.int64]
    normals: NDArray[np.float64]
    estimated_ids: NDArray[np.int64]
    candidate_owned: int
    rejected_support: int
    horizontal_triangles: int


def reconstruct(neighborhood: Neighborhood, config: MapConfig) -> Surface:
    import open3d as o3d
    from scipy.spatial import cKDTree

    positions = xyz(neighborhood.records)
    working, valid = normals(neighborhood.records)
    estimated = np.zeros(len(positions), dtype=bool)
    match config.normal_policy:
        case 'auto':
            if (~valid).any() and len(positions) >= 3:
                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(positions.astype(np.float64))
                radius = 2 * max(config.support.radii)
                cloud.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
                )
                derived = np.asarray(cloud.normals)
                support_count = cKDTree(positions).query_ball_point(
                    positions, radius, return_length=True, workers=1
                )
                estimated = (~valid) & (support_count >= 3) & np.isfinite(derived).all(axis=1)
                working[estimated] = derived[estimated]
                valid |= estimated
        case 'supplied':
            pass
        case unreachable:
            from typing_extensions import assert_never

            assert_never(unreachable)
    # Source normal fields remain untouched; orient only working plane directions.
    dominant = np.argmax(np.abs(working), axis=1)
    negative = working[np.arange(len(working)), dominant] < 0
    working[negative] *= -1
    mesh_positions, mesh_normals = positions[valid], working[valid]
    candidates = np.empty((0, 3), dtype=np.int64)
    if len(mesh_positions) >= 3:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(mesh_positions.astype(np.float64))
        cloud.normals = o3d.utility.Vector3dVector(mesh_normals)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            cloud, o3d.utility.DoubleVector(config.support.radii)
        )
        if not np.array_equal(np.asarray(mesh.vertices), mesh_positions):
            raise PipelineInputError('Ball pivoting changed source vertices')
        candidates = np.asarray(mesh.triangles).copy()
    triangles = mesh_positions[candidates].astype(np.float64)
    low = np.array(neighborhood.key) * config.tile_size
    owned = (
        (triangles.mean(axis=1)[:, :2] >= low)
        & (triangles.mean(axis=1)[:, :2] < low + config.tile_size)
    ).all(axis=1)
    candidates, triangles = candidates[owned], triangles[owned]
    edges = triangles[:, [1, 2, 0]] - triangles
    lengths = np.linalg.norm(edges, axis=2)
    cross = np.cross(edges[:, 0], -edges[:, 2])
    area = np.linalg.norm(cross, axis=1)
    face_normal = cross / np.maximum(area[:, None], 1e-30)
    tri_normals = mesh_normals[candidates]
    support = config.support
    keep = (area > support.min_double_area) & (lengths.max(axis=1) <= support.max_edge)
    keep &= (
        np.einsum('fi,fvi->fv', face_normal, tri_normals).min(axis=1)
        >= support.min_face_normal_dot
    )
    keep &= (
        np.einsum('fvi,fvi->fv', tri_normals, tri_normals[:, [1, 2, 0]]).min(axis=1)
        >= support.min_pair_normal_dot
    )
    direction = edges / np.maximum(lengths[:, :, None], 1e-30)
    tangent = np.maximum(
        np.abs(np.einsum('fvi,fvi->fv', direction, tri_normals)),
        np.abs(np.einsum('fvi,fvi->fv', direction, tri_normals[:, [1, 2, 0]])),
    )
    keep &= tangent.max(axis=1) <= support.max_edge_normal_dot
    if len(candidates):
        tree = cKDTree(mesh_positions)
        midpoint = (triangles + triangles[:, [1, 2, 0]]) * 0.5
        distances = tree.query(midpoint.reshape(-1, 3), workers=1)[0].reshape(-1, 3)
        keep &= distances.max(axis=1) <= support.midpoint_distance
        keep &= tree.query(triangles.mean(axis=1), workers=1)[0] <= support.centroid_distance
    used, inverse = np.unique(candidates[keep], return_inverse=True)
    indices = np.flatnonzero(valid)[used]
    return Surface(
        mesh_positions[used],
        neighborhood.ids[indices],
        inverse.reshape(-1, 3).astype(np.int32),
        indices,
        mesh_normals[used],
        neighborhood.ids[estimated],
        len(candidates),
        int((~keep).sum()),
        int((np.abs(face_normal[keep, 2]) >= 0.7).sum()),
    )
