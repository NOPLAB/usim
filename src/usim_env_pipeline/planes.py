"""Measured plane fitting and bounded plane-local repair without rewriting source points."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .plane_types import Patch, Plane, PlaneConfig, plane_basis


def refit(points: NDArray[np.float32] | NDArray[np.float64], ids: NDArray[np.int64]) -> Plane:
    sample = points[ids[:: max(1, len(ids) // 8000)]].astype(np.float64)
    centered = sample - sample.mean(0)
    normal = np.linalg.eigh(centered.T @ centered)[1][:, 0]
    if normal[np.argmax(np.abs(normal))] < 0:
        normal = -normal
    return Plane(normal, -float(np.median(sample @ normal)), ids)


def extract_planes(
    points: NDArray[np.float32], ids: NDArray[np.int64], config: PlaneConfig
) -> list[Plane]:
    """Deterministic regional RANSAC, refitted and bounded against original observations."""
    import open3d as o3d

    _, selected = np.unique(np.floor(points[ids] / config.voxel), axis=0, return_index=True)
    remaining = ids[selected]
    available = ids
    result: list[Plane] = []
    o3d.utility.random.seed(config.seed)
    for _ in range(config.max_planes):
        if len(remaining) < config.min_support:
            break
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points[remaining].astype(np.float64))
        model, inliers = cloud.segment_plane(config.tolerance, 3, config.iterations)
        if len(inliers) < config.min_support:
            break
        selected = np.ones(len(remaining), dtype=bool)
        selected[inliers] = False
        remaining = remaining[selected]
        model = np.asarray(model)
        members = available[np.abs(points[available] @ model[:3] + model[3]) <= config.tolerance]
        if len(members) < config.min_support:
            continue
        plane = refit(points, members)
        members = members[
            np.abs(points[members] @ plane.normal + plane.offset) <= config.tolerance
        ]
        if len(members) < config.min_support:
            continue
        plane = refit(points, members)
        members = members[
            np.abs(points[members] @ plane.normal + plane.offset) <= config.tolerance
        ]
        if len(members) < config.min_support:
            continue
        result.append(Plane(plane.normal, plane.offset, members))
        available = available[
            np.abs(points[available] @ plane.normal + plane.offset) > config.tolerance
        ]
    return result


def merge_planes(
    points: NDArray[np.float32], planes: list[Plane], config: PlaneConfig
) -> list[Plane]:
    """Merge nearby regional fits only when their combined measured support remains planar."""
    merged: list[Plane] = []
    for plane in sorted(planes, key=lambda item: len(item.ids), reverse=True):
        center = points[plane.ids].mean(0, dtype=np.float64)
        for index, other in enumerate(merged):
            if abs(plane.normal @ other.normal) < np.cos(np.deg2rad(2)):
                continue
            if abs(center @ other.normal + other.offset) > config.tolerance:
                continue
            ids = np.unique(np.concatenate((plane.ids, other.ids)))
            combined = refit(points, ids)
            residual = np.abs(points[ids] @ combined.normal + combined.offset)
            if np.quantile(residual, 0.95) > config.tolerance:
                continue
            ids = ids[residual <= config.tolerance]
            merged[index] = Plane(combined.normal, combined.offset, ids)
            break
        else:
            merged.append(plane)
    return merged


def patch_mesh(points: NDArray[np.float32], plane: Plane, config: PlaneConfig) -> Patch | None:
    """Close local discontinuities, protect observed recesses, then merge coplanar cell runs."""
    from .plane_repair import repair

    u, v = plane_basis(plane.normal)
    basis = np.column_stack((u, v))
    footprint = repair(points, plane, config)
    if footprint is None:
        return None
    lower, repaired = footprint.lower, footprint.mask
    shape = repaired.shape
    corners: list[tuple[int, int]] = []
    faces: list[tuple[int, int, int]] = []
    active: dict[tuple[int, int], int] = {}

    def rectangle(run: tuple[int, int], start: int, end: int) -> None:
        offset = len(corners)
        corners.extend(((start, run[0]), (end, run[0]), (end, run[1]), (start, run[1])))
        faces.extend(((offset, offset + 1, offset + 2), (offset, offset + 2, offset + 3)))

    for row in range(shape[0] + 1):
        edge = np.diff(np.pad(repaired[row].astype(int), 1)) if row < shape[0] else np.array([])
        runs = set(zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)))
        for run in active.keys() - runs:
            rectangle(run, active[run], row)
        active = {run: active.get(run, row) for run in runs}
    if not faces:
        return None
    coordinates = (np.asarray(corners) + lower) * config.cell
    positions = coordinates @ basis.T - plane.offset * plane.normal
    return Patch(
        plane,
        u,
        v,
        lower,
        repaired,
        footprint.observed,
        footprint.protected,
        footprint.enclosed,
        footprint.bridged,
        positions.astype(np.float32),
        np.asarray(faces, dtype=np.int32),
    )


def covered_faces(
    points: NDArray[np.float32], faces: NDArray[np.int32], patch: Patch, config: PlaneConfig
) -> NDArray[np.bool_]:
    triangles = points[faces]
    close = (
        np.abs(triangles @ patch.plane.normal + patch.plane.offset) <= config.tolerance * 2
    ).all(1)
    centers = triangles.mean(1)
    cells = (
        np.floor(np.column_stack((centers @ patch.u, centers @ patch.v)) / config.cell).astype(
            np.int64
        )
        - patch.lower
    )
    inside = ((cells >= 0) & (cells < np.asarray(patch.mask.shape))).all(1)
    selected = np.flatnonzero(close & inside)
    keep = np.zeros(len(faces), dtype=bool)
    keep[selected] = patch.mask[tuple(cells[selected].T)]
    return keep
