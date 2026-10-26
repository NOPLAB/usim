"""Ground reachability through actual triangle surfaces, not bounding-box proximity."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import open3d as o3d
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

Coordinates = NDArray[np.float32 | np.float64]


@dataclass(frozen=True, slots=True)
class MeshComponents:
    count: int
    face_parts: NDArray[np.int32]
    positions: Coordinates
    vertex_parts: NDArray[np.int32]


@dataclass(frozen=True, slots=True)
class TriangleChunks:
    order: NDArray[np.int64]
    starts: NDArray[np.int64]
    stops: NDArray[np.int64]
    owners: NDArray[np.int64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ContactGraph:
    owners: NDArray[np.int64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class GroundConnections:
    keep: NDArray[np.bool_]
    witnesses: NDArray[np.int64]
    checked_pairs: int


def mesh_components(positions: Coordinates, faces: NDArray[np.int32]) -> MeshComponents:
    """Label vertex-connected faces, welding only identical published coordinates."""
    used = np.unique(faces)
    points, inverse = np.unique(positions[used], axis=0, return_inverse=True)
    triangles = inverse[np.searchsorted(used, faces)]
    edges = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]]))
    graph = coo_matrix(
        (np.ones(len(edges), np.uint8), (edges[:, 0], edges[:, 1])),
        shape=(len(points), len(points)),
    )
    count, labels = connected_components(graph, directed=False)
    return MeshComponents(
        count,
        labels[triangles[:, 0]].astype(np.int32),
        points,
        labels.astype(np.int32),
    )


def weld_components(
    positions: Coordinates, owners: NDArray[np.int64], count: int
) -> tuple[int, NDArray[np.int32]]:
    """Reconnect tile seams and point contacts without rounding nearby positions."""
    order = np.lexsort((positions[:, 2], positions[:, 1], positions[:, 0]))
    points, parts = positions[order], owners[order]
    shared = (points[1:] == points[:-1]).all(1) & (parts[1:] != parts[:-1])
    first, second = parts[:-1][shared], parts[1:][shared]
    graph = coo_matrix((np.ones(len(first), np.uint8), (first, second)), shape=(count, count))
    return connected_components(graph, directed=False)


def triangle_chunks(
    positions: Coordinates,
    faces: NDArray[np.int32],
    owners: NDArray[np.int64],
    limit: int = 64,
) -> TriangleChunks:
    """Group nearby faces by owner, keeping each original triangle whole."""
    triangles = positions[faces].astype(np.float64)
    grid = np.floor(triangles.mean(1) / 0.5).astype(np.int64)
    order = np.lexsort((grid[:, 2], grid[:, 1], grid[:, 0], owners))
    sorted_grid, sorted_owners = grid[order], owners[order]
    changes = (sorted_owners[1:] != sorted_owners[:-1]) | (
        sorted_grid[1:] != sorted_grid[:-1]
    ).any(1)
    boundaries = np.r_[0, np.flatnonzero(changes) + 1, len(faces)]
    starts = np.concatenate(
        [
            np.arange(start, stop, limit, dtype=np.int64)
            for start, stop in zip(boundaries[:-1], boundaries[1:])
        ]
    )
    stops = np.minimum(
        starts + limit,
        np.repeat(boundaries[1:], np.ceil(np.diff(boundaries) / limit).astype(np.int64)),
    )
    low, high = triangles.min(1)[order], triangles.max(1)[order]
    return TriangleChunks(
        order,
        starts,
        stops,
        sorted_owners[starts],
        np.minimum.reduceat(low, starts),
        np.maximum.reduceat(high, starts),
    )


def native_mesh(positions: Coordinates, faces: NDArray[np.int32]) -> o3d.geometry.TriangleMesh:
    """Build only the referenced vertices of a bounded native contact chunk."""
    used, inverse = np.unique(faces, return_inverse=True)
    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(positions[used].astype(np.float64)),
        o3d.utility.Vector3iVector(inverse.reshape(-1, 3).astype(np.int32)),
    )


def native_intersection(
    first: tuple[Coordinates, NDArray[np.int32]],
    second: tuple[Coordinates, NDArray[np.int32]],
) -> bool:
    """Test actual interior/coplanar triangle contact with no deliberate gap allowance."""
    return bool(native_mesh(*first).is_intersecting(native_mesh(*second)))


def ground_connections(
    graph: ContactGraph,
    load: Callable[[int], tuple[Coordinates, NDArray[np.int32]]],
    component_count: int,
    ground_component: int,
) -> GroundConnections:
    """Traverse verified surface contacts; bounds only select possible contacts."""
    from shapely import STRtree, box

    bounds = box(graph.lower[:, 0], graph.lower[:, 1], graph.upper[:, 0], graph.upper[:, 1])
    tree = STRtree(bounds)

    @lru_cache(maxsize=256)
    def model(index: int) -> o3d.geometry.TriangleMesh:
        return native_mesh(*load(index))

    order = np.argsort(graph.owners, kind='stable')
    offsets = np.r_[0, np.cumsum(np.bincount(graph.owners, minlength=component_count))]
    keep = np.zeros(component_count, bool)
    keep[ground_component] = True
    pending = deque([ground_component])
    witnesses = []
    checked, processed = 0, 0
    while pending:
        owner = pending.popleft()
        for first in order[offsets[owner] : offsets[owner + 1]]:
            candidates = tree.query(bounds[first])
            candidates = candidates[~keep[graph.owners[candidates]]]
            candidates = candidates[
                (graph.upper[candidates] >= graph.lower[first]).all(1)
                & (graph.upper[first] >= graph.lower[candidates]).all(1)
            ]
            for second in np.sort(candidates):
                target = int(graph.owners[second])
                if keep[target]:
                    continue
                checked += 1
                if model(int(first)).is_intersecting(model(int(second))):
                    keep[target] = True
                    pending.append(target)
                    witnesses.append((int(first), int(second)))
            processed += 1
            if processed % 10000 == 0:
                print(
                    f'GROUND_CONNECTION_PROGRESS chunks={processed} '
                    f'reached={int(keep.sum())} checked_pairs={checked}',
                    flush=True,
                )
    return GroundConnections(keep, np.asarray(witnesses, np.int64).reshape(-1, 2), checked)
