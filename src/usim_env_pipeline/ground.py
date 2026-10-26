"""Source-vertex ground triangulation, separate from immutable measured BPA surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .models import Payload, Support
from .storage import tile_name
from .usd import Geometry, LayerMetadata, author_geometry

if TYPE_CHECKING:
    from pxr import Usd


@dataclass(frozen=True, slots=True)
class GroundSurface:
    ids: NDArray[np.int64]
    faces: NDArray[np.int32]


def reconstruct_ground(
    points: NDArray[np.float32], heights: NDArray[np.float64], support: Support
) -> GroundSurface:
    """Triangulate the connected lower envelope using only exact original vertices.

    Disconnected elevated floors are not terrain. XY support gates retain holes;
    no extrapolation, smoothing, or invented plane is used.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import Delaunay, cKDTree

    ids = np.flatnonzero((heights >= -0.15) & (heights <= 0.25))
    empty = GroundSurface(np.empty(0, dtype=np.int64), np.empty((0, 3), dtype=np.int32))
    if len(ids) < 3:
        return empty
    # One actual representative near each small XY cell's median height.
    _, groups = np.unique(np.floor(points[ids, :2] / 0.15), axis=0, return_inverse=True)
    order = np.lexsort((points[ids, 2], groups))
    _, starts, counts = np.unique(groups[order], return_index=True, return_counts=True)
    ids = ids[order[starts + counts // 2]]
    positions = points[ids].astype(np.float64)

    # Use coarse connectivity only to reject detached roofs, never to create faces.
    cells, groups = np.unique(np.floor(positions[:, :2] / 2), axis=0, return_inverse=True)
    order = np.lexsort((positions[:, 2], groups))
    _, starts, counts = np.unique(groups[order], return_index=True, return_counts=True)
    levels = positions[order[starts + counts // 2], 2]
    neighbors = np.asarray(list(cKDTree(cells).query_pairs(2.1)), dtype=np.int64).reshape(-1, 2)
    if len(neighbors):
        neighbors = neighbors[np.abs(levels[neighbors[:, 0]] - levels[neighbors[:, 1]]) <= 1.0]
    graph = coo_matrix(
        (np.ones(len(neighbors)), (neighbors[:, 0], neighbors[:, 1])),
        shape=(len(cells), len(cells)),
    )
    _, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    component = int(np.argmax(sizes))
    ids, positions = ids[labels[groups] == component], positions[labels[groups] == component]
    if len(ids) < 3 or np.linalg.matrix_rank(positions[:, :2] - positions[0, :2]) < 2:
        return empty
    faces = Delaunay(positions[:, :2]).simplices.copy()
    triangles = positions[faces]
    edges = triangles[:, [1, 2, 0]] - triangles
    lengths = np.linalg.norm(edges, axis=2)
    cross = np.cross(edges[:, 0], -edges[:, 2])
    area = np.linalg.norm(cross, axis=1)
    keep = (area > support.min_double_area) & (lengths.max(axis=1) <= support.max_edge)
    keep &= np.abs(cross[:, 2]) >= 0.85 * area
    tree = cKDTree(positions)
    midpoints = (triangles + triangles[:, [1, 2, 0]]) * 0.5
    distances = tree.query(midpoints.reshape(-1, 3), workers=1)[0].reshape(-1, 3)
    keep &= distances.max(axis=1) <= support.midpoint_distance
    keep &= tree.query(triangles.mean(axis=1), workers=1)[0] <= support.centroid_distance
    faces = faces[keep]
    reverse = cross[keep, 2] < 0
    faces[reverse] = faces[reverse][:, [0, 2, 1]]
    used, inverse = np.unique(faces, return_inverse=True)
    return GroundSurface(ids[used], inverse.reshape(-1, 3).astype(np.int32))


def author_ground(
    world: Usd.Stage,
    output: Path,
    points: NDArray[np.float32],
    surface: GroundSurface,
    tile_size: float,
    metadata: LayerMetadata,
) -> list[Payload]:
    """Partition global triangles by half-open centroid ownership, preserving seam vertices."""
    from pxr import UsdGeom

    if not len(surface.faces):
        return []
    positions = points[surface.ids]
    owners = np.floor(
        positions[surface.faces].mean(axis=1, dtype=np.float64)[:, :2] / tile_size
    ).astype(int)
    keys, groups = np.unique(owners, axis=0, return_inverse=True)
    folder = output / 'authored/ground'
    folder.mkdir(parents=True, exist_ok=True)
    UsdGeom.Xform.Define(world, '/World/Ground')
    entries: list[Payload] = []
    for number, key in enumerate(keys):
        name = tile_name((int(key[0]), int(key[1])))
        used, inverse = np.unique(surface.faces[groups == number], return_inverse=True)
        local = positions[used]
        author_geometry(
            folder / f'{name}.usdc',
            Geometry(
                local,
                surface.ids[used],
                np.tile(np.array([[0.30, 0.32, 0.33]], dtype=np.float32), (len(local), 1)),
                inverse.reshape(-1, 3).astype(np.int32),
            ),
            metadata,
        )
        path, payload = f'/World/Ground/{name}', f'authored/ground/{name}.usdc'
        UsdGeom.Xform.Define(world, path).GetPrim().GetPayloads().AddPayload(payload, '/Tile')
        entries.append(
            Payload(
                path=path,
                min=local.min(0).tolist(),
                max=local.max(0).tolist(),
                payload=payload,
            )
        )
    return entries
