"""Continuous smoothed CPU terrain, independent of source meshes and USD."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .models import PipelineInputError
from .simulation_ground_config import GroundConfig as GroundConfig


@dataclass(frozen=True, slots=True)
class Terrain:
    positions: NDArray[np.float32]
    faces: NDArray[np.int32]
    keys: NDArray[np.int64]
    cells: NDArray[np.int64]
    corner_indices: NDArray[np.int32]
    support_ids: NDArray[np.int64]
    smoothing_strength: float = 1.0


def reconstruct_terrain(
    points: NDArray[np.float32],
    ids: NDArray[np.int64],
    config: GroundConfig = GroundConfig(),
) -> Terrain:
    """Fill the measured floor's convex XY domain with a continuous height field.

    Fits and queries run in bounded vectorized batches. Input IDs are sorted before
    building the tree, making neighbor selection independent of caller ID order.
    Local fits anchor the smooth surface; unsupported nodes are explicitly inferred,
    not omitted. This is a single-valued height field, not overlapping sheets.
    """
    from scipy.spatial import ConvexHull, Delaunay

    from .simulation_ground_fill import smooth_heights
    from .simulation_ground_fit import fit_nodes

    empty = Terrain(
        np.empty((0, 3), np.float32),
        np.empty((0, 3), np.int32),
        np.empty((0, 2), np.int64),
        np.empty((0, 2), np.int64),
        np.empty((0, 4), np.int32),
        np.empty((0, config.neighbors), np.int64),
    )
    if points.ndim != 2 or points.shape[1] != 3 or ids.ndim != 1:
        raise PipelineInputError('Ground requires XYZ points and one-dimensional source IDs')
    if ids.dtype.kind not in 'iu' or np.any(ids < 0) or np.any(ids >= len(points)):
        raise PipelineInputError('Ground source IDs must be valid integer point indices')
    ids = np.unique(ids).astype(np.int64)
    if len(ids) < config.min_neighbors:
        return empty
    measured = points[ids].astype(np.float64)
    if not np.isfinite(measured).all():
        raise PipelineInputError('Measured ground support must be finite')

    centered = measured[:, :2] - measured[:, :2].mean(0)
    if np.linalg.matrix_rank(centered) < 2:
        return empty
    measured_cells = np.unique(np.floor(measured[:, :2] / config.cell).astype(np.int64), axis=0)
    offsets = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.int64)
    footprint = (measured_cells[:, None] + offsets).reshape(-1, 2) * config.cell
    hull = ConvexHull(footprint)
    lower = measured_cells.min(0)
    upper = measured_cells.max(0) + 1
    grid = np.stack(
        np.meshgrid(
            np.arange(lower[0], upper[0]),
            np.arange(lower[1], upper[1]),
            indexing='ij',
        ),
        axis=-1,
    ).reshape(-1, 2)
    inside = Delaunay(footprint[hull.vertices]).find_simplex((grid + 0.5) * config.cell) >= 0
    cells = grid[inside]
    if not len(cells):
        return empty
    keys, inverse = np.unique(
        (cells[:, None] + offsets).reshape(-1, 2), axis=0, return_inverse=True
    )
    corners = inverse.reshape(-1, 4)
    xy = keys.astype(np.float64) * config.cell
    heights, provenance = fit_nodes(measured, ids, xy, config)
    anchored = np.isfinite(heights)
    if anchored.sum() < config.min_neighbors:
        return empty
    provenance[~anchored] = -1
    heights, strength = smooth_heights(xy, corners, heights, config)
    # Two upward triangles split along (0, 0)--(1, 1).
    triangle_corners = np.array([[0, 1, 2], [0, 2, 3]])
    corners = corners.astype(np.int32)
    positions = np.column_stack((xy, heights)).astype(np.float32)
    faces = corners[:, triangle_corners].reshape(-1, 3).astype(np.int32)
    return Terrain(positions, faces, keys, cells, corners, provenance, strength)


def terrain_heights(
    xy: NDArray[np.float32] | NDArray[np.float64],
    terrain: Terrain,
    config: GroundConfig = GroundConfig(),
) -> NDArray[np.float64]:
    """Evaluate the exact triangular height field; uncovered positions are NaN.

    Closed cell edges may belong to either accepted neighbor. Searchsorted uses
    lexicographic global integer keys, including negative lattice coordinates.
    """
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise PipelineInputError('Terrain height queries require XY positions')
    result = np.full(len(xy), np.nan, np.float64)
    if not len(terrain.cells):
        return result
    finite = np.isfinite(xy).all(axis=1)
    scaled = np.where(finite[:, None], xy, 0).astype(np.float64) / config.cell
    base = np.floor(scaled).astype(np.int64)
    dtype = np.dtype([('x', np.int64), ('y', np.int64)])
    cells = np.ascontiguousarray(terrain.cells).view(dtype).ravel()
    # The lattice is sorted by construction. At exact outer edges the floor
    # cell may be absent, so also inspect its three closed-edge neighbors.
    for offset in ((0, 0), (-1, 0), (0, -1), (-1, -1)):
        candidate = base + offset
        index = np.searchsorted(cells, np.ascontiguousarray(candidate).view(dtype).ravel())
        safe = np.minimum(index, len(cells) - 1)
        uv = scaled - candidate
        found = finite & np.isnan(result) & (index < len(cells))
        found &= (terrain.cells[safe] == candidate).all(axis=1)
        found &= ((uv >= 0) & (uv <= 1)).all(axis=1)
        u, v = uv[found].T
        z = terrain.positions[terrain.corner_indices[safe[found]], 2].astype(np.float64)
        result[found] = np.where(
            v <= u,
            z[:, 0] * (1 - u) + z[:, 1] * (u - v) + z[:, 2] * v,
            z[:, 0] * (1 - v) + z[:, 2] * u + z[:, 3] * (v - u),
        )
    return result
