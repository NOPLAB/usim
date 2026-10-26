"""Whole-face accounting: retain measured triangles not bounded by replacement models."""

from __future__ import annotations

from itertools import product

import numpy as np
from numpy.typing import NDArray

from .simulation_ground import GroundConfig, Terrain, terrain_heights


def occupied_bounds(
    lower: NDArray[np.int64], upper: NDArray[np.int64], keys: NDArray[np.int64]
) -> NDArray[np.bool_]:
    """Prove every grid cell in a face's bounding range is occupied.

    This conservative test cannot delete a face spanning an empty passage just
    because its vertices or centroid happen to land inside replacement geometry.
    Large ranges stay unresolved rather than consuming an unbounded query.
    """
    covered = np.ones(len(lower), dtype=bool)
    if not len(keys):
        return np.zeros(len(lower), dtype=bool)
    widths = upper - lower + 1
    bounded = (widths <= 4).all(1) & (widths.prod(1) <= 64)
    covered &= bounded
    origin = np.minimum(keys.min(0), lower.min(0)) - 1
    extent = np.maximum(keys.max(0), upper.max(0)) - origin + 2
    strides = np.cumprod(np.r_[1, extent[:0:-1]])[::-1].astype(np.int64)
    codes = np.sort((keys - origin) @ strides)
    limits = np.minimum(widths.max(0), 4)
    for offset in product(*(range(int(value)) for value in limits)):
        required = bounded & (np.asarray(offset) < widths).all(1)
        indices = np.flatnonzero(required & covered)
        query = (lower[indices] + offset - origin) @ strides
        found = np.searchsorted(codes, query)
        covered[indices] &= codes[np.minimum(found, len(codes) - 1)] == query
    return covered


def terrain_covers(
    triangles: NDArray[np.float64], terrain: Terrain, config: GroundConfig, tolerance: float
) -> NDArray[np.bool_]:
    """Require full XY footprint and a conservative elevation-error envelope."""
    if not len(triangles):
        return np.empty(0, dtype=bool)
    lower = np.floor(triangles[:, :, :2].min(1) / config.cell).astype(np.int64)
    # A face lying exactly on a grid edge need not use the cell on its far side.
    upper = np.floor(np.nextafter(triangles[:, :, :2].max(1), -np.inf) / config.cell).astype(
        np.int64
    )
    upper = np.maximum(lower, upper)
    covered = occupied_bounds(lower, upper, terrain.cells)
    heights = terrain_heights(triangles[:, :, :2].reshape(-1, 2), terrain, config).reshape(-1, 3)
    error = np.abs(triangles[:, :, 2] - heights).max(1)
    xy_edges = triangles[:, [1, 2, 0], :2] - triangles[:, :, :2]
    # Both measured interpolation and terrain can vary inside the triangle.
    envelope = config.max_slope * np.linalg.norm(xy_edges, axis=2).max(1)
    covered &= np.isfinite(error) & (error + envelope <= tolerance)
    return covered


def voxels_cover(
    triangles: NDArray[np.float64], keys: NDArray[np.int64], cell: float
) -> NDArray[np.bool_]:
    if not len(triangles):
        return np.empty(0, dtype=bool)
    lower = np.floor(triangles.min(1) / cell).astype(np.int64)
    upper = np.floor(np.nextafter(triangles.max(1), -np.inf) / cell).astype(np.int64)
    return occupied_bounds(lower, np.maximum(lower, upper), keys)
