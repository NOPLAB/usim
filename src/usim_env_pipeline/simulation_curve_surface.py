"""Continuous implicit envelopes of measured residual geometry, not smoothed cube faces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes


@dataclass(frozen=True, slots=True)
class CurveSurface:
    positions: NDArray[np.float32]
    faces: NDArray[np.int32]


def quantized_surface(
    positions: NDArray[np.float64] | NDArray[np.float32], faces: NDArray[np.int32]
) -> CurveSurface:
    """Weld coordinates that become identical in USD and remove collapsed faces."""
    points, inverse = np.unique(positions.astype(np.float32), axis=0, return_inverse=True)
    indices = inverse[faces]
    triangles = points[indices].astype(np.float64)
    area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    indices = indices[area > 0]
    used, local = np.unique(indices, return_inverse=True)
    return CurveSurface(points[used], local.reshape(-1, 3).astype(np.int32))


def supported_surface(
    positions: NDArray[np.float32], faces: NDArray[np.int32], support: cKDTree
) -> CurveSurface:
    """Bound smoothing at concave gaps, including float32 publication error."""
    distance, nearest = support.query(positions, workers=4)
    outside = distance > 0.299
    exact = positions.astype(np.float64)
    centers = support.data[nearest[outside]].copy()
    neighbors = support.query_ball_point(exact[outside], distance[outside] + 1e-10, workers=4)
    for index, candidates in enumerate(neighbors):
        if len(candidates) > 1:
            tied = support.data[candidates]
            centers[index] = tied[np.lexsort((tied[:, 2], tied[:, 1], tied[:, 0]))[0]]
    exact[outside] = centers + (exact[outside] - centers) * (0.299 / distance[outside, None])
    return quantized_surface(exact, faces)


def curve_surface(keys: NDArray[np.int64], tile: tuple[int, int], cell: float) -> CurveSurface:
    """Fit a 0.1 m-scale smooth distance field with a 0.2 m supported envelope.

    Fifteen-metre core tiles include a halo wider than both distance truncation
    and Gaussian support. Only core-owned faces are returned. Edge interpolation
    is recalculated in float64 global coordinates so neighboring tiles agree.
    Large unsupported gaps are not closed by a global hull or Poisson extrapolation.
    """
    width = int(round(15 / cell))
    halo = int(np.ceil(0.4 / cell)) + int(np.ceil(0.3 / cell)) + 2
    core = np.asarray(tile, np.int64) * width
    nearby = keys[((keys[:, :2] >= core - halo) & (keys[:, :2] <= core + width + halo)).all(1)]
    if not len(nearby):
        return CurveSurface(np.empty((0, 3), np.float32), np.empty((0, 3), np.int32))
    lower = np.r_[core - halo, nearby[:, 2].min() - halo]
    upper = np.r_[core + width + halo, nearby[:, 2].max() + halo]
    shape = tuple((upper - lower + 1).tolist())
    occupied = np.zeros(shape, bool)
    occupied[tuple((nearby - lower).T)] = True
    distance = distance_transform_edt(~occupied).astype(np.float32)
    distance *= cell
    np.minimum(distance, 0.4, out=distance)
    field = gaussian_filter(distance, sigma=0.1 / cell, truncate=3.0) - 0.2
    vertices, faces, _, _ = marching_cubes(
        field, 0, gradient_direction='descent', allow_degenerate=False
    )
    lo, hi = np.floor(vertices).astype(np.int64), np.ceil(vertices).astype(np.int64)
    along = hi - lo
    edge = along.sum(1) == 1
    axes = along[edge].argmax(1)
    first, second = field[tuple(lo[edge].T)], field[tuple(hi[edge].T)]
    exact = vertices.astype(np.float64)
    exact[edge] = lo[edge]
    exact[np.flatnonzero(edge), axes] += first.astype(np.float64) / (first - second)
    global_grid = exact + lower
    owners = np.floor(global_grid[faces].min(1)[:, :2]).astype(np.int64) // width
    faces = faces[(owners == tile).all(1)]
    used, inverse = np.unique(faces, return_inverse=True)
    positions = ((global_grid[used] + 0.5) * cell).astype(np.float32)
    return supported_surface(
        positions,
        inverse.reshape(-1, 3).astype(np.int32),
        cKDTree((nearby + 0.5) * cell),
    )
