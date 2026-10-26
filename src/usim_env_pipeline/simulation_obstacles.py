"""Bounded occupied-volume proxies for measured, nonbuilding obstacle returns."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from .models import PipelineInputError


@dataclass(frozen=True, slots=True)
class ObstacleConfig:
    cell: float = 0.1
    confidence_records: int = 3
    building_margin: float = 0.2
    ground_tolerance: float = 0.08
    cleanup_radius: float = 0.4
    cleanup_min_cells: int = 5

    def __post_init__(self) -> None:
        values = (self.cell, self.building_margin, self.ground_tolerance, self.cleanup_radius)
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise PipelineInputError('Obstacle distances must be finite and positive')
        if self.confidence_records < 1 or int(self.confidence_records) != self.confidence_records:
            raise PipelineInputError('Obstacle support counts must be positive integers')
        if (
            isinstance(self.cleanup_min_cells, bool)
            or not isinstance(self.cleanup_min_cells, int)
            or self.cleanup_min_cells < 1
        ):
            raise PipelineInputError('Cleanup requires a positive integer occupied-cell count')


@dataclass(frozen=True, slots=True)
class Occupancy:
    keys: NDArray[np.int64]
    counts: NDArray[np.int64]
    ids: NDArray[np.int64]
    groups: NDArray[np.int64]


def occupied_obstacles(
    points: NDArray[np.float32], ids: NDArray[np.int64], config: ObstacleConfig
) -> Occupancy:
    """Retain every occupied volume; density reports confidence, never object existence."""
    empty = Occupancy(
        np.empty((0, 3), dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
    )
    if not len(ids):
        return empty
    keys, groups, counts = np.unique(
        np.floor(points[ids].astype(np.float64) / config.cell).astype(np.int64),
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    return Occupancy(keys, counts, ids.copy(), groups)


def voxel_surface(
    keys: NDArray[np.int64], cell: float
) -> tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.int64]]:
    """Expose only exterior voxel faces; return owning voxel for every triangle.

    Integer corners are deduplicated before meter conversion. Neighboring volumes
    have no internal faces. Edge/corner contacts are allowed, but no open boundary
    is introduced. Empty cells never gain collision through a global convex hull.
    """
    if not len(keys):
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.int32),
            np.empty(0, dtype=np.int64),
        )
    lower = keys.min(0) - 1
    shape = keys.max(0) - lower + 2
    strides = np.array([shape[1] * shape[2], shape[2], 1], dtype=np.int64)
    codes = (keys - lower) @ strides
    order = np.argsort(codes)
    sorted_codes = codes[order]
    corners = np.array(
        [
            [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
            [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]],
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
            [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
            [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        ],
        dtype=np.int64,
    )
    quads, owners = [], []
    for direction, delta in enumerate(
        ([-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1])
    ):
        neighbor_codes = codes + np.asarray(delta) @ strides
        indices = np.searchsorted(sorted_codes, neighbor_codes)
        present = sorted_codes[np.minimum(indices, len(codes) - 1)] == neighbor_codes
        exposed = np.flatnonzero(~present)
        quads.append(keys[exposed, None] + corners[direction])
        owners.append(exposed)
    all_corners = np.concatenate(quads).reshape(-1, 3)
    del quads
    vertices, inverse = np.unique(all_corners, axis=0, return_inverse=True)
    del all_corners
    indices = inverse.reshape(-1, 4).astype(np.int32)
    del inverse
    triangles = np.stack((indices[:, [0, 1, 2]], indices[:, [0, 2, 3]]), axis=1)
    return (
        (vertices * cell).astype(np.float32),
        triangles.reshape(-1, 3),
        np.repeat(np.concatenate(owners), 2),
    )
