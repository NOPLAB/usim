"""Local occupied-cell density cleanup before obstacle or planar model ownership."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .simulation_obstacles import ObstacleConfig, Occupancy


@dataclass(frozen=True, slots=True)
class DensityCleanup:
    support_counts: NDArray[np.int64]
    keep: NDArray[np.bool_]


def density_cleanup(occupied: Occupancy, config: ObstacleConfig) -> DensityCleanup:
    """Count distinct occupied neighbors, including self, in a fixed physical radius.

    Connectivity and repeated returns do not determine the decision. Disconnected
    leaves can support one another; scanning an isolated cell repeatedly cannot.
    Integer-grid queries avoid cancellation at large world coordinates. The
    minute radius allowance includes cells exactly on the configured boundary.
    """
    from scipy.spatial import cKDTree

    counts = np.zeros(len(occupied.keys), dtype=np.int64)
    if len(counts):
        tree = cKDTree(occupied.keys)
        radius = config.cleanup_radius / config.cell + 1e-10
        for start in range(0, len(counts), 32768):
            stop = min(start + 32768, len(counts))
            counts[start:stop] = tree.query_ball_point(
                occupied.keys[start:stop], radius, return_length=True, workers=4
            )
    return DensityCleanup(counts, counts >= config.cleanup_min_cells)
