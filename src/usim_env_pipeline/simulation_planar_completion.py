"""Bounded plane-local completion using observed depth to protect real openings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .plane_repair import repair
from .plane_types import Plane, PlaneConfig


@dataclass(frozen=True, slots=True)
class CompletionContext:
    """Reuse one source tree for bounded local repair instead of repeated map scans."""

    points: NDArray[np.float32]
    tree: cKDTree

    def footprint(self, plane: Plane, cell: float) -> NDArray[np.int64] | None:
        support = self.points[plane.ids]
        low, high = support.min(0).astype(np.float64) - 2, support.max(0) + 2
        center = (low + high) / 2
        ids = np.asarray(
            self.tree.query_ball_point(center, np.linalg.norm(high - low) / 2 + 1e-6),
            dtype=np.int64,
        )
        ids = np.sort(ids[((self.points[ids] >= low) & (self.points[ids] <= high)).all(1)])
        local = Plane(plane.normal, plane.offset, np.searchsorted(ids, plane.ids))
        result = repair(
            self.points[ids],
            local,
            PlaneConfig(cell=cell, bridge_radius=0.2, hole_area=0.16, min_area=0.5),
        )
        if result is None:
            return None
        return (np.argwhere(result.mask) + result.lower).astype(np.int64)
