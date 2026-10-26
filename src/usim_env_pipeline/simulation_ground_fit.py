"""Local terrain fits with footprint support and independently checked continuity."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .simulation_ground_config import GroundConfig


def fit_nodes(
    measured: NDArray[np.float64],
    ids: NDArray[np.int64],
    xy: NDArray[np.float64],
    config: GroundConfig,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Do not turn two locally planar levels into a single plausible ramp."""
    from scipy.spatial import cKDTree

    tree = cKDTree(measured[:, :2])
    heights = np.full(len(xy), np.nan)
    provenance = np.full((len(xy), config.neighbors), -1, np.int64)
    for start in range(0, len(xy), 8192):
        query = xy[start : start + 8192]
        distance, index = tree.query(
            query,
            k=config.neighbors,
            distance_upper_bound=np.nextafter(config.radius, np.inf),
            workers=1,
        )
        present = index < len(measured)
        safe_index = np.minimum(index, len(measured) - 1)
        order = np.lexsort((ids[safe_index], distance), axis=1)
        index = np.take_along_axis(safe_index, order, axis=1)
        present = np.take_along_axis(present, order, axis=1)
        samples = measured[index]

        def plane(mask: NDArray[np.bool_]):
            count = mask.sum(1)
            mean = (samples * mask[..., None]).sum(1) / np.maximum(count, 1)[:, None]
            delta = (samples - mean[:, None]) * mask[..., None]
            covariance = np.einsum('nki,nkj->nij', delta[:, :, :2], delta[:, :, :2])
            rhs = np.einsum('nki,nk->ni', delta[:, :, :2], delta[:, :, 2])
            eigen = np.linalg.eigvalsh(covariance)
            rank = (count >= 3) & (eigen[:, 0] > np.maximum(eigen[:, 1] * 1e-10, 1e-12))
            covariance[~rank] = np.eye(2)
            gradient = np.linalg.solve(covariance, rhs[..., None])[..., 0]
            height = mean[:, 2] + np.einsum('ni,ni->n', query - mean[:, :2], gradient)
            residual = delta[:, :, 2] - np.einsum('nki,ni->nk', delta[:, :, :2], gradient)
            supported = rank & (np.max(np.abs(residual), axis=1) <= config.max_error)
            supported &= np.linalg.norm(gradient, axis=1) <= config.max_slope
            return height, supported

        height, valid = plane(present)
        valid &= present.sum(1) >= config.min_neighbors
        valid &= distance[:, 0] <= config.nearest_support
        offset = samples[:, :, :2] - query[:, None]
        # The query must lie within the convex support hull, not merely near it.
        angles = np.sort(
            np.where(present, np.arctan2(offset[:, :, 1], offset[:, :, 0]), np.inf), axis=1
        )
        count = present.sum(1)
        with np.errstate(invalid='ignore'):
            gaps = np.diff(angles, axis=1)
        gaps = np.where(np.isfinite(gaps), gaps, 0)
        last = np.where(count > 0, angles[np.arange(len(query)), np.maximum(count - 1, 0)], 0)
        first = np.where(count > 0, angles[:, 0], 0)
        wrap = 2 * np.pi + first - last
        hull = np.maximum(gaps.max(1), wrap) <= np.pi + 1e-8
        hull |= distance[:, 0] <= 1e-8
        valid &= hull
        # Independent one-sided sheets expose a small step that a global fit
        # otherwise disguises as a low-gradient, low-residual ramp.
        for axis in range(2):
            left = present & (offset[:, :, axis] < 0)
            right = present & ~left
            low, low_valid = plane(left)
            high, high_valid = plane(right)
            incompatible = (
                low_valid & high_valid & (np.abs(low - high) > config.continuity_tolerance)
            )
            valid &= ~incompatible
        coincident = np.linalg.norm(offset[:, :, None] - offset[:, None], axis=3) < 0.02
        separation = np.abs(samples[:, :, None, 2] - samples[:, None, :, 2])
        ambiguous = (
            coincident
            & (separation > 2 * config.continuity_tolerance)
            & present[:, :, None]
            & present[:, None]
        ).any((1, 2))
        valid &= ~ambiguous
        stop = start + len(query)
        heights[start:stop] = np.where(valid, height, np.nan)
        provenance[start:stop] = np.where(present, ids[index], -1)
    return heights, provenance
