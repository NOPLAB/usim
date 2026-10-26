"""Bounded planar completion with depth-supported openings and Gaussian bridge seeds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .plane_types import Plane, PlaneConfig


@dataclass(frozen=True, slots=True)
class Footprint:
    lower: NDArray[np.int64]
    mask: NDArray[np.bool_]
    observed: NDArray[np.bool_]
    protected: NDArray[np.bool_]
    enclosed: NDArray[np.bool_]
    bridged: NDArray[np.bool_]


def link_seed(
    coordinates: NDArray[np.float64], lower: NDArray[np.int64], mask: NDArray[np.bool_]
) -> NDArray[np.bool_]:
    """Rasterize only segments whose endpoints came from compatible measured plates."""
    result = np.zeros_like(mask)
    for start, end in np.floor(coordinates).astype(np.int64) - lower:
        steps = int(np.abs(end - start).max()) + 1
        cells = np.rint(np.linspace(start, end, steps)).astype(np.int64)
        result[tuple(cells.T)] = True
    return result


def repair(points: NDArray[np.float32], plane: Plane, config: PlaneConfig) -> Footprint | None:
    from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes, label
    from scipy.spatial import cKDTree

    from .plane_types import plane_basis

    basis = np.column_stack(plane_basis(plane.normal))
    cells = np.floor(points[plane.ids] @ basis / config.cell).astype(np.int64)
    radius = int(np.ceil(config.bridge_radius / config.cell))
    lower = cells.min(0) - radius - 2
    shape = tuple((cells.max(0) - lower + radius + 3).tolist())
    observed = np.zeros(shape, dtype=bool)
    observed[tuple((cells - lower).T)] = True
    if observed.sum() * config.cell**2 < config.min_area:
        return None
    low, high = points[plane.ids].min(0) - 2, points[plane.ids].max(0) + 2
    local_ids = np.flatnonzero(((points >= low) & (points <= high)).all(axis=1))
    signed = points[local_ids] @ plane.normal + plane.offset
    # Above-floor objects do not establish a floor opening. Walls lack viewpoint orientation.
    depth = -signed if plane.normal[2] > 0.9 else np.abs(signed)
    recess = points[local_ids[(depth >= 0.25) & (depth <= 2)]]
    recess_cells = np.floor(recess @ basis / config.cell).astype(np.int64) - lower
    inside = ((recess_cells >= 0) & (recess_cells < np.array(shape))).all(axis=1)
    protected = np.zeros(shape, dtype=bool)
    keys, counts = np.unique(recess_cells[inside], axis=0, return_counts=True)
    protected[tuple(keys[counts >= 3].T)] = True
    protected = binary_dilation(protected, iterations=1)
    bridges = np.zeros(shape, dtype=bool)
    if plane.links is not None and len(plane.links):
        bridges = link_seed(points[plane.links] @ basis / config.cell, lower, observed)
    x, y = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    disk = x * x + y * y <= radius * radius
    seed = observed | bridges
    closed = binary_closing(seed, structure=disk) | seed
    local = binary_closing(observed, structure=disk) | observed
    bridged = closed & ~local
    holes, _ = label(binary_fill_holes(closed) & ~closed)
    sizes = np.bincount(holes.ravel())
    eligible = sizes * config.cell**2 <= config.hole_area
    eligible[0] = False
    enclosed = eligible[holes]
    repaired = closed | enclosed
    repaired[protected & ~observed] = False
    locations = np.argwhere(repaired)
    near = cKDTree(np.argwhere(seed) * config.cell).query(locations * config.cell, workers=1)[0]
    far = (near > config.bridge_radius + config.cell) & ~enclosed[tuple(locations.T)]
    repaired[tuple(locations[far].T)] = False
    return Footprint(
        lower,
        repaired,
        observed,
        protected,
        enclosed & repaired,
        bridged & repaired & ~observed,
    )
