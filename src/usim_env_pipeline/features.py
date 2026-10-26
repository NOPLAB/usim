"""Conservative crown and narrow-column inference from meter/Z-up point clouds."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Tree:
    x: float
    y: float
    height: float
    radius: float
    supported_points: int


CELL_METERS: Final = 1.0


def relative_heights(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Estimate local measured-ground offsets without projecting onto an external terrain model."""
    from scipy.ndimage import distance_transform_edt, median_filter, minimum_filter
    from scipy.stats import binned_statistic_2d

    origin = np.floor(points[:, :2].min(0)) - 1
    cells = np.floor((points[:, :2] - origin) / CELL_METERS).astype(np.int64)
    shape = tuple((cells.max(0) + 2).tolist())
    # Measured local PCD ground defines feature height; no external terrain is assumed.
    ground, _, _, _ = binned_statistic_2d(
        cells[:, 0],
        cells[:, 1],
        points[:, 2],
        statistic=lambda values: float(np.quantile(values, 0.1)),
        bins=[np.arange(shape[0] + 1), np.arange(shape[1] + 1)],
    )
    missing = ~np.isfinite(ground)
    nearest = distance_transform_edt(missing, return_distances=False, return_indices=True)
    ground[missing] = ground[tuple(nearest[:, missing])]
    ground = median_filter(minimum_filter(ground, size=5), size=3)
    return points[:, 2] - ground[cells[:, 0], cells[:, 1]]


def infer_trees(points: NDArray[np.float64]) -> list[Tree]:
    """Find volumetric crowns, rejecting flat roofs and narrow planar facades."""
    from scipy.ndimage import gaussian_filter, label, maximum_filter
    from scipy.spatial import cKDTree

    origin = np.floor(points[:, :2].min(0)) - 1
    cells = np.floor((points[:, :2] - origin) / CELL_METERS).astype(np.int64)
    shape = tuple((cells.max(0) + 2).tolist())
    relative = relative_heights(points)
    keep = (relative > 3) & (relative < 25)
    elevated, bins, heights = points[keep], cells[keep], relative[keep]
    upper = np.zeros(shape)
    lower = np.full(shape, np.inf)
    np.maximum.at(upper, (bins[:, 0], bins[:, 1]), heights)
    np.minimum.at(lower, (bins[:, 0], bins[:, 1]), heights)
    # A roof's constant height cannot supply a three-dimensional crown.
    crown = (upper - lower) > 1.2
    smooth = gaussian_filter(np.where(crown, upper, 0), sigma=1)
    maxima = (smooth == maximum_filter(smooth, size=7)) & (smooth > 4)
    groups, count = label(maxima)
    seeds = []
    for group in range(1, count + 1):
        positions = np.argwhere(groups == group)
        chosen = positions[np.argmax(smooth[tuple(positions.T)])]
        seeds.append(origin + chosen + 0.5)
    if not seeds:
        return []
    distance, ownership = cKDTree(np.asarray(seeds)).query(elevated[:, :2], workers=4)
    trees = []
    for index, center in enumerate(seeds):
        members = (ownership == index) & (distance < 7)
        cloud, h = elevated[members], heights[members]
        if len(cloud) < 100:
            continue
        covariance = np.cov(cloud[:: max(1, len(cloud) // 2000)].T)
        eigen = np.linalg.eigvalsh(covariance)
        radius = float(np.quantile(np.linalg.norm(cloud[:, :2] - center, axis=1), 0.9))
        if eigen[0] / eigen.sum() < 0.025 or radius < 1.3:
            continue
        # Recessed windows and adjoining walls are globally volumetric too.
        # Test local patches at a fixed spatial density, not scan-return density.
        _, representatives = np.unique(np.floor(cloud / 0.3), axis=0, return_index=True)
        sample = cloud[representatives]
        distances, neighbors = cKDTree(sample).query(
            sample, k=32, distance_upper_bound=1.5, workers=1
        )
        complete = np.isfinite(distances).all(axis=1)
        if complete.any():
            patches = sample[neighbors[complete]]
            offsets = patches - patches.mean(axis=1, keepdims=True)
            local = np.linalg.eigvalsh(np.einsum('nki,nkj->nij', offsets, offsets))
            planar = local[:, 0] < 0.005 * local[:, 1]
            if planar.mean() > 0.65:
                continue
        trees.append(
            Tree(
                float(center[0]),
                float(center[1]),
                float(np.quantile(h, 0.98)),
                radius,
                len(cloud),
            )
        )
    return trees


@dataclass(frozen=True, slots=True)
class Post:
    x: float
    y: float
    height: float
    radius: float
    supported_points: int


class Canopy(Protocol):
    @property
    def x(self) -> float: ...

    @property
    def y(self) -> float: ...

    @property
    def radius(self) -> float: ...


def infer_posts(
    points: NDArray[np.float64],
    trees: Sequence[Canopy],
    height: NDArray[np.float64],
) -> list[Post]:
    """Reject canopy support and broad planes; retain narrow multilevel columns."""
    from scipy.ndimage import label
    from scipy.spatial import cKDTree

    keep = (height > 1) & (height < 12)
    if trees:
        centers = np.asarray([(tree.x, tree.y) for tree in trees])
        distance, index = cKDTree(centers).query(points[:, :2], workers=4)
        radii = np.asarray([tree.radius for tree in trees])
        keep &= distance > radii[index] + 1
    cloud, h = points[keep], height[keep]
    if not len(cloud):
        return []
    origin = np.floor(cloud[:, :2].min(0)) - 1
    cells = np.floor((cloud[:, :2] - origin) / 0.3).astype(np.int64)
    shape = tuple((cells.max(0) + 2).tolist())
    occupied = np.zeros(shape, dtype=bool)
    occupied[tuple(cells.T)] = True
    groups, _ = label(occupied)
    owners = groups[tuple(cells.T)]
    order = np.argsort(owners)
    _, starts = np.unique(owners[order], return_index=True)
    ends = np.append(starts[1:], len(order))
    posts = []
    for start, end in zip(starts, ends):
        selected = order[start:end]
        part, levels = cloud[selected], h[selected]
        if len(part) < 60:
            continue
        width = np.quantile(part[:, :2], 0.95, axis=0) - np.quantile(part[:, :2], 0.05, axis=0)
        span = float(np.quantile(levels, 0.98) - np.quantile(levels, 0.02))
        if width.max() < 0.8 and span > 1.5 and len(np.unique((levels * 2).astype(int))) >= 5:
            center = np.median(part[:, :2], axis=0)
            posts.append(
                Post(
                    float(center[0]),
                    float(center[1]),
                    float(np.quantile(levels, 0.98)),
                    max(0.05, float(width.max()) / 2),
                    len(part),
                )
            )
    return posts
