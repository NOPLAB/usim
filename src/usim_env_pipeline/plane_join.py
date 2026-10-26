"""Parallel per-board Gaussian distance costs and source-bounded planar joins."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .plane_types import Plane, PlaneConfig
from .planes import refit


@dataclass(frozen=True, slots=True)
class JoinResult:
    planes: list[Plane]
    equations: NDArray[np.float64]
    edges: NDArray[np.float64]
    accepted: NDArray[np.bool_]


def gaussian_cost(
    distances: NDArray[np.float64], angles: NDArray[np.float64], config: PlaneConfig
) -> NDArray[np.float64]:
    """Negative log likelihood relative to the peak of independent zero-mean Gaussians."""
    return 0.5 * (
        (distances[:, 0] / config.distance_sigma) ** 2
        + (distances[:, 1] / config.gap_sigma) ** 2
        + (angles / config.angle_sigma) ** 2
    )


def link_costs(
    points: NDArray[np.float32],
    links: NDArray[np.int64],
    models: NDArray[np.float64],
    config: PlaneConfig,
) -> NDArray[np.float64]:
    """Evaluate each actual boundary correspondence, not only its parent board pair."""
    vertices = points[links].astype(np.float64)
    distance = (
        np.abs(np.einsum('ij,ij->i', vertices[:, 0], models[:, 1, :3]) + models[:, 1, 3])
        + np.abs(np.einsum('ij,ij->i', vertices[:, 1], models[:, 0, :3]) + models[:, 0, 3])
    ) / 2
    gap = np.linalg.norm(vertices[:, 0] - vertices[:, 1], axis=1)
    angle = np.rad2deg(
        np.arccos(np.clip(np.abs(np.einsum('ij,ij->i', models[:, 0, :3], models[:, 1, :3])), 0, 1))
    )
    return gaussian_cost(np.column_stack((distance, gap)), angle, config)


def join_planes(
    points: NDArray[np.float32], planes: list[Plane], config: PlaneConfig
) -> JoinResult:
    """Compute board costs concurrently, join compatible groups and retain source-backed links."""
    from scipy.spatial import cKDTree

    equations = np.asarray([np.append(p.normal, p.offset) for p in planes]).reshape(-1, 4)
    if not planes:
        return JoinResult([], equations, np.empty((0, 6)), np.empty(0, dtype=bool))
    samples = [p.ids[:: max(1, int(np.ceil(len(p.ids) / 2048)))] for p in planes]
    trees = [cKDTree(points[ids]) for ids in samples]
    low = np.array([points[p.ids].min(0) for p in planes])
    high = np.array([points[p.ids].max(0) for p in planes])
    reach = config.gap_sigma * np.sqrt(2 * config.join_cost)

    def evaluate(first: int) -> list[tuple[float, float, float, float, float, float]]:
        bounds = np.maximum(np.maximum(low - high[first], low[first] - high), 0)
        candidates = np.flatnonzero(np.linalg.norm(bounds, axis=1) <= reach)
        rows = []
        for second in candidates[candidates > first]:
            angle = float(
                np.rad2deg(
                    np.arccos(np.clip(abs(planes[first].normal @ planes[second].normal), 0, 1))
                )
            )
            if angle > config.angle_sigma * np.sqrt(2 * config.join_cost):
                continue
            a, b = points[samples[first]], points[samples[second]]
            distance = float(
                (
                    np.median(np.abs(a @ planes[second].normal + planes[second].offset))
                    + np.median(np.abs(b @ planes[first].normal + planes[first].offset))
                )
                / 2
            )
            gap = float(trees[second].query(a, workers=1)[0].min())
            cost = float(gaussian_cost(np.array([[distance, gap]]), np.array([angle]), config)[0])
            rows.append((float(first), float(second), distance, gap, angle, cost))
        return rows

    # Each worker owns one board's cost calculations; native NumPy/SciPy release the GIL.
    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        rows = [row for board in pool.map(evaluate, range(len(planes))) for row in board]
    edges = np.asarray(rows, dtype=np.float64).reshape(-1, 6)
    accepted = np.zeros(len(edges), dtype=bool)
    parent = list(range(len(planes)))
    groups = list(planes)

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for number in np.argsort(edges[:, 5], kind='stable'):
        first, second = edges[number, :2].astype(int)
        if edges[number, 5] > config.join_cost:
            continue
        a, b = root(first), root(second)
        if a == b:
            accepted[number] = True
            continue
        ids = np.unique(np.concatenate((groups[a].ids, groups[b].ids)))
        common = refit(points, ids)
        residual = np.abs(points[ids] @ common.normal + common.offset)
        # A low pair cost alone cannot flatten a curved chain or erase a separate level.
        if np.quantile(residual, 0.95) > config.tolerance:
            continue
        ids = ids[residual <= config.tolerance]
        if len(ids) < config.min_support:
            continue
        parent[b] = a
        groups[a] = Plane(common.normal, common.offset, ids)
        accepted[number] = True
    results = []
    for number in range(len(planes)):
        if root(number) != number:
            continue
        plane = groups[number]
        source_links, source_models = [], []
        for edge in edges[accepted]:
            first, second = edge[:2].astype(int)
            if root(first) != number:
                continue
            distance, nearest = trees[second].query(points[samples[first]], workers=1)
            order = np.flatnonzero(distance <= reach)
            links = np.column_stack((samples[first][order], samples[second][nearest[order]]))
            models = np.tile(np.array([first, second], dtype=np.int32), (len(links), 1))
            cost = link_costs(points, links, equations[models], config)
            keep = (cost <= config.join_cost) & np.isin(links, plane.ids).all(1)
            links, models = links[keep], models[keep]
            if len(links) > 256:
                select = np.linspace(0, len(links) - 1, 256).astype(int)
                links, models = links[select], models[select]
            source_links.extend(links)
            source_models.extend(models)
        results.append(
            Plane(
                plane.normal,
                plane.offset,
                plane.ids,
                np.asarray(source_links, dtype=np.int64).reshape(-1, 2),
                np.asarray(source_models, dtype=np.int32).reshape(-1, 2),
            )
        )
    return JoinResult(results, equations, edges, accepted)
