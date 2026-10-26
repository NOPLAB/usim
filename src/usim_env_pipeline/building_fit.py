"""Fit closed cuboids to connected measured facades, never to horizontal ground."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .building_types import BuildingConfig, Cuboid, Facade

Points = NDArray[np.float32] | NDArray[np.float64]


def facade_components(
    source_plane: int,
    equation: NDArray[np.float64],
    ids: NDArray[np.int64],
    points: Points,
    config: BuildingConfig,
) -> list[Facade]:
    """Keep dense, tall wall components; split unsupported gaps before filling holes."""
    normal = equation[:3].copy()
    if abs(normal[2]) > np.sin(np.deg2rad(config.vertical_angle)):
        return []
    normal[2] = 0
    normal /= np.linalg.norm(normal)
    if normal[np.argmax(np.abs(normal))] < 0:
        normal = -normal
    support = points[ids].astype(np.float64)
    offset = -float(np.median(support @ normal))
    near = np.abs(support @ normal + offset) <= config.support_distance
    ids, support = ids[near], support[near]
    if len(ids) < config.minimum_points:
        return []
    tangent = np.array([normal[1], -normal[0], 0.0])
    coordinates = np.column_stack((support @ tangent, support[:, 2]))
    cells, first, inverse = np.unique(
        np.floor(coordinates / config.cell).astype(np.int64),
        axis=0,
        return_index=True,
        return_inverse=True,
    )
    radius = max(config.gap, np.sqrt(2) * config.cell)
    pairs = cKDTree(coordinates[first]).query_pairs(radius, output_type='ndarray')
    graph = coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(cells), len(cells))
    )
    count, labels = connected_components(graph, directed=False)
    result = []
    for label in range(count):
        selected = labels[inverse] == label
        if selected.sum() < config.minimum_points:
            continue
        local = coordinates[selected]
        lower, upper = local.min(0), local.max(0)
        width, height = upper - lower
        area = width * height
        coverage = min(1.0, (labels == label).sum() * config.cell**2 / max(area, 1e-12))
        if (
            width < config.minimum_width
            or height < config.minimum_height
            or area < config.minimum_area
            or coverage < config.minimum_coverage
        ):
            continue
        result.append(
            Facade(
                (source_plane,), normal, offset, lower, upper, np.unique(ids[selected]), coverage
            )
        )
    return result


def merge_facades(facades: list[Facade], points: Points, config: BuildingConfig) -> list[Facade]:
    """Join only nearby coplanar support, keeping separated buildings as separate components."""
    kept: list[Facade] = []
    cosine = np.cos(np.deg2rad(config.corner_angle))
    for small in sorted(facades, key=lambda wall: -len(wall.support_ids)):
        for index, big in enumerate(kept):
            if big.normal @ small.normal < cosine:
                continue
            small_points = points[small.support_ids].astype(np.float64)
            if np.abs(small_points @ big.normal + big.offset).max() > config.support_distance:
                continue
            small_local = np.column_stack((small_points @ big.tangent, small_points[:, 2]))
            separation = np.maximum(
                np.maximum(big.lower - small_local.max(0), small_local.min(0) - big.upper), 0
            )
            if np.linalg.norm(separation) > config.gap:
                continue
            big_points = points[big.support_ids].astype(np.float64)
            big_local = np.column_stack((big_points @ big.tangent, big_points[:, 2]))
            if cKDTree(big_local).query(small_local, workers=1)[0].min() > config.gap:
                continue
            ids = np.union1d(big.support_ids, small.support_ids)
            support = points[ids].astype(np.float64)
            local = np.column_stack((support @ big.tangent, support[:, 2]))
            lower, upper = local.min(0), local.max(0)
            occupied = len(np.unique(np.floor(local / config.cell), axis=0))
            coverage = min(1.0, occupied * config.cell**2 / np.prod(upper - lower))
            if coverage < config.minimum_coverage:
                continue
            kept[index] = Facade(
                tuple(sorted(set((*big.source_planes, *small.source_planes)))),
                big.normal,
                big.offset,
                lower,
                upper,
                ids,
                float(coverage),
            )
            break
        else:
            kept.append(small)
    return kept


def _corner(
    first: Facade,
    second: Facade,
    indices: tuple[int, int],
    points: Points,
    config: BuildingConfig,
) -> Cuboid | None:
    """Infer a bounded envelope only when two perpendicular facades meet at their ends."""
    if abs(first.normal @ second.normal) > np.sin(np.deg2rad(config.corner_angle)):
        return None
    heights = (first.upper[1] - first.lower[1], second.upper[1] - second.lower[1])
    overlap = min(first.upper[1], second.upper[1]) - max(first.lower[1], second.lower[1])
    if overlap / max(heights) < config.height_overlap:
        return None
    corner = np.r_[
        np.linalg.solve(
            np.stack((first.normal[:2], second.normal[:2])),
            -np.array([first.offset, second.offset]),
        ),
        0.0,
    ]
    directions, lengths = [], []
    for wall in (first, second):
        position = float(corner @ wall.tangent)
        ends = np.array([wall.lower[0], wall.upper[0]])
        near = int(np.argmin(np.abs(ends - position)))
        if abs(ends[near] - position) > config.corner_gap:
            return None
        directions.append(1 if near == 0 else -1)
        lengths.append(abs(ends[1 - near] - position))
    axes = np.column_stack((first.tangent, first.normal, [0.0, 0.0, 1.0]))
    across = np.sign(directions[1] * second.tangent @ first.normal)
    lower_z = min(first.lower[1], second.lower[1])
    upper_z = max(first.upper[1], second.upper[1])
    center = corner + (
        first.tangent * directions[0] * lengths[0] / 2 + first.normal * across * lengths[1] / 2
    )
    center[2] = (lower_z + upper_z) / 2
    size = np.array([lengths[0], lengths[1], upper_z - lower_z])
    for wall, axis in ((first, 1), (second, 0)):
        local = (points[wall.support_ids] - center) @ axes
        residual = np.abs(np.abs(local[:, axis]) - size[axis] / 2)
        if residual.max() > config.support_distance:
            return None
        if (np.abs(local) > size / 2 + config.corner_gap).any():
            return None
    return Cuboid(center, size, axes, indices, 'building')


def fit_cuboids(facades: list[Facade], points: Points, config: BuildingConfig) -> list[Cuboid]:
    """Use corner-supported envelopes, otherwise bounded thin solid facade cuboids."""
    pairs = []
    for first in range(len(facades)):
        for second in range(first + 1, len(facades)):
            model = _corner(facades[first], facades[second], (first, second), points, config)
            if model is not None:
                pairs.append(model)
    used: set[int] = set()
    models = []
    for model in sorted(pairs, key=lambda value: -np.prod(value.size)):
        if any(index in used for index in model.facade_indices):
            continue
        members = list(model.facade_indices)
        for index, wall in enumerate(facades):
            if index in used or index in members:
                continue
            local = (points[wall.support_ids] - model.center) @ model.axes
            distances = np.abs(np.abs(local[:, :2]) - model.size[:2] / 2)
            if (np.abs(local) <= model.size / 2 + config.support_distance).all() and (
                distances.max(0) <= config.support_distance
            ).any():
                members.append(index)
        used.update(members)
        models.append(
            Cuboid(model.center, model.size, model.axes, tuple(sorted(members)), 'building')
        )
    for index, wall in enumerate(facades):
        if index in used:
            continue
        center = wall.tangent * ((wall.lower[0] + wall.upper[0]) / 2) - wall.normal * wall.offset
        center[2] = (wall.lower[1] + wall.upper[1]) / 2
        models.append(
            Cuboid(
                center,
                np.array(
                    [
                        wall.upper[0] - wall.lower[0],
                        config.wall_thickness,
                        wall.upper[1] - wall.lower[1],
                    ]
                ),
                np.column_stack((wall.tangent, wall.normal, [0.0, 0.0, 1.0])),
                (index,),
                'facade',
            )
        )
    return models
