"""Compact, conforming closed slabs from supported planar cell unions."""

from __future__ import annotations

from bisect import bisect_left, bisect_right

import numpy as np
from numpy.typing import NDArray


def slab_surface(
    keys: NDArray[np.int64], cell: float, thickness: float
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Merge coplanar runs, splitting shared edges to avoid rectangle T-junctions.

    The footprint remains exactly the declared cell union, including holes and
    separate components. Front/back polygon fans share all neighboring boundary
    vertices; only exposed edges are extruded. No CAD or solver dependency is used.
    """
    if not len(keys):
        return np.empty((0, 3), np.float64), np.empty((0, 3), np.int32)
    lower = keys.min(0)
    shape = tuple(keys.max(0) - lower + 1)
    mask = np.zeros(shape, bool)
    mask[tuple((keys - lower).T)] = True
    active: dict[tuple[int, int], int] = {}
    rectangles = []
    for x in range(shape[0] + 1):
        delta = np.diff(np.pad(mask[x].astype(np.int8), 1)) if x < shape[0] else np.array([])
        runs = set(zip(np.flatnonzero(delta == 1), np.flatnonzero(delta == -1)))
        for run in sorted(active.keys() - runs):
            rectangles.append(
                (
                    int(active[run] + lower[0]),
                    int(run[0] + lower[1]),
                    int(x + lower[0]),
                    int(run[1] + lower[1]),
                )
            )
        active = {run: active.get(run, x) for run in sorted(runs)}
    vertical: dict[int, set[int]] = {}
    horizontal: dict[int, set[int]] = {}
    for x0, y0, x1, y1 in rectangles:
        for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
            vertical.setdefault(x, set()).add(y)
            horizontal.setdefault(y, set()).add(x)
    vertical_nodes = {x: sorted(ys) for x, ys in vertical.items()}
    horizontal_nodes = {y: sorted(xs) for y, xs in horizontal.items()}
    lookup: dict[tuple[int, int], int] = {}
    coordinates: list[tuple[int, int]] = []
    front: list[tuple[int, int, int]] = []
    edge_counts: dict[tuple[int, int], int] = {}
    oriented: dict[tuple[int, int], tuple[int, int]] = {}

    def vertex(key: tuple[int, int]) -> int:
        if key not in lookup:
            lookup[key] = len(coordinates)
            coordinates.append(key)
        return lookup[key]

    for x0, y0, x1, y1 in sorted(rectangles):
        bottom = horizontal_nodes[y0]
        right = vertical_nodes[x1]
        top = horizontal_nodes[y1]
        left = vertical_nodes[x0]
        ring = (
            [(x, y0) for x in bottom[bisect_left(bottom, x0) : bisect_right(bottom, x1)]]
            + [(x1, y) for y in right[bisect_right(right, y0) : bisect_right(right, y1)]]
            + [(x, y1) for x in reversed(top[bisect_left(top, x0) : bisect_left(top, x1)])]
            + [(x0, y) for y in reversed(left[bisect_right(left, y0) : bisect_left(left, y1)])]
        )
        boundary = [vertex((2 * x, 2 * y)) for x, y in ring]
        center = vertex((x0 + x1, y0 + y1))
        for index, first in enumerate(boundary):
            second = boundary[(index + 1) % len(boundary)]
            front.append((center, first, second))
            edge = (min(first, second), max(first, second))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
            oriented[edge] = (first, second)
    uv = np.asarray(coordinates, np.float64) * (cell / 2)
    count = len(uv)
    positions = np.concatenate(
        (
            np.column_stack((uv, np.full(count, thickness / 2))),
            np.column_stack((uv, np.full(count, -thickness / 2))),
        )
    )
    triangles = list(front)
    triangles.extend((a + count, c + count, b + count) for a, b, c in front)
    for edge, occurrences in edge_counts.items():
        if occurrences != 1:
            continue
        first, second = oriented[edge]
        triangles.extend(((first, first + count, second + count), (first, second + count, second)))
    return positions, np.asarray(triangles, np.int32)
