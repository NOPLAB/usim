"""Source-bounded straight polygon approximations of occupied planar cells."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TypeGuard

import numpy as np
from numpy.typing import NDArray
from shapely import constrained_delaunay_triangles, get_parts, normalize, union_all
from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from .models import PipelineInputError


@dataclass(frozen=True, slots=True)
class PolygonSlab:
    positions: NDArray[np.float64]
    faces: NDArray[np.int32]
    area_m2: float
    outline_error_m: float


def _is_polygon(geometry: BaseGeometry) -> TypeGuard[Polygon]:
    return isinstance(geometry, Polygon)


def _polygon(geometry: BaseGeometry) -> Polygon:
    """Parse a native scalar polygon result without erasing its geometry type."""
    if _is_polygon(geometry):
        return geometry
    raise PipelineInputError('Native planar fitting returned a non-polygon geometry')


def _within(first: BaseGeometry, second: BaseGeometry, distance: float) -> bool:
    """Conservative continuous boundary bound, not sampled vertex distance."""
    return first.covered_by(second.buffer(distance)) and second.covered_by(first.buffer(distance))


def _footprint(keys: NDArray[np.int64]) -> BaseGeometry:
    rectangles = []
    rows = np.split(keys, np.flatnonzero(np.diff(keys[:, 1])) + 1)
    for row in rows:
        y, xs = row[0, 1], row[:, 0]
        runs = np.split(xs, np.flatnonzero(np.diff(xs) != 1) + 1)
        rectangles.extend(box(int(run[0]), int(y), int(run[-1]) + 1, int(y) + 1) for run in runs)
    return normalize(union_all(rectangles))


def _simplified(source: Polygon, neighbors: list[Polygon]) -> Polygon:
    rings = [source.exterior, *source.interiors]
    clean = [_polygon(Polygon(ring).simplify(0, preserve_topology=True)) for ring in rings]
    hole_centers = [Polygon(ring).representative_point() for ring in source.interiors]
    candidates = [source]
    for tolerance in (1.5, 1.0, 0.5, 0.0):
        proposed = []
        for index, ring in enumerate(clean):
            simplified = (
                ring
                if len(ring.exterior.coords) <= 5
                else _polygon(ring.simplify(tolerance, preserve_topology=True))
            )
            if index and not simplified.contains(hole_centers[index - 1]):
                simplified = ring
            proposed.append(list(simplified.exterior.coords))
        candidate = Polygon(proposed[0], proposed[1:])
        if not candidate.is_valid or not candidate.covers(source.representative_point()):
            continue
        if any(candidate.covers(center) for center in hole_centers):
            continue
        if any(
            not candidate.intersection(other).equals(source.intersection(other))
            for other in neighbors
        ):
            continue
        if _within(source.boundary, candidate.boundary, 1.5):
            candidates.append(candidate)
    chosen = min(
        candidates,
        key=lambda polygon: sum(
            len(ring.coords) for ring in [polygon.exterior, *polygon.interiors]
        ),
    )
    return _polygon(normalize(chosen))


def _outline_error(source: Polygon, result: Polygon) -> float:
    if source.boundary.equals(result.boundary):
        return 0.0
    low, high = 0.0, 1.5
    for _ in range(20):
        middle = (low + high) / 2
        if _within(source.boundary, result.boundary, middle):
            high = middle
        else:
            low = middle
    return high


def _extrude(
    polygons: list[Polygon], offset: NDArray[np.int64], cell: float, thickness: float
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    positions, faces = [], []
    for polygon in polygons:
        triangles = []
        for part in get_parts(constrained_delaunay_triangles(polygon)):
            triangle = _polygon(part)
            vertices = list(triangle.exterior.coords)[:-1]
            a, b, c = np.asarray(vertices)
            if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) < 0:
                vertices.reverse()
            start = vertices.index(min(vertices))
            triangles.append(tuple(vertices[start:] + vertices[:start]))
        triangles.sort()
        coordinates = sorted({point for triangle in triangles for point in triangle})
        lookup = {point: index for index, point in enumerate(coordinates)}
        top = [tuple(lookup[point] for point in triangle) for triangle in triangles]
        count, base = len(coordinates), len(positions)
        uv = (np.asarray(coordinates, np.float64) + offset) * cell
        positions.extend(np.column_stack((uv, np.full(count, thickness / 2))))
        positions.extend(np.column_stack((uv, np.full(count, -thickness / 2))))
        faces.extend(tuple(base + vertex for vertex in triangle) for triangle in top)
        faces.extend((base + a + count, base + c + count, base + b + count) for a, b, c in top)
        edges = {}
        for a, b, c in top:
            for first, second in ((a, b), (b, c), (c, a)):
                key = tuple(sorted((first, second)))
                if key in edges:
                    del edges[key]
                else:
                    edges[key] = (first, second)
        for first, second in sorted(edges.values()):
            a, b = base + first, base + second
            faces.extend(((a, a + count, b + count), (a, b + count, b)))
    return np.asarray(positions, np.float64).reshape(-1, 3), np.asarray(faces, np.int32).reshape(
        -1, 3
    )


def polygon_slab(keys: NDArray[np.int64], cell: float, thickness: float) -> PolygonSlab:
    """Simplify exact cell-union contours, constrain triangulation, and extrude.

    XY follows key*cell coordinates; Z is exactly +/- thickness/2. Simplification
    is bounded by 1.5 cells in both continuous outline directions. Original hole
    centers stay empty and disconnected components never acquire new contacts.
    The zero-tolerance/original rings participate in normal candidate selection.
    Inputs are not mutated; returned array storage is read-only.
    """
    if not isinstance(keys, np.ndarray) or keys.ndim != 2 or keys.shape[1] != 2:
        raise PipelineInputError('Planar keys must be an (N, 2) int64 array')
    if keys.dtype != np.int64:
        raise PipelineInputError('Planar keys must have int64 dtype')
    if not math.isfinite(cell) or cell <= 0 or not math.isfinite(thickness) or thickness <= 0:
        raise PipelineInputError('Cell size and thickness must be finite and positive')
    if not len(keys):
        positions, faces = np.empty((0, 3), np.float64), np.empty((0, 3), np.int32)
        area, error = 0.0, 0.0
    else:
        offset = keys.min(axis=0)
        local = np.unique(keys - offset, axis=0)
        local = local[np.lexsort((local[:, 0], local[:, 1]))]
        original = _footprint(local)
        source = [_polygon(part) for part in get_parts(original)]
        polygons = source.copy()
        tree = STRtree(source)
        for index, polygon in enumerate(source):
            x0, y0, x1, y1 = polygon.bounds
            neighbors = tree.query(box(x0 - 3, y0 - 3, x1 + 3, y1 + 3))
            polygons[index] = _simplified(polygon, [polygons[i] for i in neighbors if i != index])
        area = sum(polygon.area for polygon in polygons) * cell**2
        error = max(_outline_error(a, b) for a, b in zip(source, polygons)) * cell
        positions, faces = _extrude(polygons, offset, cell, thickness)
    positions.setflags(write=False)
    faces.setflags(write=False)
    return PolygonSlab(positions, faces, float(area), float(error))
