"""Finite, observation-supported intervals on fitted 3D lines."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray

from .models import PipelineInputError
from .plane_types import Plane


@dataclass(frozen=True, slots=True)
class LineSupport:
    width: float = 0.15
    step: float = 0.1
    max_gap: float = 0.25
    min_length: float = 1.5
    min_points: int = 12
    min_bin_points: int = 2
    min_coverage: float = 0.75

    def __post_init__(self) -> None:
        if (
            any(
                type(value) not in (int, float) or not isfinite(value) or value <= 0
                for value in (self.width, self.step, self.max_gap, self.min_length)
            )
            or self.max_gap < self.step
            or type(self.min_points) is not int
            or self.min_points < 3
            or type(self.min_bin_points) is not int
            or self.min_bin_points < 1
            or type(self.min_coverage) not in (int, float)
            or not isfinite(self.min_coverage)
            or not 0 < self.min_coverage <= 1
        ):
            raise PipelineInputError('Invalid finite-line support parameters')


@dataclass(frozen=True, slots=True)
class SupportedLine:
    endpoints: NDArray[np.float64]
    witnesses: tuple[NDArray[np.int64], ...]
    coverage: float
    max_unsupported_gap: float
    bin_origin: NDArray[np.float64] | None = None


def face_strips(
    xyz: NDArray[np.float64],
    members: NDArray[np.int64],
    first: Plane,
    second: Plane,
    origin: NDArray[np.float64],
    direction: NDArray[np.float64],
    config: LineSupport,
    *,
    plane_tolerance: float,
    normal_advantage: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Re-query original faces and prevent ambiguous shared returns proving both sides."""
    distances = np.column_stack(
        [np.abs(xyz[members] @ p.normal + p.offset) for p in (first, second)]
    )
    groups = []
    for number, plane in enumerate((first, second)):
        selected = (distances[:, number] <= plane_tolerance) & (
            distances[:, number] + normal_advantage < distances[:, 1 - number]
        )
        ids = members[selected]
        delta = xyz[ids] - origin
        t = delta @ direction
        transverse = delta @ np.cross(plane.normal, direction)
        perpendicular = np.linalg.norm(delta - t[:, None] * direction, axis=1)
        ids, t, transverse = (
            ids[perpendicular <= config.width],
            t[perpendicular <= config.width],
            transverse[perpendicular <= config.width],
        )
        if not len(ids):
            groups.append(ids)
            continue
        cells = np.column_stack((np.floor(t / config.step), np.floor(transverse / 0.05))).astype(
            np.int64
        )
        occupied = np.unique(cells, axis=0)
        bins, count = np.unique(occupied[:, 0], return_counts=True)
        eligible = bins[count >= 2]
        groups.append(ids[np.isin(cells[:, 0], eligible)])
    return groups[0], groups[1]


def plane_intersection(
    first: NDArray[np.float64], second: NDArray[np.float64], min_angle: float = 25
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Return a canonical line; parallel or shallow plane pairs have no feature line."""
    a, b = first[:3], second[:3]
    angle = np.rad2deg(np.arccos(np.clip(abs(a @ b), 0, 1)))
    if angle < min_angle:
        return None
    direction = np.cross(a, b)
    direction /= np.linalg.norm(direction)
    if direction[np.argmax(np.abs(direction))] < 0:
        direction = -direction
    origin = np.linalg.solve(np.stack((a, b, direction)), [-first[3], -second[3], 0])
    return origin, direction


def supported_intervals(
    xyz: NDArray[np.float64],
    groups: tuple[NDArray[np.int64], ...],
    origin: NDArray[np.float64],
    direction: NDArray[np.float64],
    config: LineSupport = LineSupport(),
    *,
    bin_origin: NDArray[np.float64] | None = None,
) -> list[SupportedLine]:
    """Trim to original support, split actual large gaps and reject sparse extended lines.

    Every group must supply measured support across the interval. Both plane
    faces are therefore required for intersections, not just their infinite
    equations. Witnesses are indices into unchanged source XYZ.
    """
    strips = []
    parameters = []
    for ids in groups:
        delta = xyz[ids] - origin
        t = delta @ direction
        distance = np.linalg.norm(delta - t[:, None] * direction, axis=1)
        selected = distance <= config.width
        if np.count_nonzero(selected) < config.min_points:
            return []
        strips.append(ids[selected])
        parameters.append(t[selected])
    low = max(float(t.min()) for t in parameters)
    high = min(float(t.max()) for t in parameters)
    if bin_origin is not None:
        low = float((bin_origin - origin) @ direction)
    if high - low < config.min_length:
        return []
    size = int(np.floor((high - low) / config.step)) + 1
    supported = np.ones(size, dtype=np.bool_)
    indices = []
    # Rigid transforms introduce roundoff at exact bin boundaries, not real gaps.
    bin_roundoff = (
        np.finfo(np.float64).eps
        * 64
        * max(
            1.0,
            float(np.abs(origin).max()) + max(float(np.abs(t).max()) for t in parameters),
        )
        / config.step
    )
    for t in parameters:
        local = np.floor((t - low) / config.step + bin_roundoff).astype(np.int64)
        inside = (local >= 0) & (local < size)
        occupancy = np.bincount(local[inside], minlength=size)
        supported &= occupancy >= config.min_bin_points
        indices.append(local)
    occupied = np.flatnonzero(supported)
    if not len(occupied):
        return []
    # One bin's uncertainty must not hide a gap larger than the promised maximum.
    breaks = np.flatnonzero(np.diff(occupied) * config.step > config.max_gap) + 1
    results = []
    for run in np.split(occupied, breaks):
        start, stop = int(run[0]), int(run[-1])
        coverage = len(run) / (stop - start + 1)
        if coverage < config.min_coverage:
            continue
        witnesses = []
        extents = []
        actual_gap = 0.0
        for ids, t, bins in zip(strips, parameters, indices):
            # Include measured terminal observations in adjacent partial bins,
            # without extending beyond any group's actual support.
            selected = (bins >= start - 1) & (bins <= stop + 1)
            if np.count_nonzero(selected) < config.min_points:
                break
            ordered = np.sort(t[selected])
            gap = float(np.diff(ordered).max(initial=0))
            actual_gap = max(actual_gap, gap)
            if gap > config.max_gap:
                break
            witnesses.append(ids[selected])
            extents.append((float(ordered[0]), float(ordered[-1])))
        if len(witnesses) != len(groups):
            continue
        begin, end = max(item[0] for item in extents), min(item[1] for item in extents)
        if end - begin < config.min_length:
            continue
        endpoints = origin + np.array([begin, end])[:, None] * direction
        results.append(
            SupportedLine(
                endpoints, tuple(witnesses), coverage, actual_gap, origin + low * direction
            )
        )
    return results
