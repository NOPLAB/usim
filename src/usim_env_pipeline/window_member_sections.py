"""Complete original cross-sections for finite member-axis remeasurement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class MemberSections:
    centres: NDArray[np.float64]
    original_rows: NDArray[np.int64]
    extent_centres: NDArray[np.float64]


def section_evidence(
    xyz: NDArray[np.float64], endpoints: NDArray[np.float64], normal: NDArray[np.float64]
) -> tuple[float, float]:
    """Check full original transverse sections, not only the fitted inner witnesses."""
    _, accepted, widths = section_profile(xyz, endpoints, normal)
    valid = np.isfinite(widths)
    if not np.any(valid):
        return 0.0, float('inf')
    return float(np.mean(accepted[valid])), float(np.median(widths[valid]))


def section_profile(
    xyz: NDArray[np.float64], endpoints: NDArray[np.float64], normal: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]]:
    """Retain axial locations of bounded, centred support instead of one global veto."""
    vector = endpoints[1] - endpoints[0]
    length = float(np.linalg.norm(vector))
    direction = vector / length
    radial = np.cross(normal, direction)
    radial /= np.linalg.norm(radial)
    delta = xyz - endpoints[0]
    t, r = delta @ direction, delta @ radial
    depth = delta @ normal - t * float(direction @ normal)
    selected = (t >= 0) & (t <= length) & (np.abs(r) <= 0.60) & (np.abs(depth) <= 0.18)
    bins = np.floor(t[selected] / 0.15).astype(np.int64)
    transverse = r[selected]
    section_points = xyz[selected]
    section_t = t[selected]
    size = int(np.ceil(length / 0.15))
    widths = np.full(size, np.nan, dtype=np.float64)
    centred_and_bounded = np.zeros(size, dtype=np.bool_)
    for number in np.unique(bins):
        if number >= size:
            continue
        values = np.unique(np.floor(transverse[bins == number] / 0.02)) * 0.02
        if len(values) >= 3:
            lo, hi = np.quantile(values, [0.05, 0.95])
            widths[number] = float(hi - lo)
            centred_and_bounded[number] = (
                lo > -0.50 and hi < 0.50 and abs(float((lo + hi) / 2)) <= 0.10
            )
            if centred_and_bounded[number]:
                near = np.abs(section_t - (float(number) + 0.5) * 0.15) <= 0.45
                points = np.unique(section_points[near], axis=0)
                if len(points) < 6:
                    centred_and_bounded[number] = False
                    continue
                _, _, axes = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
                centred_and_bounded[number] = abs(float(axes[0] @ direction)) >= np.cos(
                    np.deg2rad(15)
                )
    return (np.arange(size, dtype=np.float64) + 0.5) * 0.15, centred_and_bounded, widths


def member_section_centres(
    xyz: NDArray[np.float64],
    origin: NDArray[np.float64],
    direction: NDArray[np.float64],
    normal: NDArray[np.float64],
    low: float,
    high: float,
) -> MemberSections:
    """Follow bounded occupied components, not a censored tube through a broad bar.

    Components are occupied 0.02 m depth cells and 0.025 m transverse cells.
    A single empty cell is allowed within a component; separated sheets remain
    separate. The closest component to the source-derived search axis is used.
    Unbounded/wall-like sections are omitted, not converted to centres.
    Legacy centres retain point medians. Extent centres use bounded support
    transverse midpoints so denser sampling on one side does not pull the axis.
    Depth remains a robust observed-surface location, not a completed thickness.
    """
    radial = np.cross(normal, direction)
    radial /= np.linalg.norm(radial)
    delta = xyz - origin
    t, r = delta @ direction, delta @ radial
    depth = delta @ normal - t * float(direction @ normal)
    depth_direction = normal - direction * float(direction @ normal)
    depth_direction /= float(depth_direction @ depth_direction)
    chosen = np.flatnonzero(
        (t >= low) & (t <= high) & (np.abs(r) <= 0.60) & (np.abs(depth) <= 0.22)
    )
    bins = np.floor(t[chosen] / 0.10).astype(np.int64)
    centres = []
    extent_centres = []
    members = []
    for number in np.unique(bins):
        rows = chosen[np.abs(t[chosen] - (float(number) + 0.5) * 0.10) <= 0.15]
        if len(rows) < 3:
            continue
        depth_cells = np.floor(depth[rows] / 0.02).astype(np.int64)
        occupied = np.unique(depth_cells)
        groups = np.split(occupied, np.flatnonzero(np.diff(occupied) > 2) + 1)
        group = min(
            groups,
            key=lambda values: (
                float(np.min(np.abs((values + 0.5) * 0.02))),
                -int(np.count_nonzero(np.isin(depth_cells, values))),
            ),
        )
        rows = rows[np.isin(depth_cells, group)]
        if len(rows) < 3:
            continue
        radial_cells = np.floor(r[rows] / 0.025).astype(np.int64)
        occupied = np.unique(radial_cells)
        groups = np.split(occupied, np.flatnonzero(np.diff(occupied) > 2) + 1)
        group = min(
            groups,
            key=lambda values: (
                float(np.min(np.abs((values + 0.5) * 0.025))),
                -int(np.count_nonzero(np.isin(radial_cells, values))),
            ),
        )
        left, right = float(group[0] * 0.025), float((group[-1] + 1) * 0.025)
        if left <= -0.575 or right >= 0.575 or right - left > 0.70:
            continue
        rows = rows[np.isin(radial_cells, group)]
        points = np.unique(xyz[rows], axis=0)
        if len(points) >= 3:
            centres.append(np.median(points, axis=0))
            extent_centres.append(
                origin
                + float(np.median((points - origin) @ direction)) * direction
                + 0.5 * float(r[rows].min() + r[rows].max()) * radial
                + float(np.median(depth[rows])) * depth_direction
            )
            members.append(rows)
    return MemberSections(
        np.asarray(centres, dtype=np.float64).reshape(-1, 3),
        np.unique(np.concatenate(members)) if members else np.empty(0, dtype=np.int64),
        np.asarray(extent_centres, dtype=np.float64).reshape(-1, 3),
    )
