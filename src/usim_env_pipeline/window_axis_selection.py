"""Conservative member-axis competition; no rectangle closure or endpoint snapping."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree

from .edge_line_support import LineSupport, supported_intervals
from .window_member_axes import MemberAxis, facade_frame
from .window_member_sections import section_evidence, section_profile


def split_member_sections(
    xyz: NDArray[np.float64], candidates: list[MemberAxis]
) -> list[tuple[int, MemberAxis]]:
    """Keep locally valid sections; a long wall attachment must not veto a whole bar."""
    normal = facade_frame(xyz)[:, 2]
    config = LineSupport(
        width=0.10,
        step=0.10,
        max_gap=0.25,
        min_length=0.5,
        min_points=12,
        min_bin_points=1,
        min_coverage=0.7,
    )
    parts = []
    for number, axis in enumerate(candidates):
        positions, valid, _ = section_profile(xyz, axis.endpoints, normal)
        occupied = np.flatnonzero(valid)
        if not len(occupied):
            continue
        # At most two intervening section bins, e.g. a measured transverse
        # junction. This never fills a gap in the original axial witnesses.
        breaks = np.flatnonzero(np.diff(occupied) > 3) + 1
        direction = axis.endpoints[1] - axis.endpoints[0]
        direction /= np.linalg.norm(direction)
        t = (xyz[axis.witnesses] - axis.endpoints[0]) @ direction
        for run in np.split(occupied, breaks):
            low = float(positions[run[0]] - 0.075)
            high = float(positions[run[-1]] + 0.075)
            rows = axis.witnesses[(t >= low) & (t <= high)]
            for line in supported_intervals(
                xyz,
                (rows,),
                axis.endpoints[0],
                direction,
                config,
                bin_origin=axis.bin_origin,
            ):
                thin, width = section_evidence(xyz, line.endpoints, normal)
                assert line.bin_origin is not None
                parts.append(
                    (
                        number,
                        MemberAxis(
                            line.endpoints,
                            line.witnesses[0],
                            line.coverage,
                            line.max_unsupported_gap,
                            line.bin_origin,
                            thin,
                            width,
                            axis.feature_points,
                        ),
                    )
                )
    return parts


def competing(first: MemberAxis, second: MemberAxis) -> bool:
    """Suppress only overlapping, nearly identical hypotheses sharing most support."""
    a, b = first.endpoints, second.endpoints
    direction = a[1] - a[0]
    length = float(np.linalg.norm(direction))
    direction /= length
    other = b[1] - b[0]
    other_length = float(np.linalg.norm(other))
    other /= other_length
    cosine = float(direction @ other)
    if abs(cosine) < np.cos(np.deg2rad(8)):
        return False
    positions = (b - a[0]) @ direction
    low = max(0.0, float(positions.min()))
    high = min(length, float(positions.max()))
    if high - low < 0.6 * min(length, other_length):
        return False
    overlap_ends = a[0] + np.asarray([low, high])[:, None] * direction
    residual = np.cross(overlap_ends - b[0], other)
    rms = float(
        np.sqrt(
            (residual[0] @ residual[0] + residual[0] @ residual[1] + residual[1] @ residual[1]) / 3
        )
    )
    complete = first.component_rows is not None and second.component_rows is not None
    tolerance = (
        max(
            0.08,
            min(
                0.15,
                0.5
                * min(
                    first.median_section_width_m,
                    second.median_section_width_m,
                ),
            ),
        )
        if complete
        else 0.08
    )
    if rms > tolerance:
        return False
    first_ids = first.component_rows if first.component_rows is not None else first.witnesses
    second_ids = second.component_rows if second.component_rows is not None else second.witnesses
    shared = len(np.intersect1d(first_ids, second_ids))
    return shared >= max(12, 0.7 * min(len(first_ids), len(second_ids)))


def _connected(first: NDArray[np.float64], second: NDArray[np.float64]) -> bool:
    """Reward measured crossings within both finite spans, without creating a junction."""
    a, b = first[1] - first[0], second[1] - second[0]
    al, bl = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    a, b = a / al, b / bl
    cosine = float(a @ b)
    if abs(cosine) > np.cos(np.deg2rad(25)):
        return False
    matrix = np.column_stack((a, -b))
    parameters = np.linalg.lstsq(matrix, second[0] - first[0], rcond=None)[0]
    if not (0 <= parameters[0] <= al and 0 <= parameters[1] <= bl):
        return False
    first_point = first[0] + parameters[0] * a
    second_point = second[0] + parameters[1] * b
    return float(np.linalg.norm(first_point - second_point)) <= 0.10


def select_member_axes(
    candidates: list[MemberAxis], *, connectivity: bool, require_sections: bool = True
) -> list[int]:
    """Choose supported narrow members; connectivity only breaks local competitions."""
    eligible = [
        i
        for i, axis in enumerate(candidates)
        if not require_sections or axis.thin_fraction >= 0.75
    ]
    scores = []
    for i in eligible:
        axis = candidates[i]
        length = float(np.linalg.norm(axis.endpoints[1] - axis.endpoints[0]))
        neighbors = (
            sum(_connected(axis.endpoints, candidates[j].endpoints) for j in eligible if i != j)
            if connectivity
            else 0
        )
        section_weight = (
            axis.thin_fraction / (0.05 + axis.median_section_width_m) if require_sections else 1.0
        )
        score = length * axis.coverage * section_weight * (1 + 0.03 * min(neighbors, 3))
        scores.append((score, i))
    kept: list[int] = []
    for _, index in sorted(scores, key=lambda item: (-item[0], item[1])):
        if not any(competing(candidates[index], candidates[old]) for old in kept):
            kept.append(index)
    return kept


def direction_field_coverage(
    xyz: NDArray[np.float64],
    axis: MemberAxis,
    query_rows: NDArray[np.int64],
    eigenframe: NDArray[np.float64],
    linearity: NDArray[np.float64],
    counts: NDArray[np.int64],
) -> float:
    """Require sustained original local directions, not just nearby points in a tube."""
    scores = np.where(np.isfinite(linearity), linearity, -np.inf)
    scales = np.argmax(scores, axis=1)
    rows = np.arange(len(query_rows))
    directions = eigenframe[rows, scales, :, 0]
    eligible = (scores[rows, scales] >= 0.55) & (counts[rows, scales] >= 10)
    direction = axis.endpoints[1] - axis.endpoints[0]
    length = float(np.linalg.norm(direction))
    direction /= length
    aligned = np.abs(directions @ direction) >= np.cos(np.deg2rad(15))
    delta = xyz[query_rows] - axis.endpoints[0]
    t = delta @ direction
    distance = np.linalg.norm(np.cross(delta, direction), axis=1)
    chosen = eligible & aligned & (distance <= 0.20) & (t >= 0) & (t <= length)
    bins = np.unique(np.floor(t[chosen] / 0.15).astype(np.int64))
    return len(bins) / (int(np.floor(length / 0.15)) + 1)


def original_orientation_consistency(
    xyz: NDArray[np.float64], axis: MemberAxis, tree: KDTree
) -> tuple[float, float]:
    """Use uncensored spherical support to distinguish a real member from a crossing."""
    direction = axis.endpoints[1] - axis.endpoints[0]
    length = float(np.linalg.norm(direction))
    direction /= length
    positions = np.linspace(0.0, length, int(np.ceil(length / 0.25)) + 1)
    centres = axis.endpoints[0] + positions[:, None] * direction
    neighborhoods = tree.query_ball_point(centres, 0.50, return_sorted=True)
    valid, aligned = 0, 0
    for rows in neighborhoods:
        points = np.unique(xyz[rows], axis=0)
        if len(points) < 10:
            continue
        delta = points - points.mean(axis=0)
        values, frame = np.linalg.eigh(delta.T @ delta / len(points))
        if values[-1] <= 0 or (values[-1] - values[-2]) / values[-1] < 0.35:
            continue
        valid += 1
        aligned += abs(float(frame[:, -1] @ direction)) >= np.cos(np.deg2rad(10))
    return (aligned / valid if valid else 0.0), valid / len(centres)
