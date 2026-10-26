"""Measured linear-member hypotheses from multiscale features and original depth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

from .edge_line_support import LineSupport, supported_intervals
from .window_contour_features import ContourFeatures
from .window_member_sections import member_section_centres, section_evidence


@dataclass(frozen=True, slots=True)
class MemberAxis:
    """A measured member centre hypothesis, never proof of a two-face crease."""

    endpoints: NDArray[np.float64]
    witnesses: NDArray[np.int64]
    coverage: float
    max_gap_m: float
    bin_origin: NDArray[np.float64]
    thin_fraction: float
    median_section_width_m: float
    feature_points: int
    component_rows: NDArray[np.int64] | None = None


def facade_frame(xyz: NDArray[np.float64]) -> NDArray[np.float64]:
    """Estimate a facade-local frame without a scanner origin or reference line."""
    centered = xyz - np.median(xyz, axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    normal = axes[-1].copy()
    normal[2] = 0
    normal /= np.linalg.norm(normal)
    if normal[np.argmax(np.abs(normal))] < 0:
        normal *= -1
    up = np.asarray([0.0, 0.0, 1.0])
    horizontal = np.cross(up, normal)
    return np.column_stack((horizontal, up, normal))


def _refit(
    xyz: NDArray[np.float64],
    rows: NDArray[np.int64],
    normal: NDArray[np.float64],
    count: int,
    seed_direction: NDArray[np.float64],
    seed_depth: float,
) -> list[MemberAxis]:
    """Fit equal-weight original-point sections, then measure actual finite support."""
    points = xyz[rows]
    origin = np.median(points, axis=0)
    origin += (seed_depth - float(origin @ normal)) * normal
    _, _, axes = np.linalg.svd(points - origin, full_matrices=False)
    direction = axes[0]
    if abs(float(direction @ normal)) > np.sin(np.deg2rad(15)) or abs(
        float(direction @ seed_direction)
    ) < np.cos(np.deg2rad(15)):
        return []
    chosen = rows
    component_members = rows
    for _ in range(2):
        original_t = (points - origin) @ direction
        sections = member_section_centres(
            xyz,
            origin,
            direction,
            normal,
            float(original_t.min()) - 0.15,
            float(original_t.max()) + 0.15,
        )
        centres = sections.centres
        component_members = sections.original_rows
        if len(centres) < 6:
            return []
        origin = centres.mean(axis=0)
        _, _, axes = np.linalg.svd(centres - origin, full_matrices=False)
        direction = axes[0]
        if abs(float(direction @ seed_direction)) < np.cos(np.deg2rad(15)) or abs(
            float(direction @ normal)
        ) > np.sin(np.deg2rad(15)):
            return []
    delta = xyz - origin
    t = delta @ direction
    original_t = (points - origin) @ direction
    chosen = np.flatnonzero(
        (np.linalg.norm(np.cross(delta, direction), axis=1) <= 0.24)
        & (t >= float(original_t.min()) - 0.15)
        & (t <= float(original_t.max()) + 0.15)
    )
    config = LineSupport(
        width=0.10,
        step=0.10,
        max_gap=0.25,
        min_length=0.5,
        min_points=12,
        min_bin_points=1,
        min_coverage=0.7,
    )
    result = []
    for line in supported_intervals(xyz, (chosen,), origin, direction, config):
        thin, width = section_evidence(xyz, line.endpoints, normal)
        assert line.bin_origin is not None
        component_t = (xyz[component_members] - line.endpoints[0]) @ direction
        line_length = float(np.linalg.norm(line.endpoints[1] - line.endpoints[0]))
        component_rows = component_members[(component_t >= 0) & (component_t <= line_length)]
        result.append(
            MemberAxis(
                line.endpoints,
                line.witnesses[0],
                line.coverage,
                line.max_unsupported_gap,
                line.bin_origin,
                thin,
                width,
                count,
                component_rows,
            )
        )
    return result


def member_axis_proposals(
    xyz: NDArray[np.float64],
    features: ContourFeatures,
    *,
    use_centroid: bool = False,
) -> list[MemberAxis]:
    """Propose all facade-plane directions; keep depth modes and original support.

    This is an experimental combination, not an implementation of NWLC or a
    learned semantic window detector. Architectural connectivity is not used
    here. No reference coordinates, window sizes or periodicity are inputs.
    """
    if len(xyz) < 12 or not len(features.query_rows):
        return []
    _, singular, axes = np.linalg.svd(xyz - np.median(xyz, axis=0), full_matrices=False)
    if (
        singular[1] <= 0
        or singular[2] / singular[1] > 0.20
        or abs(float(axes[-1, 2])) > np.sin(np.deg2rad(15))
    ):
        return []
    frame = facade_frame(xyz)
    finite = np.isfinite(features.linearity)
    scores = np.where(finite, features.linearity, -np.inf)
    scales = np.argmax(scores, axis=1)
    indices = np.arange(len(scales))
    selected = (scores[indices, scales] >= 0.55) & (
        features.neighbor_counts[indices, scales] >= 10
    )
    if use_centroid:
        selected |= np.any(
            (features.centroid_displacement_norm >= 0.20)
            & (features.boundary_evidence >= 0.25)
            & (features.neighbor_counts >= 10),
            axis=1,
        )
    query_rows = features.query_rows[selected]
    if len(query_rows) < 12:
        return []
    local = np.asarray(xyz[query_rows] @ frame, dtype=np.float64)
    vectors = features.eigenframe[indices[selected], scales[selected], :, 0] @ frame[:, :2]
    norms = np.linalg.norm(vectors, axis=1)
    valid = norms >= np.cos(np.deg2rad(15))
    query_rows, local, vectors, norms = (
        query_rows[valid],
        local[valid],
        vectors[valid],
        norms[valid],
    )
    vectors /= norms[:, None]
    proposals: list[MemberAxis] = []
    poses: list[tuple[float, float, float, float, float]] = []
    for degrees in range(0, 180, 3):
        angle = np.deg2rad(degrees)
        direction = np.asarray([np.cos(angle), np.sin(angle)])
        radial = np.asarray([-direction[1], direction[0]])
        aligned = np.abs(vectors @ direction) >= np.cos(np.deg2rad(15))
        chosen = np.flatnonzero(aligned)
        if len(chosen) < 12:
            continue
        offsets = local[chosen, :2] @ radial
        origin = float(offsets.min()) - 0.16
        edges = np.arange(origin, float(offsets.max()) + 0.24, 0.08)
        histogram, _ = np.histogram(offsets, edges)
        peaks, _ = find_peaks(histogram, distance=3, height=8)
        for peak in peaks:
            offset = float((edges[peak] + edges[peak + 1]) / 2)
            strip = chosen[np.abs(offsets - offset) <= 0.18]
            if len(strip) < 12:
                continue
            depths = local[strip, 2]
            depth_edges = np.arange(float(depths.min()) - 0.1, float(depths.max()) + 0.16, 0.06)
            depth_hist, _ = np.histogram(depths, depth_edges)
            depth_peaks, _ = find_peaks(depth_hist, distance=3, height=6)
            for dp in depth_peaks:
                depth = float((depth_edges[dp] + depth_edges[dp + 1]) / 2)
                group = strip[np.abs(depths - depth) <= 0.12]
                along = local[group, :2] @ direction
                if len(group) < 12 or np.ptp(along) < 0.5:
                    continue
                lo, hi = float(along.min()), float(along.max())
                duplicate = any(
                    abs(degrees - old[0]) <= 6
                    and abs(offset - old[1]) < 0.12
                    and abs(depth - old[2]) < 0.10
                    and min(hi, old[4]) - max(lo, old[3]) > 0.5 * min(hi - lo, old[4] - old[3])
                    for old in poses
                )
                if duplicate:
                    continue
                poses.append((float(degrees), offset, depth, lo, hi))
                proposals.extend(
                    _refit(
                        xyz,
                        query_rows[group],
                        frame[:, 2],
                        len(group),
                        frame[:, :2] @ direction,
                        depth,
                    )
                )
    return proposals
