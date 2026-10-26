"""Remove covered subpaths while retaining the original polyline geometry."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import cos, isfinite, radians

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .curve_support import resample_curve
from .models import PipelineInputError


@dataclass(frozen=True, slots=True)
class CurveFragment:
    source_index: int
    start_arc: float
    end_arc: float
    xyz: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class CurveTrimming:
    retained: tuple[CurveFragment, ...]
    removed: tuple[CurveFragment, ...]
    source_indices: NDArray[np.int64]
    representative_fragments: NDArray[np.int64]
    covered_segments: NDArray[np.float64]
    reference_segments: NDArray[np.float64]


def _covered_cells(
    points: NDArray[np.float64],
    reference: NDArray[np.float64],
    distance: float,
    cosine: float,
) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    """Certify each cell using one actual finite reference segment for BOTH ends."""
    vectors = reference[:, 1] - reference[:, 0]
    lengths = np.linalg.norm(vectors, axis=1)
    directions = vectors / lengths[:, None]
    cell_vectors = np.diff(points, axis=0)
    cell_lengths = np.linalg.norm(cell_vectors, axis=1)
    cell_directions = cell_vectors / cell_lengths[:, None]
    tree = cKDTree((reference[:, 0] + reference[:, 1]) / 2)
    nearby = tree.query_ball_point(
        (points[:-1] + points[1:]) / 2,
        distance + cell_lengths / 2 + lengths.max() / 2,
    )
    counts = np.fromiter((len(rows) for rows in nearby), dtype=np.int64, count=len(nearby))
    hits = np.full(len(cell_lengths), len(reference), dtype=np.int64)
    interior = np.zeros(len(cell_lengths), dtype=np.bool_)
    if counts.sum():
        cells = np.repeat(np.arange(len(cell_lengths)), counts)
        rows = np.concatenate(nearby).astype(np.int64)
        aligned = np.abs(np.sum(cell_directions[cells] * directions[rows], axis=1)) >= cosine
        inside = np.ones(len(rows), dtype=np.bool_)
        for endpoints in (points[:-1], points[1:]):
            delta = endpoints[cells] - reference[rows, 0]
            projection = np.sum(delta * directions[rows], axis=1)
            inside &= (projection >= -1e-12) & (projection <= lengths[rows] + 1e-12)
            clipped = np.clip(projection, 0, lengths[rows])
            residual = delta - clipped[:, None] * directions[rows]
            aligned &= np.sum(residual * residual, axis=1) <= distance**2
        np.minimum.at(hits, cells[aligned], rows[aligned])
        np.logical_or.at(interior, cells[aligned & inside], True)
    hits[hits == len(reference)] = -1
    return hits, interior


def trim_duplicate_curve_overlaps(
    curves: Sequence[NDArray[np.float64]],
    ranking: NDArray[np.float64],
    *,
    distance: float = 0.03,
    angle_degrees: float = 5,
    spacing: float = 0.01,
) -> CurveTrimming:
    """Greedily remove overlap intervals, never average/refit/join original paths.

    Each source edge is subdivided without shortcutting any original vertex.
    A removed cell has both ends within one retained segment's distance capsule
    and an unoriented direction agreement. Capsule convexity covers the entire
    cell. Certificates include both segments and the FINAL retained fragment.
    Only retained fragments become references, so copies cannot cause chaining.

    To preserve endpoint contacts, a covered run shorter than four distances is
    kept unless the entire curve is also covered by finite segment interiors.
    This is geometric equivalence at distance resolution, not physical identity.
    All lengths are input coordinate units. Inputs are never modified.
    """
    if (
        not isfinite(distance)
        or distance <= 0
        or not isfinite(angle_degrees)
        or not 0 < angle_degrees <= 90
        or not isfinite(spacing)
        or spacing <= 0
    ):
        raise PipelineInputError('Invalid overlap trimming distance, angle or spacing')
    if (
        not isinstance(ranking, np.ndarray)
        or ranking.dtype != np.float64
        or ranking.shape != (len(curves),)
        or not np.isfinite(ranking).all()
    ):
        raise PipelineInputError('Ranking must be finite float64 with one value per curve')
    sampled = [resample_curve(curve, spacing=spacing) for curve in curves]
    lengths = np.array([samples.length for samples in sampled], dtype=np.float64)
    order = np.lexsort((np.arange(len(curves)), -lengths, -ranking))
    retained: list[CurveFragment] = []
    removed: list[CurveFragment] = []
    sources, owners, covered, references = [], [], [], []
    cosine = cos(radians(angle_degrees))
    for index in order:
        curve, samples = curves[index], sampled[index]
        segment_lengths = np.linalg.norm(np.diff(curve, axis=0), axis=1)
        vertices = curve[np.r_[True, segment_lengths > 0]]
        original_arc = np.r_[0.0, np.cumsum(segment_lengths[segment_lengths > 0])]
        arc = np.unique(np.r_[original_arc, samples.arc_lengths, samples.length])
        points = np.column_stack([np.interp(arc, original_arc, vertices[:, i]) for i in range(3)])
        distinct = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 0]
        points, arc = points[distinct], arc[distinct]
        arc[-1] = samples.length
        hits = np.full(len(arc) - 1, -1, dtype=np.int64)
        interior = np.zeros(len(hits), dtype=np.bool_)
        segments = np.empty((0, 2, 3), dtype=np.float64)
        fragment_ids = np.empty(0, dtype=np.int64)
        if retained:
            segments = np.concatenate(
                [np.stack((f.xyz[:-1], f.xyz[1:]), axis=1) for f in retained]
            )
            fragment_ids = np.repeat(
                np.arange(len(retained), dtype=np.int64), [len(f.xyz) - 1 for f in retained]
            )
            positive = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1) > 0
            segments, fragment_ids = segments[positive], fragment_ids[positive]
            hits, interior = _covered_cells(points, segments, distance, cosine)
        mask = hits >= 0
        bounds = np.flatnonzero(np.r_[True, mask[1:] != mask[:-1], True])
        full_interior = bool(mask.all() and interior.all())
        for start, stop in zip(bounds[:-1], bounds[1:], strict=True):
            run_length = arc[stop] - arc[start]
            if samples.closed and mask[0] and mask[-1] and len(bounds) > 2:
                if start == 0:
                    run_length += arc[-1] - arc[bounds[-2]]
                elif stop == len(mask):
                    run_length += arc[bounds[1]]
            if mask[start] and run_length < 4 * distance and not full_interior:
                hits[start:stop] = -1
        mask = hits >= 0
        if mask.any():
            sources.append(np.full(mask.sum(), index, dtype=np.int64))
            owners.append(fragment_ids[hits[mask]])
            covered.append(np.stack((points[:-1][mask], points[1:][mask]), axis=1))
            references.append(segments[hits[mask]])
        if not mask.any():
            retained.append(CurveFragment(int(index), 0, samples.length, curve.copy()))
            continue
        bounds = np.flatnonzero(np.r_[True, mask[1:] != mask[:-1], True])
        for start, stop in zip(bounds[:-1], bounds[1:], strict=True):
            fragment = CurveFragment(
                int(index), float(arc[start]), float(arc[stop]), points[start : stop + 1].copy()
            )
            (removed if mask[start] else retained).append(fragment)
    arrays = (
        np.concatenate(sources) if sources else np.empty(0, dtype=np.int64),
        np.concatenate(owners) if owners else np.empty(0, dtype=np.int64),
        np.concatenate(covered) if covered else np.empty((0, 2, 3), dtype=np.float64),
        np.concatenate(references) if references else np.empty((0, 2, 3), dtype=np.float64),
    )
    for fragment in (*retained, *removed):
        fragment.xyz.setflags(write=False)
    for array in arrays:
        array.setflags(write=False)
    return CurveTrimming(tuple(retained), tuple(removed), *arrays)
