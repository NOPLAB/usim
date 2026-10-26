"""Select unchanged representative curves by finite 3D overlap, not screen overlap."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import cos, isfinite, radians

import numpy as np
from numpy.typing import NDArray

from .curve_support import CurveSamples, resample_curve, unsupported_curve_length
from .models import PipelineInputError


@dataclass(frozen=True, slots=True)
class CurveSuppression:
    kept_indices: NDArray[np.int64]
    representative_indices: NDArray[np.int64]
    covered_fractions: NDArray[np.float64]
    max_uncovered_lengths: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class _CurveGeometry:
    samples: CurveSamples
    starts: NDArray[np.float64]
    directions: NDArray[np.float64]
    segment_lengths: NDArray[np.float64]
    sample_directions: NDArray[np.float64]
    low: NDArray[np.float64]
    high: NDArray[np.float64]


def _overlap(
    candidate: _CurveGeometry,
    representative: _CurveGeometry,
    distance: float,
    cosine: float,
) -> tuple[float, float]:
    """Test bounded arc samples against actual finite segments and local tangents."""
    covered = np.zeros(len(candidate.samples.xyz), dtype=np.bool_)
    for start in range(0, len(covered), 128):
        stop = min(start + 128, len(covered))
        delta = candidate.samples.xyz[start:stop, None, :] - representative.starts
        projection = np.einsum('bsi,si->bs', delta, representative.directions)
        projection = np.clip(projection, 0, representative.segment_lengths)
        residual = delta - projection[:, :, None] * representative.directions
        near = np.sum(residual * residual, axis=2) <= distance**2
        aligned = (
            np.abs(candidate.sample_directions[start:stop] @ representative.directions.T) >= cosine
        )
        covered[start:stop] = np.any(near & aligned, axis=1)
    fraction = float(candidate.samples.length_weights[covered].sum() / candidate.samples.length)
    return min(1.0, fraction), unsupported_curve_length(candidate.samples, covered)


def suppress_duplicate_curves(
    curves: Sequence[NDArray[np.float64]],
    ranking: NDArray[np.float64],
    *,
    distance: float = 0.03,
    angle_degrees: float = 5.0,
    min_overlap: float = 0.95,
    spacing: float = 0.01,
) -> CurveSuppression:
    """Keep original rows; suppress only against an already retained representative.

    Higher caller-provided ranking wins, then longer arc length, then lower input
    index. Scores need not be probabilities. A loser must have min_overlap of its
    OWN length near similarly directed finite representative segments. Its longest
    uncovered run must also be no longer than distance, protecting unique tails.
    Directions are unoriented. Curves are not clustered transitively or averaged.

    All lengths use the input coordinate unit. Exact repeated endpoints mark closed
    curves. Overlap is an arc-length-weighted, bounded-spacing estimate, not proof
    that nearby physical edges are the same object. Separations below distance can
    be unresolved. Empty input returns typed empty arrays. No input is modified.
    """
    if (
        not isfinite(distance)
        or distance <= 0
        or not isfinite(angle_degrees)
        or not 0 < angle_degrees <= 90
        or not isfinite(min_overlap)
        or not 0 < min_overlap <= 1
        or not isfinite(spacing)
        or spacing <= 0
    ):
        raise PipelineInputError('Invalid curve-overlap distance, angle, fraction or spacing')
    if (
        not isinstance(ranking, np.ndarray)
        or ranking.dtype != np.float64
        or ranking.shape != (len(curves),)
        or not np.isfinite(ranking).all()
    ):
        raise PipelineInputError('Ranking must be finite float64 with one value per curve')
    geometry = []
    for curve in curves:
        samples = resample_curve(curve, spacing=spacing)
        vectors = np.diff(curve, axis=0)
        lengths = np.linalg.norm(vectors, axis=1)
        nonzero = lengths > 0
        starts = curve[:-1][nonzero]
        lengths = lengths[nonzero]
        directions = vectors[nonzero] / lengths[:, None]
        segments = np.minimum(
            np.searchsorted(np.cumsum(lengths), samples.arc_lengths, side='right'),
            len(lengths) - 1,
        )
        geometry.append(
            _CurveGeometry(
                samples,
                starts,
                directions,
                lengths,
                directions[segments],
                curve.min(axis=0),
                curve.max(axis=0),
            )
        )
    indices = np.arange(len(curves), dtype=np.int64)
    lengths = np.asarray([curve.samples.length for curve in geometry], dtype=np.float64)
    order = np.lexsort((indices, -lengths, -ranking))
    representatives = indices.copy()
    fractions = np.ones(len(curves), dtype=np.float64)
    gaps = np.zeros(len(curves), dtype=np.float64)
    kept: list[int] = []
    cosine = cos(radians(angle_degrees))
    for index in order:
        candidate = geometry[index]
        for previous in kept:
            representative = geometry[previous]
            if np.any(candidate.low > representative.high + distance) or np.any(
                representative.low > candidate.high + distance
            ):
                continue
            fraction, gap = _overlap(candidate, representative, distance, cosine)
            if fraction >= min_overlap and gap <= distance:
                representatives[index] = previous
                fractions[index] = fraction
                gaps[index] = gap
                break
        else:
            kept.append(int(index))
    selected = np.sort(np.asarray(kept, dtype=np.int64))
    for result in (selected, representatives, fractions, gaps):
        result.setflags(write=False)
    return CurveSuppression(selected, representatives, fractions, gaps)
