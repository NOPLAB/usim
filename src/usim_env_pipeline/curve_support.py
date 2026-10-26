"""Arc-length proximity audits, not verification of physical edges."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, isfinite

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .models import PipelineInputError


@dataclass(frozen=True, slots=True)
class CurveSamples:
    xyz: NDArray[np.float64]
    arc_lengths: NDArray[np.float64]
    length_weights: NDArray[np.float64]
    length: float
    closed: bool


@dataclass(frozen=True, slots=True)
class CurveSupportAudit:
    samples: CurveSamples
    nearest_distances: NDArray[np.float64]
    nearest_source_ids: NDArray[np.int64]
    supported_fraction: float
    p95_distance: float
    max_unsupported_length: float


def _xyz(points: NDArray[np.float64], name: str) -> None:
    if (
        not isinstance(points, np.ndarray)
        or points.dtype != np.float64
        or points.ndim != 2
        or points.shape[1] != 3
        or not np.isfinite(points).all()
    ):
        raise PipelineInputError(f'{name} must be a finite float64 array of shape (N, 3)')


def resample_curve(curve: NDArray[np.float64], *, spacing: float = 0.025) -> CurveSamples:
    """Uniformly sample a polyline; an exactly repeated endpoint denotes closure.

    Open curves retain both endpoints. Closed curves sample the entire closing
    segment but omit the duplicate endpoint. Weights are arc-length Voronoi
    cells, with half cells at open endpoints, and sum to the original length.
    """
    _xyz(curve, 'curve')
    if not isfinite(spacing) or spacing <= 0:
        raise PipelineInputError('spacing must be finite and positive')
    if len(curve) < 2:
        raise PipelineInputError('curve must have at least two distinct points')
    segments = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    if not np.isfinite(segments).all():
        raise PipelineInputError('curve segment lengths must be finite')
    # Remove consecutive duplicates only; retain the final closing segment.
    vertices = curve[np.r_[True, segments > 0]]
    if len(vertices) < 2:
        raise PipelineInputError('curve must have positive length')
    arc = np.r_[0.0, np.cumsum(segments[segments > 0])]
    length = float(arc[-1])
    if not isfinite(length) or not isfinite(length / spacing):
        raise PipelineInputError('curve length and sampling count must be finite')
    closed = bool(np.array_equal(vertices[0], vertices[-1]))
    intervals = max(1, ceil(length / spacing))
    positions = np.linspace(0, length, intervals + 1)
    if closed:
        positions = positions[:-1].copy()
    xyz = np.column_stack([np.interp(positions, arc, vertices[:, i]) for i in range(3)])
    if closed:
        weights = np.full(len(positions), length / intervals, dtype=np.float64)
    else:
        boundaries = np.r_[0.0, (positions[:-1] + positions[1:]) / 2, length]
        weights = np.diff(boundaries)
    return CurveSamples(xyz, positions, weights, length, closed)


def unsupported_curve_length(samples: CurveSamples, supported: NDArray[np.bool_]) -> float:
    """Return the longest uncovered arc-cell run, including a closed seam."""
    if not supported.any():
        return samples.length
    order = np.arange(len(supported))
    if samples.closed:
        # Cut the cycle at a supported cell so a seam-spanning gap stays one run.
        order = np.roll(order, -int(np.flatnonzero(supported)[0]))
    longest = current = 0.0
    for index in order:
        if supported[index]:
            current = 0.0
        else:
            current += float(samples.length_weights[index])
            longest = max(longest, current)
    return longest


def audit_curve_support(
    curves: Sequence[NDArray[np.float64]],
    source_xyz: NDArray[np.float64],
    source_ids: NDArray[np.int64],
    *,
    spacing: float = 0.025,
    support_distance: float = 0.05,
) -> tuple[CurveSupportAudit, ...]:
    """Report nearest original observations and length-weighted proximity metrics.

    A sample represents its arc-length Voronoi cell. Fraction, p95 (weighted
    inverse empirical CDF), and consecutive unsupported length use these cells;
    they are bounded-spacing estimates, not continuous geometric proofs.
    IDs are supplied original source IDs, never indices into a filtered cloud.
    Equidistant nearest observations use the lowest original row index.
    Empty curve sequences return (); empty source clouds and degenerate curves
    raise PipelineInputError. Inputs are never modified.
    """
    _xyz(source_xyz, 'source_xyz')
    if not len(source_xyz):
        raise PipelineInputError('source_xyz must contain at least one observation')
    if (
        not isinstance(source_ids, np.ndarray)
        or source_ids.dtype != np.int64
        or source_ids.shape != (len(source_xyz),)
    ):
        raise PipelineInputError('source_ids must be an int64 array matching source_xyz')
    if not isfinite(spacing) or spacing <= 0:
        raise PipelineInputError('spacing must be finite and positive')
    if not isfinite(support_distance) or support_distance < 0:
        raise PipelineInputError('support_distance must be finite and nonnegative')
    tree = cKDTree(source_xyz)
    audits = []
    for curve in curves:
        samples = resample_curve(curve, spacing=spacing)
        distances, rows = tree.query(samples.xyz, k=1)
        # cKDTree does not specify a tie order; resolve only ambiguous samples.
        if len(source_xyz) > 1:
            pair_distances, _ = tree.query(samples.xyz, k=2)
            for index in np.flatnonzero(pair_distances[:, 0] == pair_distances[:, 1]):
                candidates = tree.query_ball_point(
                    samples.xyz[index], np.nextafter(distances[index], np.inf)
                )
                candidate_rows = np.asarray(candidates, dtype=np.int64)
                squared = np.sum((source_xyz[candidate_rows] - samples.xyz[index]) ** 2, axis=1)
                rows[index] = candidate_rows[squared == squared.min()].min()
        supported = distances <= support_distance
        fraction = float(samples.length_weights[supported].sum() / samples.length)
        order = np.argsort(distances, kind='stable')
        cumulative = np.cumsum(samples.length_weights[order])
        quantile = min(int(np.searchsorted(cumulative, 0.95 * samples.length)), len(order) - 1)
        audits.append(
            CurveSupportAudit(
                samples,
                distances,
                source_ids[rows].copy(),
                fraction,
                float(distances[order[quantile]]),
                unsupported_curve_length(samples, supported),
            )
        )
    return tuple(audits)
