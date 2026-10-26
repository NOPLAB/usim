"""Conservative, CPU-only local geometry evidence on immutable weak-SOR neighborhoods."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

import numpy as np
from numpy.typing import NDArray

from .models import PipelineInputError


@dataclass(frozen=True, slots=True)
class GeometricConfig:
    """Experimental meter-scale defaults, not a density rejection rule.

    Residual cutoff = clip(median(r) + mad_multiplier * 1.4826 *
    median(abs(r - median(r))), residual_floor_m, residual_ceiling_m),
    where r contains orthogonal residuals of the final trimmed support only.
    """

    scales: tuple[tuple[int, float], ...] = ((12, 0.4), (32, 1.2), (64, 3.0))
    batch_size: int = 1024
    workers: int = 4
    min_neighbors: int = 6
    trim_iterations: int = 2
    trim_fraction: float = 0.2
    plane_flatness: float = 0.1
    plane_min_ratio: float = 0.05
    line_max_ratio: float = 0.03
    min_spread_m: float = 0.03
    mad_multiplier: float = 3.0
    residual_floor_m: float = 0.03
    residual_ceiling_m: float = 0.12
    min_contradicted_scales: int = 2

    def __post_init__(self) -> None:
        integers = (
            self.batch_size,
            self.workers,
            self.min_neighbors,
            self.trim_iterations,
            self.min_contradicted_scales,
        )
        if any(type(value) is not int or value < 1 for value in integers):
            raise PipelineInputError('Geometry integer parameters must be positive integers')
        if self.min_neighbors < 6 or self.min_contradicted_scales < 2:
            raise PipelineInputError('Geometry requires six neighbors and two contradictions')
        if type(self.scales) is not tuple or not self.scales:
            raise PipelineInputError('Geometry scales must be a nonempty immutable tuple')
        for scale in self.scales:
            if type(scale) is not tuple or len(scale) != 2:
                raise PipelineInputError('Geometry scales require (neighbor count, radius)')
            count, radius = scale
            if type(count) is not int or count < self.min_neighbors:
                raise PipelineInputError('Geometry scale neighbor count is too small')
            if not math.isfinite(radius) or radius <= 0:
                raise PipelineInputError('Geometry radii must be finite and positive')
        if len(set(self.scales)) != len(self.scales):
            raise PipelineInputError('Geometry evidence scales must be distinct')
        floats = (
            self.trim_fraction,
            self.plane_flatness,
            self.plane_min_ratio,
            self.line_max_ratio,
            self.min_spread_m,
            self.mad_multiplier,
            self.residual_floor_m,
            self.residual_ceiling_m,
        )
        if any(not math.isfinite(value) or value <= 0 for value in floats):
            raise PipelineInputError('Geometry thresholds must be finite and positive')
        if not (
            self.trim_fraction < 1
            and self.plane_flatness < 1
            and self.line_max_ratio < self.plane_min_ratio < 1
            and self.residual_floor_m <= self.residual_ceiling_m
        ):
            raise PipelineInputError('Geometry threshold ranges are inconsistent')


@dataclass(frozen=True, slots=True)
class GeometricSelection:
    sor_keep: NDArray[np.bool_]
    candidates: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class GeometricResult:
    keep: NDArray[np.bool_]
    geometry_rejected: NDArray[np.bool_]
    supported_scales: NDArray[np.int32]
    consistent_scales: NDArray[np.int32]
    evaluated: NDArray[np.bool_]
    config: GeometricConfig


DEFAULT_GEOMETRY: Final = GeometricConfig()


def _fit_scale(
    offsets: NDArray[np.float64], valid: NDArray[np.bool_], config: GeometricConfig
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Vectorized PCA, retaining only confident models stable through both trims."""
    # Exact duplicate coordinates do not contribute independent support or PCA weight.
    order = np.lexsort((offsets[:, :, 2], offsets[:, :, 1], offsets[:, :, 0]), axis=1)
    offsets = np.take_along_axis(offsets, order[:, :, None], axis=1)
    valid = np.take_along_axis(valid, order, axis=1)
    distinct = np.ones(valid.shape, dtype=np.bool_)
    distinct[:, 1:] = np.any(np.not_equal(offsets[:, 1:], offsets[:, :-1]), axis=2)
    # Invalid entries are removed before this call by setting their coordinates to inf.
    valid &= distinct
    offsets = np.where(valid[:, :, None], offsets, 0.0)
    confident = np.greater_equal(np.sum(valid, axis=1), config.min_neighbors)
    previous_kind = np.zeros(len(offsets), dtype=np.int8)
    iteration = 0
    while True:
        count = np.sum(valid, axis=1)
        center = np.sum(offsets * valid[:, :, None], axis=1) / np.maximum(count[:, None], 1)
        centered = offsets - center[:, None]
        weighted = centered * valid[:, :, None]
        covariance = (weighted.transpose(0, 2, 1) @ weighted) / np.maximum(count[:, None, None], 1)
        eigenvalues, axes = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        largest = eigenvalues[:, 2]
        middle = eigenvalues[:, 1]
        ratio = middle / np.maximum(largest, np.finfo(np.float64).tiny)
        spread = largest >= config.min_spread_m**2
        line = spread & (ratio <= config.line_max_ratio)
        plane = (
            spread
            & (middle >= config.min_spread_m**2)
            & (ratio >= config.plane_min_ratio)
            & (eigenvalues[:, 0] <= config.plane_flatness * middle)
        )
        kind = np.where(line, 1, np.where(plane, 2, 0))
        confident &= (count >= config.min_neighbors) & (kind != 0)
        if iteration:
            confident &= kind == previous_kind
        previous_kind = kind
        normal_residual = np.abs(np.sum(centered * axes[:, None, :, 0], axis=2))
        # Two perpendicular eigenvectors avoid cancellation in line distances.
        line_residual = np.hypot(normal_residual, np.sum(centered * axes[:, None, :, 1], axis=2))
        residual = np.where(line[:, None], line_residual, normal_residual)
        if iteration == config.trim_iterations:
            break
        ranks = np.argsort(np.where(valid, residual, np.inf), axis=1, kind='stable')
        retain = np.floor(count * (1.0 - config.trim_fraction)).astype(np.int64)
        selected = np.less(np.arange(valid.shape[1])[None], retain[:, None])
        trimmed = np.zeros_like(valid)
        np.put_along_axis(trimmed, ranks, selected, axis=1)
        valid &= trimmed
        iteration += 1

    # Sorted masked medians avoid empty-neighborhood warnings and per-row Python loops.
    def median(values: NDArray[np.float64]) -> NDArray[np.float64]:
        ordered = np.sort(np.where(valid, values, np.inf), axis=1)
        lower = np.maximum((count - 1) // 2, 0)
        upper = np.maximum(count // 2, 0)
        rows = np.arange(len(count))
        return np.where(count > 0, (ordered[rows, lower] + ordered[rows, upper]) / 2, 0)

    location = median(residual)
    mad = median(np.abs(residual - location[:, None]))
    cutoff = np.clip(
        location + config.mad_multiplier * 1.4826 * mad,
        config.residual_floor_m,
        config.residual_ceiling_m,
    )
    query_plane = np.abs(np.sum(center * axes[:, :, 0], axis=1))
    query_line = np.hypot(query_plane, np.sum(center * axes[:, :, 1], axis=1))
    query_residual = np.where(line, query_line, query_plane)
    return confident, confident & (query_residual <= cutoff)


def geometric_outliers(
    points: NDArray[np.float64],
    selection: GeometricSelection,
    config: GeometricConfig = DEFAULT_GEOMETRY,
) -> GeometricResult:
    """Reject candidates only with >=2 confident contradictions and no consistent scale.

    All result arrays use original source order. Only weak-SOR retained candidate
    records are evaluated. All queries use the same weak-SOR support, even after
    a rejection. The query record itself is excluded by index, including duplicate
    ties. Unsupported, underdetermined or nonplanar neighborhoods provide no vote.
    Working fit memory is O(batch_size * max_neighbors); the immutable KDTree and
    original-order result arrays are O(N). No density score enters this decision.
    """
    from scipy.spatial import KDTree

    sor_keep, candidates = selection.sor_keep, selection.candidates
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise PipelineInputError('Geometry requires finite XYZ coordinates')
    for mask in (sor_keep, candidates):
        if mask.shape != (len(points),) or mask.dtype != np.bool_:
            raise PipelineInputError('Geometry masks must be original-order boolean vectors')
    positions = np.asarray(points, dtype=np.float64)
    support_ids = np.flatnonzero(sor_keep)
    query_ids = np.flatnonzero(sor_keep & candidates)
    supported = np.zeros(len(points), dtype=np.int32)
    consistent = np.zeros(len(points), dtype=np.int32)
    evaluated = sor_keep & candidates
    if len(support_ids) > 1 and len(query_ids):
        support = positions[support_ids]
        tree = KDTree(support, leafsize=16)
        local_ids = np.searchsorted(support_ids, query_ids)
        max_neighbors = max(scale[0] for scale in config.scales)
        query_k = min(max_neighbors + 1, len(support))
        for start in range(0, len(query_ids), config.batch_size):
            ids = query_ids[start : start + config.batch_size]
            distances, indices = tree.query(positions[ids], k=query_k, workers=config.workers)
            own = local_ids[start : start + len(ids)]
            distances = np.where(indices == own[:, None], np.inf, distances)
            order = np.argsort(distances, axis=1, kind='stable')
            distances = np.take_along_axis(distances, order, axis=1)[:, :max_neighbors]
            indices = np.take_along_axis(indices, order, axis=1)[:, :max_neighbors]
            offsets = support[indices] - positions[ids, None]
            for neighbors, radius in config.scales:
                valid = distances[:, :neighbors] <= radius
                scale_offsets = np.where(valid[:, :, None], offsets[:, :neighbors], np.inf)
                confident, agrees = _fit_scale(scale_offsets, valid.copy(), config)
                supported[ids] += confident
                consistent[ids] += agrees
    rejected = evaluated & (supported >= config.min_contradicted_scales) & (consistent == 0)
    return GeometricResult(
        sor_keep & ~rejected, rejected, supported, consistent, evaluated, config
    )
