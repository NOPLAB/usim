"""Multiscale features on full, unchanged original-point radius neighborhoods."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree

from .models import PipelineInputError


@dataclass(frozen=True, slots=True)
class ContourFeatures:
    """Read-only arrays, with leading dimensions (query, scale).

    ``query_rows`` preserves caller order and repetitions; ``radii`` is in meters
    when XYZ is in meters. Eigenvalues descend; eigenframe *columns* correspond
    to them, so the last column is an unoriented normal when tangent_valid.
    Displacement vectors are in the input coordinate frame, divided by radius.
    Invalid estimates are NaN, never silently filled; counts are always defined.
    """

    query_rows: NDArray[np.int64]
    radii: tuple[float, ...]
    neighbor_counts: NDArray[np.int64]
    eigenvalues: NDArray[np.float64]
    eigenframe: NDArray[np.float64]
    centroid_displacement: NDArray[np.float64]
    centroid_displacement_norm: NDArray[np.float64]
    linearity: NDArray[np.float64]
    planarity: NDArray[np.float64]
    scattering: NDArray[np.float64]
    surface_variation: NDArray[np.float64]
    tangent_valid: NDArray[np.bool_]
    angular_gap: NDArray[np.float64]
    boundary_evidence: NDArray[np.float64]


def contour_features(
    xyz: NDArray[np.float64],
    query_rows: NDArray[np.int64],
    radii: tuple[float, ...],
    *,
    batch_size: int = 256,
    min_neighbors: int = 6,
    min_tangent_ratio: float = 1e-3,
    max_surface_variation: float = 0.1,
) -> ContourFeatures:
    """Return features only; no resampling, surface merging or normal orientation.

    At each radius r, use ALL distinct coordinates p with 0 < ||p-q|| <= r.
    Exact duplicates (including extra copies of q) do not add support. No k cap,
    voxel collapse, scanner origin, annotation or target coordinate is used.
    For d=(p-q)/r, w=exp(-||d||²/2), displacement=sum(w*d)/sum(w).
    This is our simple weighted centroid displacement, NOT a faithful NWLC
    reproduction. Its norm is a rotation-invariant scalar.

    PCA uses unweighted, centered population covariance of d, times r².
    With descending eigenvalues a >= b >= c >= 0, linearity=(a-b)/a,
    planarity=(b-c)/a, scattering=c/a, surface_variation=c/(a+b+c).
    Their interpretation is line-like, planar and volumetric support, not
    calibrated probabilities. Nonempty zero-spread support has zero eigenvalues
    and undefined shape ratios. Empty support has NaN PCA and displacement.

    Tangent evidence requires min_neighbors distinct neighbors, b/a above
    min_tangent_ratio, and surface_variation <= max_surface_variation. Project
    query-relative d onto the first two eigenvectors, omit numerically axial
    projections (length <= 64*eps), sort polar angles and take the largest
    circular gap in radians, including wraparound. At least three projections
    are required. boundary_evidence=gap/(2*pi): a half-plane is about 0.5,
    a densely sampled plane interior about 0. No angles are imputed for lines,
    sparse or volumetric support; tangent_valid marks estimates actually made.

    Eigenvector signs use their largest absolute component positive (first
    component wins exact ties). Repeated eigenvalues have no unique geometric
    eigenframe; its deterministic numerical basis must not be interpreted as
    an oriented normal. Scalar features do not depend on those signs.
    RAM is O(N + Q*S + B*N) worst case: one tree, output, and at most B full
    neighborhoods, never a full-cloud radius graph. Dense radii can still be
    expensive; batch_size bounds simultaneous query neighborhoods, not their k.
    """
    if (
        xyz.ndim != 2
        or xyz.shape[1] != 3
        or xyz.dtype != np.float64
        or not np.isfinite(xyz).all()
        or query_rows.ndim != 1
        or query_rows.dtype != np.int64
        or np.any(query_rows < 0)
        or np.any(query_rows >= len(xyz))
        or not radii
        or any(not isfinite(r) or r <= 0 for r in radii)
        or any(a >= b for a, b in zip(radii, radii[1:]))
        or type(batch_size) is not int
        or batch_size < 1
        or type(min_neighbors) is not int
        or min_neighbors < 3
        or not 0 < min_tangent_ratio < 1
        or not 0 <= max_surface_variation < 1 / 3
    ):
        raise PipelineInputError('Require finite float64 XYZ, int64 rows and increasing radii')
    shape = (len(query_rows), len(radii))
    counts = np.zeros(shape, dtype=np.int64)
    eigenvalues = np.full((*shape, 3), np.nan)
    eigenframe = np.full((*shape, 3, 3), np.nan)
    displacement = np.full((*shape, 3), np.nan)
    displacement_norm = np.full(shape, np.nan)
    linearity = np.full(shape, np.nan)
    planarity = np.full(shape, np.nan)
    scattering = np.full(shape, np.nan)
    variation = np.full(shape, np.nan)
    tangent_valid = np.zeros(shape, dtype=np.bool_)
    gaps = np.full(shape, np.nan)
    evidence = np.full(shape, np.nan)
    tree = KDTree(xyz, copy_data=True)
    eps = 64 * np.finfo(np.float64).eps
    for start in range(0, len(query_rows), batch_size):
        rows = query_rows[start : start + batch_size]
        neighborhoods = tree.query_ball_point(
            xyz[rows], np.nextafter(radii[-1], np.inf), return_sorted=True, workers=1
        )
        for local, ids in enumerate(neighborhoods):
            i = start + local
            delta = np.unique(xyz[ids] - xyz[rows[local]], axis=0)
            distances = np.linalg.norm(delta, axis=1)
            for j, radius in enumerate(radii):
                d = delta[(distances > 0) & (distances <= radius)] / radius
                counts[i, j] = len(d)
                if not len(d):
                    continue
                weights = np.exp(-np.sum(d * d, axis=1) / 2)
                displacement[i, j] = np.sum(weights[:, None] * d, axis=0) / weights.sum()
                displacement_norm[i, j] = np.linalg.norm(displacement[i, j])
                centered = d - d.mean(axis=0)
                values, frame = np.linalg.eigh(centered.T @ centered / len(d))
                values = np.maximum(values[::-1], 0)
                frame = frame[:, ::-1]
                pivots = np.argmax(np.abs(frame), axis=0)
                frame *= np.where(frame[pivots, np.arange(3)] < 0, -1.0, 1.0)
                eigenvalues[i, j] = values * radius**2
                eigenframe[i, j] = frame
                a, b, c = values
                if a <= 0:
                    continue
                linearity[i, j] = (a - b) / a
                planarity[i, j] = (b - c) / a
                scattering[i, j] = c / a
                variation[i, j] = c / values.sum()
                if (
                    len(d) < min_neighbors
                    or b / a <= min_tangent_ratio
                    or variation[i, j] > max_surface_variation
                ):
                    continue
                tangent = d @ frame[:, :2]
                tangent = tangent[np.linalg.norm(tangent, axis=1) > eps]
                if len(tangent) < 3:
                    continue
                angles = np.sort(np.arctan2(tangent[:, 1], tangent[:, 0]))
                gaps[i, j] = np.max(np.diff(np.append(angles, angles[0] + 2 * np.pi)))
                evidence[i, j] = gaps[i, j] / (2 * np.pi)
                tangent_valid[i, j] = True
    result = ContourFeatures(
        query_rows.copy(),
        radii,
        counts,
        eigenvalues,
        eigenframe,
        displacement,
        displacement_norm,
        linearity,
        planarity,
        scattering,
        variation,
        tangent_valid,
        gaps,
        evidence,
    )
    for value in (
        result.query_rows,
        counts,
        eigenvalues,
        eigenframe,
        displacement,
        displacement_norm,
        linearity,
        planarity,
        scattering,
        variation,
        tangent_valid,
        gaps,
        evidence,
    ):
        value.setflags(write=False)
    return result
