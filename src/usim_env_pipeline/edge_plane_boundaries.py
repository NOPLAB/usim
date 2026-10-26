"""Straight face boundaries backed only by original local observations."""

from __future__ import annotations

from math import isfinite

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .edge_line_support import LineSupport, SupportedLine, supported_intervals
from .models import PipelineInputError
from .plane_types import Plane, plane_basis


def observed_boundary_lines(
    xyz: NDArray[np.float64],
    plane: Plane,
    seed_ids: NDArray[np.int64],
    support: LineSupport = LineSupport(),
    *,
    neighborhood_radius: float = 0.3,
    angle_threshold: float = 150.0,
    tangent_tolerance: float = 15.0,
    fit_tolerance: float = 0.05,
    iterations: int = 256,
    max_lines: int = 8,
    seed: int = 7,
) -> list[SupportedLine]:
    """Validate proposals against the whole observed face, then fit finite lines.

    Complete projected radius neighborhoods distinguish true angular vacancies
    from truncated edge-point neighborhoods. The vacancy bisector determines
    an unoriented boundary tangent. Only tangent-compatible, measured boundary
    rows may support the RANSAC fit or its finite intervals; interior face rows
    never extend the line. All indices refer to the unchanged input XYZ.
    """
    if (
        any(
            type(value) not in (int, float) or not isfinite(value) or value <= 0
            for value in (
                neighborhood_radius,
                angle_threshold,
                tangent_tolerance,
                fit_tolerance,
            )
        )
        or angle_threshold > 360
        or tangent_tolerance > 90
        or type(iterations) is not int
        or iterations < 1
        or type(max_lines) is not int
        or max_lines < 1
        or type(seed) is not int
        or seed < 0
    ):
        raise PipelineInputError('Invalid observed-boundary line parameters')
    if (
        xyz.ndim != 2
        or xyz.shape[1] != 3
        or xyz.dtype != np.float64
        or not np.isfinite(xyz).all()
        or plane.normal.shape != (3,)
        or plane.normal.dtype != np.float64
        or not np.isfinite(plane.normal).all()
        or not np.isclose(np.linalg.norm(plane.normal), 1, rtol=0, atol=1e-8)
        or type(plane.offset) not in (int, float)
        or not isfinite(plane.offset)
        or plane.ids.ndim != 1
        or plane.ids.dtype != np.int64
        or seed_ids.ndim != 1
        or seed_ids.dtype != np.int64
        or np.any(plane.ids < 0)
        or np.any(plane.ids >= len(xyz))
        or np.any(seed_ids < 0)
        or np.any(seed_ids >= len(xyz))
    ):
        raise PipelineInputError('Boundary lines require finite XYZ, a unit plane and int64 IDs')
    members = np.unique(plane.ids)
    proposals = np.intersect1d(members, seed_ids)
    if len(proposals) < support.min_points:
        return []
    basis = np.column_stack(plane_basis(plane.normal))
    reference = xyz[members[0]]
    coordinates = (xyz[members] - reference) @ basis
    tree = cKDTree(coordinates)
    local_ids = np.searchsorted(members, proposals)
    tangents = np.zeros((len(proposals), 2), dtype=np.float64)
    boundary = np.zeros(len(proposals), dtype=np.bool_)
    angle_roundoff = np.finfo(np.float64).eps * 64
    for index, local_id in enumerate(local_ids):
        neighbors = tree.query_ball_point(
            coordinates[local_id], np.nextafter(neighborhood_radius, np.inf)
        )
        delta = coordinates[neighbors] - coordinates[local_id]
        directions = delta[np.sum(delta * delta, axis=1) > 1e-24]
        angles = np.unique(np.arctan2(directions[:, 1], directions[:, 0]))
        if len(angles) < 8:
            continue
        gaps = np.diff(np.append(angles, angles[0] + 2 * np.pi))
        largest = int(np.argmax(gaps))
        if gaps[largest] + angle_roundoff < np.deg2rad(angle_threshold):
            continue
        bisector = angles[largest] + gaps[largest] / 2
        tangents[index] = [-np.sin(bisector), np.cos(bisector)]
        boundary[index] = True
    ids = proposals[boundary]
    points = coordinates[local_ids[boundary]]
    tangents = tangents[boundary]
    generator = np.random.default_rng(seed)
    cosine = np.cos(np.deg2rad(tangent_tolerance))
    results: list[SupportedLine] = []
    # Each pass consumes its fitted observations even when finite support fails.
    # A sparse proposal cannot repeatedly displace another boundary candidate.
    for _ in range(max_lines):
        if len(ids) < support.min_points or len(results) >= max_lines:
            break
        best = np.zeros(len(ids), dtype=np.bool_)
        best_count = 0
        best_error = np.inf
        for _ in range(iterations):
            pair = generator.choice(len(ids), 2, replace=False)
            direction = points[pair[1]] - points[pair[0]]
            length = float(np.linalg.norm(direction))
            if length <= 1e-12:
                continue
            direction /= length
            perpendicular = np.array([-direction[1], direction[0]])
            distance = np.abs((points - points[pair[0]]) @ perpendicular)
            inliers = (distance <= fit_tolerance) & (
                np.abs(tangents @ direction) + angle_roundoff >= cosine
            )
            count = int(np.count_nonzero(inliers))
            error = float(np.sum(distance[inliers] ** 2))
            if count > best_count or (count == best_count and error < best_error):
                best, best_count, best_error = inliers, count, error
        if best_count < support.min_points:
            break
        # Orthogonal least squares, followed by tangent and distance validation
        # of the refitted model rather than trusting the sampled pair.
        center = points[best].mean(axis=0)
        _, _, axes = np.linalg.svd(points[best] - center, full_matrices=False)
        direction = axes[0]
        perpendicular = np.array([-direction[1], direction[0]])
        fitted = (np.abs((points - center) @ perpendicular) <= fit_tolerance) & (
            np.abs(tangents @ direction) + angle_roundoff >= cosine
        )
        if np.count_nonzero(fitted) >= support.min_points:
            origin = (
                reference
                - (reference @ plane.normal + plane.offset) * plane.normal
                + basis @ center
            )
            direction3 = basis @ direction
            if direction3[np.argmax(np.abs(direction3))] < 0:
                direction3 = -direction3
            intervals = supported_intervals(xyz, (ids[fitted],), origin, direction3, support)
            for line in intervals[: max_lines - len(results)]:
                line.endpoints.setflags(write=False)
                for witnesses in line.witnesses:
                    witnesses.setflags(write=False)
                results.append(line)
        remaining = ~(best | fitted)
        ids, points, tangents = ids[remaining], points[remaining], tangents[remaining]
    return results
