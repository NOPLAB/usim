"""Reconcile finite ownership spans without inventing support or joining known holes."""

from __future__ import annotations

from dataclasses import replace
from itertools import permutations

import numpy as np
from numpy.typing import NDArray

from .edge_line_refit import refit_edge_union
from .edge_line_support import face_strips, plane_intersection, supported_intervals
from .edge_line_types import EdgeLineConfig, MeasuredEdge
from .plane_types import Plane


def _frame(
    edge: MeasuredEdge, config: EdgeLineConfig
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    if edge.kind == 2:
        first_plane: NDArray[np.float64] = edge.equations[0]
        second_plane: NDArray[np.float64] = edge.equations[1]
        return plane_intersection(first_plane, second_plane, config.min_dihedral)
    first: NDArray[np.float64] = edge.line.endpoints[0]
    last: NDArray[np.float64] = edge.line.endpoints[1]
    direction = last - first
    length = float(np.linalg.norm(direction))
    if length == 0:
        return None
    direction /= length
    if direction[np.argmax(np.abs(direction))] < 0:
        direction = -direction
    return first - (first @ direction) * direction, direction


def _roundoff(edge: MeasuredEdge) -> float:
    scale: float = np.abs(edge.line.endpoints).max()
    return float(np.finfo(np.float64).eps * 128 * max(1.0, scale))


def _order(
    edge: MeasuredEdge,
) -> tuple[int, tuple[float, ...], tuple[tuple[int, ...], ...], list[tuple[float, ...]]]:
    points: list[list[float]] = edge.line.endpoints.tolist()
    equations: list[float] = edge.equations.ravel().tolist()
    witnesses: list[list[int]] = [np.sort(ids).tolist() for ids in edge.line.witnesses]
    endpoints = sorted(tuple(row) for row in points)
    return (
        -edge.kind,
        tuple(equations),
        tuple(tuple(ids) for ids in witnesses),
        endpoints,
    )


def _same_feature(
    xyz: NDArray[np.float64],
    first: MeasuredEdge,
    second: MeasuredEdge,
    config: EdgeLineConfig,
) -> bool:
    """Closeness proposes a match; exact geometry or shared original faces proves it."""
    a, b = _frame(first, config), _frame(second, config)
    if a is None or b is None:
        return False
    epsilon = max(_roundoff(first), _roundoff(second))
    coincident = True
    for edge, own, (origin, direction) in ((first, a, b), (second, b, a)):
        if any(len(ids) < config.support.min_points for ids in edge.line.witnesses):
            return False
        source_origin, source_direction = own
        parameters = [(xyz[ids] - source_origin) @ source_direction for ids in edge.line.witnesses]
        limits = [max(float(t.min()) for t in parameters), min(float(t.max()) for t in parameters)]
        endpoints = source_origin + np.asarray(limits)[:, None] * source_direction
        # Compare the unchanged pre-ownership evidence, not arbitrary owned cuts.
        delta = endpoints - origin
        distance: NDArray[np.float64] = np.linalg.norm(
            delta - (delta @ direction)[:, None] * direction, axis=1
        )
        if np.any(distance > config.plane_tolerance):
            return False
        coincident &= bool(np.all(distance <= epsilon))
    # A boundary can match either crease face; two creases must match both.
    if first.kind > second.kind:
        first, second = second, first
    for mapping in permutations(range(second.kind), first.kind):
        matched = True
        for face, other_face in enumerate(mapping):
            equation: NDArray[np.float64] = first.equations[face]
            other: NDArray[np.float64] = second.equations[other_face]
            if equation[:3] @ other[:3] < 0:
                other = -other
            exact = bool(
                np.allclose(equation[:3], other[:3], atol=1e-12, rtol=0)
                and abs(equation[3] - other[3]) <= epsilon
            )
            if exact and coincident:
                continue
            shared = np.intersect1d(first.line.witnesses[face], second.line.witnesses[other_face])
            # Merely similar equations can describe different nearby physical plates.
            # No averaging or transitive distance clustering establishes their identity.
            if len(shared) < config.support.min_points:
                matched = False
                break
            points = xyz[shared]
            first_residual: NDArray[np.float64] = points @ equation[:3] + equation[3]
            second_residual: NDArray[np.float64] = points @ other[:3] + other[3]
            if np.any(np.abs(first_residual) > config.plane_tolerance) or np.any(
                np.abs(second_residual) > config.plane_tolerance
            ):
                matched = False
                break
        if matched:
            return True
    return False


def _validated(
    xyz: NDArray[np.float64],
    representative: MeasuredEdge,
    members: NDArray[np.int64],
    spans: list[tuple[float, float]],
    config: EdgeLineConfig,
) -> list[MeasuredEdge]:
    frame = _frame(representative, config)
    if frame is None:
        return []
    origin, direction = frame
    epsilon = _roundoff(representative)
    parameters = (xyz[members] - origin) @ direction
    equations = representative.equations
    results: list[MeasuredEdge] = []
    for low, high in spans:
        selected = members[(parameters >= low - epsilon) & (parameters <= high + epsilon)]
        if representative.kind == 2:
            groups = face_strips(
                xyz,
                selected,
                Plane(equations[0, :3], float(equations[0, 3]), selected),
                Plane(equations[1, :3], float(equations[1, 3]), selected),
                origin,
                direction,
                config.support,
                plane_tolerance=config.plane_tolerance,
                normal_advantage=config.normal_advantage,
            )
        else:
            residual: NDArray[np.float64] = xyz[selected] @ equations[0, :3] + equations[0, 3]
            residual = np.abs(residual)
            groups = (selected[residual <= config.plane_tolerance],)
        for line in supported_intervals(xyz, groups, origin, direction, config.support):
            begin, end = (line.endpoints - origin) @ direction
            anchors = []
            for ids in line.witnesses:
                t = (xyz[ids] - origin) @ direction
                # Keep the nearest outward anchors on independently sampled faces.
                # Removing those alternately erodes a real common interval to zero.
                first = t[t <= begin + epsilon].max()
                last = t[t >= end - epsilon].min()
                anchors.append(ids[(t >= first - epsilon) & (t <= last + epsilon)])
            # Recompute scores on these actual observations, never inherited clips.
            for fresh in supported_intervals(
                xyz, tuple(anchors), origin, direction, config.support
            ):
                results.append(replace(representative, line=fresh, equations=equations.copy()))
    return results


def reconcile_edge_lines(
    xyz: NDArray[np.float64], edges: list[MeasuredEdge], config: EdgeLineConfig
) -> list[MeasuredEdge]:
    """Revalidate compatible predictions on unchanged global original observations.

    Inputs may be short ownership clips of already supported candidates. Their
    witness arrays still refer to global XYZ, including pre-clip halo observations.
    Only overlapping/touching owned intervals are united. Distinct components stay
    separate even when their gap is below the sampling-gap ceiling.

    Refit unions from original parent faces, not averaged prediction endpoints.
    Nonidentical models need shared original witnesses on every matched face.
    A failed union refit leaves the original hypotheses separate.
    Boundary witnesses must already be validated boundary proposals, not arbitrary
    plane inliers. Crease precedence never turns those rows into a second face.
    """
    clusters: list[list[MeasuredEdge]] = []
    for edge in sorted(edges, key=_order):
        for cluster in clusters:
            if edge.kind == cluster[0].kind and all(
                _same_feature(xyz, member, edge, config) for member in cluster
            ):
                cluster.append(edge)
                break
        else:
            clusters.append([edge])
    validated: list[MeasuredEdge] = []
    for cluster in clusters:
        # Prefer the existing plane model with the lowest measured face residual,
        # independently of input order. No synthetic averaged equations are made.
        representative = min(
            cluster,
            key=lambda edge: (
                sum(
                    float(np.mean(np.abs(xyz[ids] @ equation[:3] + equation[3])))
                    if len(ids)
                    else float('inf')
                    for ids, equation in zip(edge.line.witnesses, edge.equations)
                ),
                _order(edge),
            ),
        )
        fitted = refit_edge_union(
            xyz,
            [representative, *(edge for edge in cluster if edge is not representative)],
            config,
        )
        if fitted is None:
            for edge in cluster:
                frame = _frame(edge, config)
                if frame is None:
                    continue
                origin, direction = frame
                low, high = sorted(float(t) for t in (edge.line.endpoints - origin) @ direction)
                members = np.unique(np.concatenate(edge.line.witnesses))
                validated.extend(_validated(xyz, edge, members, [(low, high)], config))
            continue
        representative = fitted
        frame = _frame(representative, config)
        if frame is None:
            continue
        origin, direction = frame
        intervals = sorted(
            tuple(sorted(float(t) for t in (edge.line.endpoints - origin) @ direction))
            for edge in cluster
        )
        spans: list[tuple[float, float]] = []
        epsilon = max(_roundoff(edge) for edge in cluster)
        for low, high in intervals:
            if spans and low <= spans[-1][1] + epsilon:
                spans[-1] = (spans[-1][0], max(high, spans[-1][1]))
            else:
                spans.append((low, high))
        members = np.unique(
            np.concatenate([ids for edge in cluster for ids in edge.line.witnesses])
        )
        validated.extend(_validated(xyz, representative, members, spans, config))
    creases = [edge for edge in validated if edge.kind == 2]
    results = list(creases)
    for edge in validated:
        if edge.kind != 1:
            continue
        frame = _frame(edge, config)
        if frame is None:
            continue
        origin, direction = frame
        low, high = sorted(float(t) for t in (edge.line.endpoints - origin) @ direction)
        spans = [(low, high)]
        for crease in creases:
            if not _same_feature(xyz, edge, crease, config):
                continue
            begin, end = sorted(float(t) for t in (crease.line.endpoints - origin) @ direction)
            uncovered: list[tuple[float, float]] = []
            for low, high in spans:
                if end <= low or begin >= high:
                    uncovered.append((low, high))
                else:
                    if low < begin:
                        uncovered.append((low, begin))
                    if end < high:
                        uncovered.append((end, high))
            spans = uncovered
        results.extend(_validated(xyz, edge, edge.line.witnesses[0], spans, config))
    return sorted(results, key=_order)
