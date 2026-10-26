"""Length-based evaluation of finite, typed 3D edge segments in metres."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, fsum, isfinite, radians, sqrt
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .models import PipelineInputError

IntegerScalar: TypeAlias = (
    np.int8 | np.int16 | np.int32 | np.int64 | np.uint8 | np.uint16 | np.uint32 | np.uint64
)


@dataclass(frozen=True, slots=True)
class EdgeTypeQuality:
    kind: Literal[1, 2]
    raw_predicted_length: float
    predicted_length: float
    reference_length: float
    matched_predicted_length: float
    matched_reference_length: float
    duplicate_prediction_length: float
    false_positive_length: float
    precision: float | None
    recall: float | None
    fragment_counts: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class EdgeQuality:
    boundary: EdgeTypeQuality
    crease: EdgeTypeQuality
    distance_tolerance: float
    angle_tolerance_degrees: float


def _geometry(
    segments: NDArray[np.float64], kinds: NDArray[IntegerScalar], name: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if (
        not isinstance(segments, np.ndarray)
        or segments.dtype != np.float64
        or segments.ndim != 3
        or segments.shape[1:] != (2, 3)
        or not np.isfinite(segments).all()
    ):
        raise PipelineInputError(
            f'{name} segments must be finite float64 arrays of shape (N, 2, 3)'
        )
    if (
        not isinstance(kinds, np.ndarray)
        or kinds.shape != (len(segments),)
        or not np.issubdtype(kinds.dtype, np.integer)
        or not np.isin(kinds, (1, 2)).all()
    ):
        raise PipelineInputError(
            f'{name} kinds must be an integer (N,) array containing only 1 or 2'
        )
    with np.errstate(over='ignore', invalid='ignore'):
        vectors = segments[:, 1] - segments[:, 0]
        lengths = np.hypot.reduce(vectors, axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise PipelineInputError(f'{name} segments must have finite, positive lengths')
    return vectors / lengths[:, None], lengths


def _union(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for low, high in sorted(intervals):
        if high <= low:
            continue
        if result and low <= result[-1][1]:
            result[-1] = (result[-1][0], max(high, result[-1][1]))
        else:
            result.append((low, high))
    return result


def _roundoff(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
    # This is coordinate roundoff, not the positional matching tolerance.
    return (
        64
        * float(np.finfo(np.float64).eps)
        * max(1.0, float(np.abs(first).max()), float(np.abs(second).max()))
    )


def _unique_intervals(
    segments: NDArray[np.float64],
    directions: NDArray[np.float64],
    lengths: NDArray[np.float64],
) -> list[list[tuple[float, float]]]:
    """Assign coincident spans to their first prediction, without filling gaps."""
    unique: list[list[tuple[float, float]]] = []
    for index, (segment, direction, length) in enumerate(zip(segments, directions, lengths)):
        duplicates: list[tuple[float, float]] = []
        for previous in segments[:index]:
            delta = previous - segment[0]
            distance = np.linalg.norm(np.cross(delta, direction), axis=1)
            if np.any(distance > _roundoff(segment, previous)):
                continue
            projected = delta @ direction
            low = max(0.0, float(projected.min()))
            high = min(float(length), float(projected.max()))
            if high > low:
                duplicates.append((low, high))
        remaining: list[tuple[float, float]] = []
        start = 0.0
        for low, high in _union(duplicates):
            if low > start:
                remaining.append((start, low))
            start = high
        if length > start:
            remaining.append((start, float(length)))
        unique.append(remaining)
    return unique


def _match_interval(
    predicted: NDArray[np.float64],
    direction: NDArray[np.float64],
    length: float,
    reference: NDArray[np.float64],
    reference_direction: NDArray[np.float64],
    reference_length: float,
    distance_tolerance: float,
    minimum_cosine: float,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    cosine = float(direction @ reference_direction)
    if cosine == 0 or abs(cosine) < minimum_cosine - 64 * np.finfo(np.float64).eps:
        return None
    delta = predicted[0] - reference[0]
    offset = float(delta @ reference_direction)
    # Orthogonal projection must lie inside the finite reference, with no
    # endpoint caps: a nearby but longitudinally disjoint line is not a hit.
    ends = sorted((-offset / cosine, (reference_length - offset) / cosine))
    low, high = max(0.0, ends[0]), min(length, ends[1])
    if high <= low:
        return None
    tolerance = distance_tolerance + _roundoff(predicted, reference)
    perpendicular = np.cross(delta, reference_direction)
    slope = np.cross(direction, reference_direction)
    first_distance = float(np.linalg.norm(perpendicular + low * slope))
    last_distance = float(np.linalg.norm(perpendicular + high * slope))
    # Squared distance is convex along the line. Testing both endpoints is
    # sufficient for the whole interval, but testing only its centre is not.
    if max(first_distance, last_distance) > tolerance:
        slope_length = float(np.linalg.norm(slope))
        if slope_length == 0:
            return None
        middle = (low + high) / 2
        at_middle = perpendicular + middle * slope
        unit_slope = slope / slope_length
        along = float(at_middle @ unit_slope)
        closest = at_middle - along * unit_slope
        square_radius = tolerance**2 - float(closest @ closest)
        if square_radius <= 0:
            return None
        centre = middle - along / slope_length
        radius = sqrt(square_radius) / slope_length
        low, high = max(low, centre - radius), min(high, centre + radius)
        if high <= low:
            return None
    projected = sorted((offset + cosine * low, offset + cosine * high))
    return (low, high), (
        max(0.0, projected[0]),
        min(reference_length, projected[1]),
    )


def _evaluate_type(
    kind: Literal[1, 2],
    predicted: NDArray[np.float64],
    directions: NDArray[np.float64],
    lengths: NDArray[np.float64],
    reference: NDArray[np.float64],
    reference_directions: NDArray[np.float64],
    reference_lengths: NDArray[np.float64],
    distance_tolerance: float,
    minimum_cosine: float,
) -> EdgeTypeQuality:
    unique = _unique_intervals(predicted, directions, lengths)
    predicted_matches: list[list[tuple[float, float]]] = [[] for _ in predicted]
    reference_matches: list[list[tuple[float, float]]] = [[] for _ in reference]
    fragments = [0 for _ in reference]
    for i, (segment, direction, length) in enumerate(zip(predicted, directions, lengths)):
        for j, (target, target_direction, target_length) in enumerate(
            zip(reference, reference_directions, reference_lengths)
        ):
            match = _match_interval(
                segment,
                direction,
                float(length),
                target,
                target_direction,
                float(target_length),
                distance_tolerance,
                minimum_cosine,
            )
            if match is not None:
                predicted_matches[i].append(match[0])
                reference_matches[j].append(match[1])
                fragments[j] += 1
    raw_length = fsum(lengths)
    predicted_length = fsum(high - low for intervals in unique for low, high in intervals)
    matched_predicted_length = fsum(
        max(0.0, min(high, end) - max(low, start))
        for owned, matched in zip(unique, predicted_matches)
        for low, high in owned
        for start, end in _union(matched)
    )
    reference_length = fsum(reference_lengths)
    matched_reference_length = fsum(
        high - low for intervals in reference_matches for low, high in _union(intervals)
    )
    return EdgeTypeQuality(
        kind=kind,
        raw_predicted_length=raw_length,
        predicted_length=predicted_length,
        reference_length=reference_length,
        matched_predicted_length=matched_predicted_length,
        matched_reference_length=matched_reference_length,
        duplicate_prediction_length=max(0.0, raw_length - predicted_length),
        false_positive_length=max(0.0, predicted_length - matched_predicted_length),
        precision=matched_predicted_length / predicted_length if predicted_length else None,
        recall=matched_reference_length / reference_length if reference_length else None,
        fragment_counts=tuple(fragments),
    )


def evaluate_edge_quality(
    predicted: NDArray[np.float64],
    predicted_kinds: NDArray[IntegerScalar],
    reference: NDArray[np.float64],
    reference_kinds: NDArray[IntegerScalar],
    *,
    distance_tolerance: float = 0.10,
    angle_tolerance_degrees: float = 5.0,
) -> EdgeQuality:
    """Evaluate boundary (1) and crease (2) lengths separately, without mutation.

    Coordinates and returned lengths are metres. Arrays must be finite float64
    (N, 2, 3) segments of positive length, with integer (N,) kinds. Tolerances
    require distance > 0 and 0 <= angle < 90 degrees. Endpoint order is irrelevant.

    A prediction point matches its orthogonal projection onto a same-kind
    reference only inside that finite reference, within the direction tolerance
    and within the distance tolerance over the *whole* credited interval.
    Analytic clipping, not sampling or centre distance, determines coverage.
    Predicted and reference matched lengths use their respective arc lengths;
    the latter is the projected interval, not the angled prediction's length.

    Precision uses unique predicted length. Only coincident collinear prediction
    overlap is removed (coordinate-scaled float64 roundoff); nearby distinct lines
    remain separate even inside the matching tolerance. Duplicate length is the
    raw minus unique length. Interval unions prevent repeated credit on either
    side. False-positive length is unique unmatched predicted length, including
    the entire unique output in an edge-free region. Zero denominators give None.

    Fragment counts count original predictions with positive matched length per
    reference, in same-kind reference input order, including duplicate fragments.
    References are independent annotations: overlapping reference annotations
    are not deduplicated. The caller selects nonduplicated reference spans,
    evaluated regions, and ambiguity exclusions. No display/minimum-length gate,
    ROI clipping, uncertainty inference, or unsupported-gap bridging is applied.
    Pairwise work is O(P**2 + P*R), intended for small evaluation regions.
    """
    if (
        type(distance_tolerance) not in (int, float)
        or not isfinite(distance_tolerance)
        or distance_tolerance <= 0
        or type(angle_tolerance_degrees) not in (int, float)
        or not isfinite(angle_tolerance_degrees)
        or not 0 <= angle_tolerance_degrees < 90
    ):
        raise PipelineInputError('Require finite distance > 0 and 0 <= angle < 90 degrees')
    directions, lengths = _geometry(predicted, predicted_kinds, 'Predicted')
    reference_directions, reference_lengths = _geometry(reference, reference_kinds, 'Reference')
    results: list[EdgeTypeQuality] = []
    kinds: tuple[Literal[1, 2], ...] = (1, 2)
    for kind in kinds:
        selected = predicted_kinds == kind
        targets = reference_kinds == kind
        results.append(
            _evaluate_type(
                kind,
                predicted[selected],
                directions[selected],
                lengths[selected],
                reference[targets],
                reference_directions[targets],
                reference_lengths[targets],
                distance_tolerance,
                cos(radians(angle_tolerance_degrees)),
            )
        )
    return EdgeQuality(
        results[0], results[1], float(distance_tolerance), float(angle_tolerance_degrees)
    )
