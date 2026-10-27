"""Analytic finite-segment coverage, not segment counts or centre distances."""

from dataclasses import FrozenInstanceError, astuple
from itertools import permutations
from math import cos, radians, sin, sqrt, tan

import numpy as np
from numpy.typing import NDArray
import pytest

from usim_env_pipeline.edge_quality import EdgeTypeQuality, evaluate_edge_quality
from usim_env_pipeline.models import PipelineInputError


def lines(*intervals: tuple[float, float], y: float = 0) -> NDArray[np.float64]:
    return np.array(
        [[[low, y, 0], [high, y, 0]] for low, high in intervals], dtype=np.float64
    ).reshape((-1, 2, 3))


def boundary(
    predicted: NDArray[np.float64], reference: NDArray[np.float64], **tolerances: float
) -> EdgeTypeQuality:
    return evaluate_edge_quality(
        predicted,
        np.ones(len(predicted), dtype=np.int64),
        reference,
        np.ones(len(reference), dtype=np.int64),
        **tolerances,
    ).boundary


def test_perfect_matches_are_separate_typed_frozen_results():
    segments = np.concatenate((lines((0, 4)), lines((0, 3), y=2)))
    kinds = np.array([1, 2], dtype=np.uint8)
    result = evaluate_edge_quality(segments, kinds, segments.copy(), kinds.copy())
    assert result.distance_tolerance == 0.10
    assert result.angle_tolerance_degrees == 5.0
    for metric, kind, length in ((result.boundary, 1, 4), (result.crease, 2, 3)):
        assert metric.kind == kind
        assert metric.raw_predicted_length == metric.predicted_length == length
        assert metric.reference_length == metric.matched_reference_length == length
        assert metric.matched_predicted_length == length
        assert metric.duplicate_prediction_length == metric.false_positive_length == 0
        assert metric.precision == metric.recall == 1
        assert metric.fragment_counts == (1,)
    with pytest.raises(FrozenInstanceError):
        setattr(result, 'distance_tolerance', 2)
    with pytest.raises(FrozenInstanceError):
        setattr(result.boundary, 'precision', 0)


def test_partial_coverage_uses_physical_lengths_on_both_sides():
    metric = boundary(lines((-2, 2), (6, 8)), lines((0, 10)))
    assert metric.predicted_length == 6
    assert metric.reference_length == 10
    assert metric.matched_predicted_length == metric.matched_reference_length == 4
    assert metric.false_positive_length == 2
    assert metric.precision == pytest.approx(2 / 3)
    assert metric.recall == pytest.approx(0.4)
    assert metric.fragment_counts == (2,)


def test_duplicate_success_does_not_improve_precision_or_recall():
    original = boundary(lines((0, 2), (10, 12)), lines((0, 4)))
    doubled = boundary(lines((0, 2), (10, 12), (2, 0)), lines((0, 4)))
    assert doubled.raw_predicted_length == 6
    assert doubled.predicted_length == original.predicted_length == 4
    assert doubled.matched_predicted_length == original.matched_predicted_length == 2
    assert doubled.matched_reference_length == original.matched_reference_length == 2
    assert doubled.precision == original.precision == 0.5
    assert doubled.recall == original.recall == 0.5
    assert doubled.duplicate_prediction_length == 2
    assert doubled.fragment_counts == (2,)


def test_nested_and_partial_duplicates_use_unions_not_pairwise_subtraction():
    predicted = lines((0, 4), (1, 3), (2, 6), (2.5, 3.5))
    for order in permutations(range(4)):
        metric = boundary(predicted[list(order)], lines((0, 2), (3, 5)))
        assert metric.raw_predicted_length == 11
        assert metric.predicted_length == 6
        assert metric.duplicate_prediction_length == 5
        assert metric.matched_predicted_length == metric.matched_reference_length == 4
        assert metric.precision == pytest.approx(2 / 3)
        assert metric.recall == 1
        assert metric.fragment_counts == (2, 3)


def test_multiple_references_do_not_credit_prediction_length_twice():
    metric = boundary(lines((0, 4)), lines((0, 3), (1, 4)))
    assert metric.predicted_length == metric.matched_predicted_length == 4
    assert metric.precision == 1
    # Independent annotations are evaluated separately, as documented.
    assert metric.reference_length == metric.matched_reference_length == 6


@pytest.mark.parametrize('interval', [(10, 11), (10.01, 11), (-1, -0.01)])
def test_collinear_but_longitudinally_disjoint_is_not_a_match(interval):
    metric = boundary(lines(interval), lines((0, 10)))
    assert metric.matched_predicted_length == metric.matched_reference_length == 0
    assert metric.precision == metric.recall == 0
    assert metric.false_positive_length == pytest.approx(abs(interval[1] - interval[0]))


@pytest.mark.parametrize('offset,matched', [(0.05, 4), (0.10, 4), (0.1001, 0)])
def test_parallel_lines_require_actual_distance_and_overlap(offset, matched):
    metric = boundary(lines((2, 6), y=offset), lines((0, 10)))
    assert metric.matched_predicted_length == matched
    assert metric.matched_reference_length == matched
    disjoint = boundary(lines((10.01, 11), y=offset), lines((0, 10)))
    assert disjoint.matched_reference_length == 0


def test_nearby_distinct_parallel_predictions_are_not_geometric_duplicates():
    predicted = np.concatenate((lines((0, 2), y=-0.04), lines((0, 2), y=0.04)))
    metric = boundary(predicted, lines((0, 4)))
    assert metric.raw_predicted_length == metric.predicted_length == 4
    assert metric.duplicate_prediction_length == 0
    assert metric.matched_predicted_length == 4
    assert metric.matched_reference_length == 2
    assert metric.recall == 0.5


@pytest.mark.parametrize('height', [0.0, 0.08])
def test_angled_divergence_clips_distance_across_the_matched_interval(height):
    angle = radians(4)
    predicted = np.array(
        [[[-10, -10 * tan(angle), height], [10, 10 * tan(angle), height]]],
        dtype=np.float64,
    )
    metric = boundary(predicted, lines((-10, 10)))
    radius = sqrt(0.10**2 - height**2)
    assert metric.matched_predicted_length == pytest.approx(2 * radius / sin(angle))
    assert metric.matched_reference_length == pytest.approx(2 * radius / tan(angle))
    assert metric.matched_predicted_length < metric.predicted_length
    assert metric.recall == pytest.approx(radius / (10 * tan(angle)))


def test_matching_end_of_a_line_is_kept_even_when_its_centre_is_too_far():
    angle = radians(4)
    predicted = np.array([[[0, 0, 0], [10, 10 * tan(angle), 0]]], dtype=np.float64)
    metric = boundary(predicted, lines((0, 10)))
    assert metric.matched_predicted_length == pytest.approx(0.10 / sin(angle))
    assert metric.matched_reference_length == pytest.approx(0.10 / tan(angle))


def test_short_angled_prediction_credits_projected_reference_arc_length():
    angle = radians(4)
    predicted = np.array([[[0, 0, 0], [cos(angle), sin(angle), 0]]], dtype=np.float64)
    metric = boundary(predicted, lines((0, 2)))
    assert metric.matched_predicted_length == pytest.approx(1)
    assert metric.matched_reference_length == pytest.approx(cos(angle))
    assert metric.precision == pytest.approx(1)


def test_angle_gate_is_independent_of_distance_gate_and_is_configurable():
    angle = radians(6)
    predicted = np.array([[[0, 0, 0], [0.1, 0.1 * tan(angle), 0]]], dtype=np.float64)
    assert boundary(predicted, lines((0, 2))).matched_reference_length == 0
    assert boundary(
        predicted, lines((0, 2)), angle_tolerance_degrees=6
    ).matched_reference_length == pytest.approx(0.1)
    assert boundary(lines((0, 1), y=0.15), lines((0, 2))).precision == 0
    assert boundary(lines((0, 1), y=0.15), lines((0, 2)), distance_tolerance=0.2).precision == 1


def test_perpendicular_lines_do_not_match_even_with_nearly_ninety_degree_tolerance():
    predicted = np.array([[[0.5, -1, 0], [0.5, 1, 0]]], dtype=np.float64)
    assert (
        boundary(
            predicted, lines((0, 2)), angle_tolerance_degrees=89.99999999999999
        ).matched_reference_length
        == 0
    )


def test_type_mismatch_is_false_positive_and_missed_reference():
    result = evaluate_edge_quality(lines((0, 2)), np.array([1]), lines((0, 2)), np.array([2]))
    assert result.boundary.false_positive_length == 2
    assert result.boundary.precision == 0
    assert result.boundary.recall is None
    assert result.crease.predicted_length == 0
    assert result.crease.precision is None
    assert result.crease.recall == 0
    assert result.crease.fragment_counts == (0,)


@pytest.mark.parametrize('reverse_prediction', [False, True])
@pytest.mark.parametrize('reverse_reference', [False, True])
def test_endpoint_reversal_preserves_distance_clipping(reverse_prediction, reverse_reference):
    angle = radians(4)
    predicted = np.array([[[-3, -3 * tan(angle), 0], [4, 4 * tan(angle), 0]]])
    reference = lines((-2, 5))
    expected = boundary(predicted, reference)
    if reverse_prediction:
        predicted = predicted[:, ::-1]
    if reverse_reference:
        reference = reference[:, ::-1]
    actual = boundary(predicted, reference)
    assert all(a == pytest.approx(b) for a, b in zip(astuple(actual), astuple(expected)))


def test_empty_inputs_and_empty_reference_do_not_fabricate_recall():
    empty = boundary(lines(), lines())
    assert empty.predicted_length == empty.reference_length == 0
    assert empty.precision is None and empty.recall is None
    assert empty.false_positive_length == 0
    assert empty.fragment_counts == ()
    missing = boundary(lines(), lines((0, 2)))
    assert missing.precision is None and missing.recall == 0
    assert missing.fragment_counts == (0,)
    negative = boundary(lines((0, 2), (1, 3)), lines())
    assert negative.raw_predicted_length == 4
    assert negative.predicted_length == negative.false_positive_length == 3
    assert negative.duplicate_prediction_length == 1
    assert negative.precision == 0 and negative.recall is None


def test_short_fragments_match_long_reference_without_filling_the_gap():
    metric = boundary(lines((0, 0.8), (0.9, 1.8)), lines((0, 1.8)))
    assert metric.predicted_length == pytest.approx(1.7)
    assert metric.matched_predicted_length == pytest.approx(1.7)
    assert metric.matched_reference_length == pytest.approx(1.7)
    assert metric.precision == 1
    assert metric.recall == pytest.approx(1.7 / 1.8)
    assert metric.fragment_counts == (2,)


def test_adjacent_short_fragments_cover_long_reference_without_becoming_duplicates():
    metric = boundary(lines((0, 0.9), (0.9, 1.8)), lines((0, 1.8)))
    assert metric.matched_reference_length == pytest.approx(1.8)
    assert metric.duplicate_prediction_length == 0
    assert metric.precision == metric.recall == 1
    assert metric.fragment_counts == (2,)


def test_rigid_transform_preserves_lengths_clipping_duplicates_and_inputs():
    angle = radians(4)
    predicted = np.concatenate(
        (
            lines((0, 4), (1, 3)),
            np.array([[[-3, -3 * tan(angle), 0.08], [4, 4 * tan(angle), 0.08]]]),
            lines((0, 2), y=0.10),
        )
    )
    reference = lines((-2, 5))
    original = predicted.copy()
    rotation = np.array([[0.36, 0.48, -0.8], [-0.8, 0.6, 0], [0.48, 0.64, 0.6]])
    shift = np.array([143, -89, 21])
    predicted.setflags(write=False)
    expected = boundary(predicted, reference)
    actual = boundary(predicted @ rotation + shift, reference @ rotation + shift)
    assert all(a == pytest.approx(b) for a, b in zip(astuple(actual), astuple(expected)))
    np.testing.assert_array_equal(predicted, original)
    assert expected.duplicate_prediction_length == 2


@pytest.mark.parametrize(
    'invalid',
    [
        [],
        np.zeros((2, 3), dtype=np.float64),
        np.zeros((1, 3, 2), dtype=np.float64),
        lines((0, 2)).astype(np.float32),
        lines((0, 2)).astype(np.int64),
        np.full((1, 2, 3), np.nan),
        np.full((1, 2, 3), np.inf),
        lines((0, 0)),
        np.array([[[-1e308, 0, 0], [1e308, 0, 0]]]),
    ],
)
@pytest.mark.parametrize('side', ['predicted', 'reference'])
def test_invalid_segment_arrays_are_rejected(invalid, side):
    predicted, reference = lines((0, 2)), lines((0, 2))
    if side == 'predicted':
        predicted = invalid
    else:
        reference = invalid
    with pytest.raises(PipelineInputError):
        evaluate_edge_quality(predicted, np.array([1]), reference, np.array([1]))


@pytest.mark.parametrize(
    'invalid',
    [
        [1],
        np.array([1.0]),
        np.array([True]),
        np.array([[1]]),
        np.array([1, 2]),
        np.array([0]),
        np.array([3]),
        np.array([-1]),
    ],
)
@pytest.mark.parametrize('side', ['predicted', 'reference'])
def test_invalid_kind_arrays_are_rejected(invalid, side):
    predicted_kinds = invalid if side == 'predicted' else np.array([1])
    reference_kinds = invalid if side == 'reference' else np.array([1])
    with pytest.raises(PipelineInputError):
        evaluate_edge_quality(lines((0, 2)), predicted_kinds, lines((0, 2)), reference_kinds)


@pytest.mark.parametrize(
    'parameters',
    [
        {'distance_tolerance': 0},
        {'distance_tolerance': -0.1},
        {'distance_tolerance': float('nan')},
        {'distance_tolerance': float('inf')},
        {'distance_tolerance': True},
        {'angle_tolerance_degrees': -1},
        {'angle_tolerance_degrees': 90},
        {'angle_tolerance_degrees': float('nan')},
        {'angle_tolerance_degrees': float('inf')},
        {'angle_tolerance_degrees': False},
    ],
)
def test_invalid_tolerances_are_rejected(parameters):
    with pytest.raises(PipelineInputError):
        boundary(lines((0, 2)), lines((0, 2)), **parameters)
