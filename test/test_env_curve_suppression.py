"""Geometric copies are removed without moving curves or merging distinct paths."""

import numpy as np
from numpy.typing import NDArray
import pytest

from usim_env_pipeline.curve_suppression import suppress_duplicate_curves
from usim_env_pipeline.models import PipelineInputError


def line(start: float = 0, stop: float = 1, *, y: float = 0, z: float = 0) -> NDArray[np.float64]:
    return np.array([[start, y, z], [stop, y, z]], dtype=np.float64)


def circle(*, z: float = 0) -> NDArray[np.float64]:
    angle = np.linspace(0, 2 * np.pi, 129)
    points = np.column_stack((np.cos(angle), np.sin(angle), np.full(len(angle), z)))
    points[-1] = points[0]
    return points


@pytest.mark.parametrize('reverse', [False, True])
def test_higher_rank_keeps_exact_original_row_when_copy_is_reversed(reverse: bool) -> None:
    # Given an unchanged line and a differently sampled copy.
    original = line(0, 2)
    duplicate = line(0, 2)[::-1] if reverse else line(0, 2)
    before = [original.copy(), duplicate.copy()]
    # When the second row has stronger evidence.
    result = suppress_duplicate_curves([original, duplicate], np.array([0.4, 0.9]))
    # Then only that original row survives, with a direct trace for the loser.
    np.testing.assert_array_equal(result.kept_indices, [1])
    np.testing.assert_array_equal(result.representative_indices, [1, 1])
    for current, previous in zip((original, duplicate), before, strict=True):
        np.testing.assert_array_equal(current, previous)


def test_contained_fragment_is_removed_when_longer_representative_has_equal_rank() -> None:
    # Given a supported complete span and an internal duplicate.
    curves = [line(0.2, 0.8), line(0, 1)]
    # When confidence is tied.
    result = suppress_duplicate_curves(curves, np.ones(2))
    # Then length breaks the tie without erasing coverage.
    np.testing.assert_array_equal(result.kept_indices, [1])
    assert result.covered_fractions[0] == pytest.approx(1)
    assert result.max_uncovered_lengths[0] == 0


def test_unique_extension_is_preserved_even_when_ninety_five_percent_overlaps() -> None:
    # Given a weaker span with a real additional 0.2 m tail.
    curves = [line(0, 10), line(0, 10.2)]
    # When its high overlap alone would have allowed suppression.
    result = suppress_duplicate_curves(curves, np.array([1.0, 0.5]))
    # Then the tail-length guard keeps both original spans.
    np.testing.assert_array_equal(result.kept_indices, [0, 1])


@pytest.mark.parametrize('offset', [[0, 0.04, 0], [0, 0, 0.04]])
def test_parallel_edge_is_kept_when_separation_exceeds_distance(offset: list[float]) -> None:
    # Given nearby but resolved lateral or depth-separated edges.
    curves = [line(), line() + offset]
    # When compared in actual 3D coordinates.
    result = suppress_duplicate_curves(curves, np.array([1.0, 0.5]))
    # Then neither is mistaken for a screen-space duplicate.
    np.testing.assert_array_equal(result.kept_indices, [0, 1])


def test_crossing_is_kept_when_short_curve_is_near_but_not_tangent_aligned() -> None:
    # Given a short perpendicular edge entirely within the distance tube.
    crossing = np.array([[0.5, -0.005, 0], [0.5, 0.005, 0]])
    # When tangent direction is part of the overlap contract.
    result = suppress_duplicate_curves([line(), crossing], np.array([1.0, 0.5]))
    # Then proximity at an intersection does not erase it.
    np.testing.assert_array_equal(result.kept_indices, [0, 1])


def test_curved_path_is_not_a_duplicate_of_its_endpoint_chord() -> None:
    # Given two curves with equal endpoints and different interior geometry.
    arch = np.array([[0, 0, 0], [0.5, 1, 0], [1, 0, 0]], dtype=np.float64)
    # When the whole polyline, rather than its endpoints, is compared.
    result = suppress_duplicate_curves([line(), arch], np.array([1.0, 0.5]))
    # Then the curved path is retained.
    np.testing.assert_array_equal(result.kept_indices, [0, 1])


def test_closed_curve_copy_survives_seam_rotation_and_reverse_direction() -> None:
    # Given one circle, a cyclically shifted reversed copy and a separate depth ring.
    original = circle()
    shifted = np.roll(original[:-1][::-1], 31, axis=0)
    shifted = np.vstack((shifted, shifted[0]))
    # When the seam has no geometric significance.
    result = suppress_duplicate_curves(
        [original, shifted, circle(z=0.06)], np.array([1.0, 0.5, 0.4])
    )
    # Then only the true copy is suppressed.
    np.testing.assert_array_equal(result.kept_indices, [0, 2])
    np.testing.assert_array_equal(result.representative_indices, [0, 0, 2])


def test_nontransitive_chain_does_not_merge_edges_beyond_direct_tolerance() -> None:
    # Given A close to B and B close to C, but A separated from C.
    curves = [line(y=0), line(y=0.02), line(y=0.04)]
    # When suppression uses retained representatives, not connected components.
    result = suppress_duplicate_curves(curves, np.array([3.0, 2.0, 1.0]))
    # Then C remains, and every loser points directly to a final survivor.
    np.testing.assert_array_equal(result.kept_indices, [0, 2])
    np.testing.assert_array_equal(result.representative_indices, [0, 0, 2])


def test_empty_input_and_fixed_ties_produce_readonly_deterministic_indices() -> None:
    # Given identical read-only inputs with fixed equal ranking.
    curves = [line(), line()]
    for curve in curves:
        curve.setflags(write=False)
    # When run independently with the same inputs.
    first = suppress_duplicate_curves(curves, np.ones(2))
    second = suppress_duplicate_curves(curves, np.ones(2))
    # Then ties keep the first row and outputs cannot be accidentally edited.
    np.testing.assert_array_equal(first.kept_indices, second.kept_indices)
    np.testing.assert_array_equal(first.kept_indices, [0])
    assert not first.representative_indices.flags.writeable
    empty = suppress_duplicate_curves([], np.empty(0))
    assert empty.kept_indices.shape == empty.representative_indices.shape == (0,)
    assert empty.kept_indices.dtype == np.int64


@pytest.mark.parametrize(
    'ranking',
    [np.empty(0), np.array([np.nan]), np.array([np.inf]), np.ones(1, dtype=np.float32)],
)
def test_invalid_ranking_is_rejected_at_boundary(ranking: NDArray[np.float64]) -> None:
    with pytest.raises(PipelineInputError):
        suppress_duplicate_curves([line()], ranking)


@pytest.mark.parametrize('distance', [0, -1, np.nan, np.inf])
def test_invalid_distance_is_rejected(distance: float) -> None:
    with pytest.raises(PipelineInputError):
        suppress_duplicate_curves([], np.empty(0), distance=distance)


@pytest.mark.parametrize('angle', [0, 91, np.nan])
def test_invalid_direction_tolerance_is_rejected(angle: float) -> None:
    with pytest.raises(PipelineInputError):
        suppress_duplicate_curves([], np.empty(0), angle_degrees=angle)


@pytest.mark.parametrize('fraction', [0, 1.1, np.nan])
def test_invalid_overlap_fraction_is_rejected(fraction: float) -> None:
    with pytest.raises(PipelineInputError):
        suppress_duplicate_curves([], np.empty(0), min_overlap=fraction)


@pytest.mark.parametrize('spacing', [0, -1, np.inf])
def test_invalid_sampling_spacing_is_rejected(spacing: float) -> None:
    with pytest.raises(PipelineInputError):
        suppress_duplicate_curves([], np.empty(0), spacing=spacing)


def test_degenerate_curve_is_rejected_by_shared_resampling_contract() -> None:
    with pytest.raises(PipelineInputError):
        suppress_duplicate_curves([np.zeros((2, 3))], np.ones(1))
