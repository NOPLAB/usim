"""Partial duplicate removal preserves unique source paths and auditable coverage."""

import numpy as np
from numpy.typing import NDArray
import pytest

from usim_env_pipeline.curve_overlap import trim_duplicate_curve_overlaps
from usim_env_pipeline.models import PipelineInputError


def line(stop: float = 1, *, start: float = 0, y: float = 0) -> NDArray[np.float64]:
    return np.array([[start, y, 0], [stop, y, 0]], dtype=np.float64)


def length(points: NDArray[np.float64]) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def test_drifting_duplicate_is_trimmed_only_where_it_is_close() -> None:
    # Given a 6 m line and a near-copy drifting from 0 to 6 cm away.
    original = line(6)
    drift = np.array([[0, 0, 0], [6, 0.06, 0]], dtype=np.float64)
    # When the duplicate distance is 3 cm.
    result = trim_duplicate_curve_overlaps([original, drift], np.array([1.0, 0.5]))
    # Then about 3 m is removed, and the original drifting extension survives.
    assert sum(length(f.xyz) for f in result.removed) == pytest.approx(3, abs=0.02)
    tail = [f for f in result.retained if f.source_index == 1]
    np.testing.assert_array_equal(tail[-1].xyz[-1], drift[-1])
    np.testing.assert_allclose(tail[0].xyz[:, 1], tail[0].xyz[:, 0] * 0.01, atol=1e-15)
    assert tail[0].xyz[0, 0] == pytest.approx(3, abs=0.02)


def test_unique_tail_and_total_original_length_are_preserved_in_partition() -> None:
    # Given a weaker curve that extends beyond the stronger curve.
    curves = [line(10), line(10.2)]
    # When only their overlap is removed.
    result = trim_duplicate_curve_overlaps(curves, np.array([1.0, 0.5]))
    # Then the outer endpoint remains, and no source path is silently discarded.
    tail = [f for f in result.retained if f.source_index == 1]
    assert tail[-1].xyz[-1, 0] == 10.2
    assert tail[0].xyz[0, 0] <= 10.04
    partition = sum(length(f.xyz) for f in (*result.retained, *result.removed))
    assert partition == pytest.approx(20.2, abs=1e-12)


def test_collinear_endpoint_contact_does_not_shorten_either_curve() -> None:
    # Given two spans that touch but do not overlap.
    curves = [line(), line(2, start=1)]
    # When a capsule alone would include a short tangential extension.
    result = trim_duplicate_curve_overlaps(curves, np.array([1.0, 0.5]))
    # Then the minimum shared-length guard preserves the contact exactly.
    assert not result.removed
    for fragment, original in zip(result.retained, curves, strict=True):
        np.testing.assert_array_equal(fragment.xyz, original)


def test_tangent_line_arc_contact_is_not_a_duplicate() -> None:
    # Given a line joining a quarter-circle tangentially.
    straight = np.array([[1, -1, 0], [1, 0, 0]], dtype=np.float64)
    angle = np.linspace(0, np.pi / 2, 65)
    arc = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(len(angle))))
    # When the endpoint contact is shorter than a supported overlap run.
    result = trim_duplicate_curve_overlaps([straight, arc], np.array([1.0, 0.5]))
    # Then the arc is neither shortened nor replaced by a chord.
    assert not result.removed
    np.testing.assert_array_equal(result.retained[1].xyz, arc)


def test_partial_trim_keeps_original_nonuniform_corner_vertices() -> None:
    # Given a polyline whose original corners do not coincide with the sampling grid.
    path = np.array([[0, 0, 0], [1.003, 0, 0], [1.003, 1.337, 0], [2.113, 1.337, 0]])
    # When its first leg is covered by another curve.
    result = trim_duplicate_curve_overlaps([line(1.003), path], np.array([1.0, 0.5]))
    # Then every surviving corner is exact, not a shortcut between uniform samples.
    retained = np.concatenate([f.xyz for f in result.retained if f.source_index == 1])
    for vertex in path[1:]:
        assert np.any(np.all(retained == vertex, axis=1))


def test_closed_reversed_copy_and_separate_depth_ring_are_distinguished() -> None:
    # Given a closed circle, a reversed copy and another depth layer.
    angle = np.linspace(0, 2 * np.pi, 129)
    circle = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(len(angle))))
    circle[-1] = circle[0]
    # When direction is unoriented and distance is genuinely three-dimensional.
    result = trim_duplicate_curve_overlaps(
        [circle, circle[::-1], circle + [0, 0, 0.06]], np.array([1.0, 0.5, 0.4])
    )
    # Then only the reversed copy is removed, including its seam.
    assert [f.source_index for f in result.retained] == [0, 2]
    assert {f.source_index for f in result.removed} == {1}
    assert sum(length(f.xyz) for f in result.removed) == pytest.approx(length(circle))


def test_perpendicular_crossing_and_separate_parallel_path_are_retained() -> None:
    # Given a sub-centimetre crossing and a resolved nearby parallel edge.
    crossing = np.array([[0.5, -0.005, 0], [0.5, 0.005, 0]])
    # When their closest points alone would be misleading.
    result = trim_duplicate_curve_overlaps(
        [line(), crossing, line(y=0.04)], np.array([1.0, 0.5, 0.4])
    )
    # Then both distinct paths remain unmodified.
    assert not result.removed
    assert [f.source_index for f in result.retained] == [0, 1, 2]


def test_removed_cells_point_to_final_references_without_transitive_collapse() -> None:
    # Given an A-B-C chain whose extremes exceed the distance tolerance.
    curves = [line(y=0), line(y=0.02), line(y=0.04)]
    # When only surviving geometry becomes a new reference.
    result = trim_duplicate_curve_overlaps(curves, np.array([3.0, 2.0, 1.0]))
    # Then C survives, and every removed B cell has direct retained-A evidence.
    assert [f.source_index for f in result.retained] == [0, 2]
    np.testing.assert_array_equal(result.source_indices, 1)
    np.testing.assert_array_equal(result.representative_fragments, 0)
    np.testing.assert_allclose(result.covered_segments[:, :, 1], 0.02)
    np.testing.assert_allclose(result.reference_segments[:, :, 1], 0)


def test_stronger_later_source_uses_fragment_id_not_raw_row_as_reference() -> None:
    # Given a stronger source row that is not the first input row.
    curves = [line(y=0.02), line()]
    # When ranking chooses that row as the first retained fragment.
    result = trim_duplicate_curve_overlaps(curves, np.array([0.3, 0.9]))
    # Then source-row IDs and retained-fragment IDs remain distinct.
    assert [f.source_index for f in result.retained] == [1]
    np.testing.assert_array_equal(result.source_indices, 0)
    np.testing.assert_array_equal(result.representative_fragments, 0)


def test_inputs_are_immutable_and_empty_result_has_complete_array_shapes() -> None:
    # Given read-only inputs and equal ranking.
    curves = [line(), line()[::-1]]
    for curve in curves:
        curve.setflags(write=False)
    # When trimming chooses a deterministic original row.
    result = trim_duplicate_curve_overlaps(curves, np.ones(2))
    # Then original arrays stay equal, and outputs are independent and read-only.
    np.testing.assert_array_equal(result.retained[0].xyz, curves[0])
    assert not np.shares_memory(result.retained[0].xyz, curves[0])
    assert not result.covered_segments.flags.writeable
    empty = trim_duplicate_curve_overlaps([], np.empty(0))
    assert empty.retained == empty.removed == ()
    assert empty.covered_segments.shape == empty.reference_segments.shape == (0, 2, 3)


@pytest.mark.parametrize('distance', [0, -1, np.nan])
def test_invalid_distance_is_rejected(distance: float) -> None:
    with pytest.raises(PipelineInputError):
        trim_duplicate_curve_overlaps([], np.empty(0), distance=distance)


@pytest.mark.parametrize('angle', [0, 91, np.inf])
def test_invalid_angle_is_rejected(angle: float) -> None:
    with pytest.raises(PipelineInputError):
        trim_duplicate_curve_overlaps([], np.empty(0), angle_degrees=angle)


@pytest.mark.parametrize('spacing', [0, -1, np.nan])
def test_invalid_spacing_is_rejected(spacing: float) -> None:
    with pytest.raises(PipelineInputError):
        trim_duplicate_curve_overlaps([], np.empty(0), spacing=spacing)


def test_bad_rank_and_degenerate_path_are_rejected() -> None:
    with pytest.raises(PipelineInputError):
        trim_duplicate_curve_overlaps([line()], np.empty(0))
    with pytest.raises(PipelineInputError):
        trim_duplicate_curve_overlaps([np.zeros((2, 3))], np.ones(1))
