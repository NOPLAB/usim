"""Curve proximity is measured along length against unchanged source observations."""

from dataclasses import FrozenInstanceError

import numpy as np
from numpy.typing import NDArray
import pytest

from usim_env_pipeline.curve_support import audit_curve_support, resample_curve
from usim_env_pipeline.models import PipelineInputError


def line(x: list[float] | NDArray[np.float64]) -> NDArray[np.float64]:
    return np.column_stack((x, np.zeros(len(x)), np.zeros(len(x))))


def test_uniform_straight_samples_preserve_tail_and_ignore_repeated_points() -> None:
    curve = line([0, 0, 0.007, 0.007, 0.081, 0.103, 0.103])
    samples = resample_curve(curve)
    np.testing.assert_allclose(samples.xyz[:, 0], np.linspace(0, 0.103, 6))
    assert np.diff(samples.arc_lengths).max() <= 0.025
    assert samples.length_weights.sum() == pytest.approx(0.103)
    np.testing.assert_allclose(samples.xyz[-1], curve[-1])
    assert not samples.closed
    with pytest.raises(FrozenInstanceError):
        setattr(samples, 'length', 2)


def test_closed_curve_keeps_closing_segment_without_duplicate_sample() -> None:
    curve = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0]],
        dtype=np.float64,
    )
    samples = resample_curve(curve, spacing=0.25)
    assert samples.closed
    assert samples.length == 4
    assert len(samples.xyz) == 16
    np.testing.assert_allclose(samples.xyz[-1], [0, 0.25, 0])
    np.testing.assert_allclose(samples.length_weights, 0.25)


def test_circle_has_dense_original_support_and_source_id_mapping() -> None:
    angle = np.linspace(0, 2 * np.pi, 257)
    curve = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(len(angle))))
    curve[-1] = curve[0]
    samples = resample_curve(curve)
    source = samples.xyz + [0, 0, 0.1]
    ids = np.arange(len(source), dtype=np.int64) * 17 + 101
    audit = audit_curve_support([curve], source, ids, support_distance=0.11)[0]
    assert audit.samples.closed
    assert audit.samples.length == pytest.approx(2 * np.pi, rel=0.0001)
    assert audit.supported_fraction == pytest.approx(1)
    assert audit.p95_distance == pytest.approx(0.1)
    assert audit.max_unsupported_length == 0
    np.testing.assert_allclose(audit.nearest_distances, 0.1)
    np.testing.assert_array_equal(audit.nearest_source_ids, ids)
    assert np.diff(audit.samples.arc_lengths).max() <= 0.025


def test_length_weights_do_not_count_open_endpoints_as_full_cells() -> None:
    curve = line([0, 0.0001, 0.0002, 0.06])
    audit = audit_curve_support(
        [curve], line([0]), np.array([91], dtype=np.int64), support_distance=0.001
    )[0]
    assert audit.supported_fraction == pytest.approx(1 / 6)
    assert audit.p95_distance == pytest.approx(0.06)
    assert audit.max_unsupported_length == pytest.approx(0.05)
    np.testing.assert_allclose(audit.samples.length_weights, [0.01, 0.02, 0.02, 0.01])


def test_source_gap_is_a_length_not_sample_count_and_ignores_input_sampling_density() -> None:
    source = line([0, 0.1, 0.2, 0.8, 0.9, 1])
    curves = [line([0, 1]), line([0, 0.001, 0.002, 0.003, 0.75, 1])]
    audits = audit_curve_support(
        curves, source, np.arange(6, dtype=np.int64), spacing=0.1, support_distance=1e-10
    )
    for audit in audits:
        assert audit.supported_fraction == pytest.approx(0.5)
        assert audit.max_unsupported_length == pytest.approx(0.5)
        assert audit.p95_distance == pytest.approx(0.3)
    np.testing.assert_allclose(audits[0].nearest_distances, audits[1].nearest_distances)


def test_closed_unsupported_run_wraps_across_seam() -> None:
    curve = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=float)
    audit = audit_curve_support(
        [curve],
        curve[1:4],
        np.array([9, 17, 23], dtype=np.int64),
        spacing=0.25,
        support_distance=0,
    )[0]
    assert audit.max_unsupported_length == pytest.approx(1.75)
    assert audit.supported_fraction == pytest.approx(3 / 16)


def test_fully_unsupported_curve_reports_whole_length() -> None:
    audit = audit_curve_support([line([0, 1])], line([10]), np.array([42], dtype=np.int64))[0]
    assert audit.supported_fraction == 0
    assert audit.max_unsupported_length == 1
    assert np.all(audit.nearest_source_ids == 42)


def test_audit_is_deterministic_preserves_inputs_and_maps_ties_to_first_original_row() -> None:
    curve = line([0, 0.1])
    source = np.array([[0, 1, 0], [0, -1, 0], [0, 1, 0]], dtype=np.float64)
    ids = np.array([93, 4, 7], dtype=np.int64)
    before = [array.copy() for array in (curve, source, ids)]
    for array in (curve, source, ids):
        array.setflags(write=False)
    first = audit_curve_support([curve], source, ids)[0]
    second = audit_curve_support([curve], source, ids)[0]
    np.testing.assert_array_equal(first.nearest_source_ids, 93)
    np.testing.assert_array_equal(first.nearest_distances, second.nearest_distances)
    for array, original in zip((curve, source, ids), before, strict=True):
        np.testing.assert_array_equal(array, original)
    assert not np.shares_memory(first.samples.xyz, curve)
    assert not np.shares_memory(first.nearest_source_ids, ids)


@pytest.mark.parametrize(
    'curve',
    [
        np.empty((0, 3)),
        np.zeros((1, 3)),
        np.zeros((3, 3)),
        np.zeros((3, 2)),
        np.array([[0, 0, 0], [np.nan, 0, 0]]),
        np.zeros((2, 3), dtype=np.float32),
    ],
)
def test_invalid_curves_fail_at_boundary(curve: NDArray[np.float64]) -> None:
    with pytest.raises(PipelineInputError):
        resample_curve(curve)


@pytest.mark.parametrize('spacing', [0, -1, np.nan, np.inf])
def test_invalid_spacing_fails_even_without_curves(spacing: float) -> None:
    with pytest.raises(PipelineInputError):
        audit_curve_support([], line([0]), np.array([1], dtype=np.int64), spacing=spacing)


def test_empty_sequence_is_typed_empty_but_missing_source_or_ids_fail() -> None:
    assert audit_curve_support([], line([0]), np.array([5], dtype=np.int64)) == ()
    with pytest.raises(PipelineInputError):
        audit_curve_support([], np.empty((0, 3)), np.empty(0, dtype=np.int64))
    with pytest.raises(PipelineInputError):
        audit_curve_support([line([0, 1])], line([0]), np.empty(0, dtype=np.int64))
    with pytest.raises(PipelineInputError):
        audit_curve_support([line([0, 1])], line([0]), np.array([1.0]))
    with pytest.raises(PipelineInputError):
        audit_curve_support([line([0, 1])], line([np.inf]), np.array([1], dtype=np.int64))


@pytest.mark.parametrize('distance', [-1, np.nan, np.inf])
def test_invalid_support_distance_fails(distance: float) -> None:
    with pytest.raises(PipelineInputError):
        audit_curve_support(
            [], line([0]), np.array([1], dtype=np.int64), support_distance=distance
        )
