"""Exact packed live counts against a tuple-key oracle and the fixed-grid analysis."""

from collections import Counter
from dataclasses import FrozenInstanceError

import numpy as np
from numpy.typing import NDArray
import pytest

from usim_env_pipeline.live_voxel_counts import LiveVoxelCounter, LiveVoxelParameters
from usim_env_pipeline.models import PipelineInputError
from usim_env_pipeline.voxel_thresholds import analyze_thresholds


def tuple_oracle(
    xyz: NDArray[np.float64], keep: NDArray[np.bool_], size: float
) -> NDArray[np.int64]:
    cells = [tuple(int(value) for value in np.floor(row * (1.0 / size))) for row in xyz[keep]]
    populations = Counter(cells)
    result = np.zeros(len(xyz), dtype=np.int64)
    result[keep] = [populations[cell] for cell in cells]
    return result


def as_records(xyz: NDArray[np.float64]) -> NDArray[np.void]:
    records = np.zeros(
        len(xyz), dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('intensity', '<f4')]
    )
    for axis, name in enumerate(('x', 'y', 'z')):
        records[name] = xyz[:, axis]
    return records


@pytest.mark.parametrize('size', [0.001, 0.1, 1.0, 10.0])
@pytest.mark.parametrize('structured', [False, True])
def test_counts_match_tuple_oracle_and_fixed_grid(size: float, structured: bool):
    random = np.random.default_rng(137)
    xyz = random.uniform([-225, -225, -30], [225, 225, 30], (300, 3))
    xyz[10:50] = xyz[0]
    xyz[70:90] = xyz[1]
    xyz[90:100] = -0.0001
    keep = random.random(len(xyz)) > 0.2
    source = as_records(xyz) if structured else xyz
    actual = LiveVoxelCounter(source, keep).count(size)
    assert actual.dtype == np.int64
    np.testing.assert_array_equal(actual, tuple_oracle(xyz, keep, size))
    if size == 0.1:
        np.testing.assert_array_equal(
            actual, analyze_thresholds(as_records(xyz), keep).counts_per_record
        )


@pytest.mark.parametrize('size', [0.001, 0.1, 1.0, 10.0])
@pytest.mark.parametrize('multiple', [-7, -3, -1, 0, 1, 3, 7])
def test_float64_nextafter_boundaries(size: float, multiple: int):
    boundary = size * multiple
    values = [
        np.nextafter(boundary, -np.inf),
        boundary,
        np.nextafter(boundary, np.inf),
    ]
    xyz = np.zeros((12, 3), dtype=np.float64)
    xyz[:, 0] = np.repeat(values, [3, 4, 5])
    xyz[:, 1] = -xyz[:, 0]
    keep = np.ones(len(xyz), dtype=np.bool_)
    np.testing.assert_array_equal(
        LiveVoxelCounter(xyz, keep).count(size), tuple_oracle(xyz, keep, size)
    )


def test_original_order_sor_population_excludes_geometry():
    xyz = np.zeros((8, 3), dtype=np.float64)
    xyz[:, 0] = [1, 0, 2, 0, 1, 0, 0, 1]
    sor_keep = np.array([True, True, False, True, False, True, True, True])
    geometry_keep = np.array([False, True, False, False, False, False, False, False])
    counter = LiveVoxelCounter(as_records(xyz), sor_keep)
    counts = counter.count(0.1)
    np.testing.assert_array_equal(counts, [2, 4, 0, 4, 0, 4, 4, 2])
    assert not np.array_equal(counter.select(counts, 0), geometry_keep)
    np.testing.assert_array_equal(
        counter.select(counts, 2), [False, True, False, True, False, True, True, False]
    )


def test_inclusive_thresholds_zero_through_maximum_without_regrouping(monkeypatch):
    populations = np.array([1, 5, 100, 101])
    xyz = np.zeros((int(populations.sum()) + 1, 3), dtype=np.float64)
    xyz[:-1, 0] = np.repeat(np.arange(4), populations)
    keep = np.ones(len(xyz), dtype=np.bool_)
    keep[-1] = False
    counter = LiveVoxelCounter(xyz, keep)
    unique = np.unique
    calls = []

    def tracked_unique(array, **kwargs):
        assert array.ndim == 1 and array.dtype == np.int64
        assert 'axis' not in kwargs
        calls.append(kwargs)
        return unique(array, **kwargs)

    monkeypatch.setattr(np, 'unique', tracked_unique)
    counts = counter.count(0.1)
    for threshold in (0, 1, 5, 99, 100):
        selected = counter.select(counts, threshold)
        np.testing.assert_array_equal(
            selected, np.r_[np.repeat(populations > threshold, populations), False]
        )
    assert calls == [{'return_inverse': True, 'return_counts': True}]


@pytest.mark.parametrize('rows,keep', [(0, False), (5, False)])
def test_empty_and_no_survivors(rows: int, keep: bool):
    counter = LiveVoxelCounter(
        np.zeros((rows, 3), dtype=np.float64), np.full(rows, keep, dtype=np.bool_)
    )
    np.testing.assert_array_equal(counter.count(0.001), np.zeros(rows, dtype=np.int64))
    assert not counter.select(counter.count(10.0), 0).any()


@pytest.mark.parametrize(
    'xyz',
    [
        [[2, 3, 4], [2, 3, 4], [2, 3, 4]],
        [[-2, 0, 4], [-2, 0, 4], [3, 0, 4]],
        [[0, -225, -30], [0, 225, 30], [0, -225, -30]],
        [[-225, -225, -30], [225, 225, 30], [-225, -225, -30]],
        [[0, 0, 0], [2.0**62, 0, 0], [2.0**62, 0, 0]],
        [[-(2.0**63), 0, 0], [-1024, 0, 0], [-1024, 0, 0]],
    ],
)
def test_zero_width_axes_source_span_and_exact_63_bit_capacity(xyz):
    points = np.array(xyz, dtype=np.float64)
    keep = np.ones(len(points), dtype=np.bool_)
    size = 1.0 if np.abs(points).max() > 1e10 else 0.001
    np.testing.assert_array_equal(
        LiveVoxelCounter(points, keep).count(size), tuple_oracle(points, keep, size)
    )


@pytest.mark.parametrize(
    'xyz',
    [
        [[-(2.0**63), 0, 0], [0, 0, 0]],
        [[0, 0, 0], [2.0**32, 2.0**31, 0]],
        [[0, 0, 0], [1e12, 1e12, 1e12]],
        [[2.0**63, 0, 0]],
        [[np.nextafter(-(2.0**63), -np.inf), 0, 0]],
        [[np.finfo(np.float64).max, 0, 0]],
    ],
)
def test_packing_and_coordinate_overflow_rejected(xyz):
    points = np.array(xyz, dtype=np.float64)
    counter = LiveVoxelCounter(points, np.ones(len(points), dtype=np.bool_))
    with pytest.raises(PipelineInputError):
        counter.count(0.001 if points.max() == np.finfo(np.float64).max else 1.0)


@pytest.mark.parametrize('structured', [False, True])
def test_constructor_owns_immutable_snapshot_and_results_are_independent(structured: bool):
    xyz = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]], dtype=np.float64)
    source = as_records(xyz) if structured else xyz.copy()
    keep = np.array([True, True, False])
    original_source, original_keep = source.copy(), keep.copy()
    counter = LiveVoxelCounter(source, keep)
    np.testing.assert_array_equal(source, original_source)
    np.testing.assert_array_equal(keep, original_keep)
    if structured:
        source['x'] = 100
    else:
        source[:] = 100
    keep[:] = False
    first = counter.count(0.1)
    first[:] = 99
    np.testing.assert_array_equal(counter.count(0.1), [2, 2, 0])
    assert counter._xyz.dtype == np.float64
    assert not counter._xyz.flags.writeable
    assert not counter._selected_ids.flags.writeable


@pytest.mark.parametrize('size', [0.001, 1.0, 10.0])
@pytest.mark.parametrize('threshold', [0, 100])
def test_valid_frozen_parameters(size: float, threshold: int):
    parameters = LiveVoxelParameters(size, threshold)
    assert parameters.cell_size == size and parameters.threshold == threshold
    with pytest.raises(FrozenInstanceError):
        setattr(parameters, 'threshold', 1)


@pytest.mark.parametrize(
    'size',
    [0, -1, np.nextafter(0.001, 0), np.nextafter(10.0, np.inf), np.nan, np.inf, True],
)
def test_invalid_cell_size_rejected_by_parameters_and_counter(size):
    with pytest.raises(PipelineInputError):
        LiveVoxelParameters(size, 0)
    counter = LiveVoxelCounter(np.zeros((0, 3)), np.zeros(0, dtype=np.bool_))
    with pytest.raises(PipelineInputError):
        counter.count(size)


@pytest.mark.parametrize('threshold', [-1, 101, 0.5, np.nan, np.inf, True])
def test_invalid_threshold_rejected_by_parameters_and_selection(threshold):
    with pytest.raises(PipelineInputError):
        LiveVoxelParameters(0.1, threshold)
    counter = LiveVoxelCounter(np.zeros((0, 3)), np.zeros(0, dtype=np.bool_))
    with pytest.raises(PipelineInputError):
        counter.select(np.zeros(0, dtype=np.int64), threshold)


@pytest.mark.parametrize(
    'source',
    [
        np.zeros(3),
        np.zeros((2, 2)),
        np.array([[np.nan, 0, 0]]),
        np.array([[np.inf, 0, 0]]),
        np.zeros(2, dtype=[('x', 'f8'), ('y', 'f8')]),
        np.zeros(2, dtype=[('x', 'f8', 2), ('y', 'f8'), ('z', 'f8')]),
        np.zeros((2, 3), dtype=np.complex128),
        np.zeros((2, 3), dtype='U1'),
    ],
)
def test_invalid_source_even_if_sor_rejects_it(source):
    with pytest.raises(PipelineInputError):
        LiveVoxelCounter(source, np.zeros(len(source), dtype=np.bool_))


@pytest.mark.parametrize('keep', [np.array([1, 0]), np.array([True]), np.ones((2, 1), bool)])
def test_invalid_sor_mask(keep):
    with pytest.raises(PipelineInputError):
        LiveVoxelCounter(np.zeros((2, 3)), keep)


@pytest.mark.parametrize('counts', [np.zeros(2, dtype=np.int32), np.zeros(1, dtype=np.int64)])
def test_selection_rejects_malformed_cached_counts(counts):
    counter = LiveVoxelCounter(np.zeros((2, 3)), np.ones(2, dtype=np.bool_))
    with pytest.raises(PipelineInputError):
        counter.select(counts, 0)
