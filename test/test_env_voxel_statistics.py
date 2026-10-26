"""Occupied-voxel semantics and exact Before/After SOR artifacts."""

import csv
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pytest

from usim_env_pipeline.models import PipelineInputError
from usim_env_pipeline.voxel_statistics import main, occupied_voxels, plot_distribution


def test_floor_boundaries_and_negative_coordinates():
    # Given exact cell boundaries and adjacent representable coordinates.
    x = [-0.2, np.nextafter(-0.1, -np.inf), -0.1, -1e-12, 0, 0.1, 0.2]
    points = np.column_stack([x, np.zeros(len(x)), np.zeros(len(x))])
    # When counting in the fixed source-origin grid.
    result = occupied_voxels(points)
    # Then negatives floor rather than truncate and boundary equality starts a cell.
    np.testing.assert_array_equal(result.indices[:, 0], [-2, -1, 0, 1, 2])
    np.testing.assert_array_equal(result.indices[:, 1:], 0)
    np.testing.assert_array_equal(result.counts, [2, 2, 1, 1, 1])


def test_positive_boundary_neighbors_are_not_snapped():
    # Given immediately below, exactly on, and immediately above a boundary.
    x = [np.nextafter(0.1, -np.inf), 0.1, np.nextafter(0.1, np.inf)]
    points = np.column_stack([x, np.zeros(3), np.zeros(3)])
    # When assigning half-open cells.
    result = occupied_voxels(points)
    # Then only the below-boundary record remains in cell zero.
    np.testing.assert_array_equal(result.indices[:, 0], [0, 1])
    np.testing.assert_array_equal(result.counts, [1, 2])


@pytest.mark.parametrize('boundary,index', [(0.3, 3), (0.7, 7), (-0.3, -3), (-0.7, -7)])
def test_decimal_boundaries_and_adjacent_floats_remain_distinct(boundary: float, index: int):
    # Given a nominal decimal boundary and its immediately adjacent representable values.
    x = [np.nextafter(boundary, -np.inf), boundary, np.nextafter(boundary, np.inf)]
    points = np.column_stack([x, np.zeros(3), np.zeros(3)])
    original = points.copy()
    # When assigning half-open cells without a tolerance.
    result = occupied_voxels(points)
    # Then equality begins the new cell and the truly adjacent lower value remains distinct.
    np.testing.assert_array_equal(result.indices[:, 0], [index - 1, index])
    np.testing.assert_array_equal(result.counts, [1, 2])
    np.testing.assert_array_equal(points, original)


def test_structured_duplicates_and_keep_mask_use_independent_occupied_cells():
    # Given duplicate source rows and a second occupied voxel removed entirely.
    records = np.array(
        [(0.01, 0.01, 0.01), (0.01, 0.01, 0.01), (1, 2, 3)],
        dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8')],
    )
    # When counting only retained source rows.
    result = occupied_voxels(records, np.array([True, True, False]))
    # Then duplicates count separately and no emptied voxel enters the distribution.
    np.testing.assert_array_equal(result.indices, [[0, 0, 0]])
    np.testing.assert_array_equal(result.counts, [2])
    assert result.summary['total_points'] == 2
    assert result.summary['occupied_voxels'] == 1


@pytest.mark.parametrize('masked', [False, True])
def test_empty_population_has_no_voxels_or_undefined_numeric_statistics(masked: bool):
    # Given no input rows, or a mask retaining no rows.
    points = np.zeros((3 if masked else 0, 3))
    keep = np.zeros(3, dtype=np.bool_) if masked else None
    # When evaluating the selected population.
    result = occupied_voxels(points, keep)
    # Then all arrays are empty and undefined statistics are JSON-safe nulls.
    assert result.indices.shape == (0, 3)
    assert result.counts.size == result.point_counts.size == result.frequencies.size == 0
    assert result.summary == {
        'min': None,
        'max': None,
        'mean': None,
        'median': None,
        'occupied_voxels': 0,
        'total_points': 0,
    }


def test_sparse_huge_world_uses_only_occupied_indices():
    # Given two observations whose bounding volume cannot be allocated.
    points = np.array([[-1e12, 0, 1e12], [1e12, 0, -1e12]])
    # When computing sparse occupancy.
    result = occupied_voxels(points)
    # Then only the two cells exist with full signed int64 coordinates.
    np.testing.assert_array_equal(result.indices, [[-(10**13), 0, 10**13], [10**13, 0, -(10**13)]])
    np.testing.assert_array_equal(result.counts, [1, 1])


@pytest.mark.parametrize('counts,median', [([1, 2, 2, 9], 2), ([1, 2, 5, 9], 3.5)])
def test_exact_frequency_conserves_points_and_uses_voxel_median(counts: list[int], median: float):
    # Given known cell populations, including a long tail.
    points = np.repeat(np.column_stack([np.arange(4), np.zeros(4), np.zeros(4)]), counts, axis=0)
    # When computing the exact discrete distribution.
    result = occupied_voxels(points)
    # Then its weighted sum counts source rows and median weights each occupied voxel once.
    assert int(np.dot(result.point_counts, result.frequencies)) == sum(counts)
    assert int(result.frequencies.sum()) == 4
    assert result.summary['median'] == median
    assert result.summary['mean'] == sum(counts) / 4
    assert result.summary['min'] == 1
    assert result.summary['max'] == 9


@pytest.mark.parametrize(
    'points', [np.zeros((2, 2)), np.array([[np.nan, 0, 0]]), np.array([[1e20, 0, 0]])]
)
def test_invalid_xyz_is_rejected(points: NDArray[np.float64]):
    # Given malformed, nonfinite, or nonrepresentable voxel coordinates.
    # When parsing the analysis input.
    with pytest.raises(PipelineInputError):
        # Then invalid indices cannot silently overflow or enter the statistics.
        _ = occupied_voxels(points)


@pytest.mark.parametrize('keep', [np.array([1, 0]), np.array([True])])
def test_invalid_mask_is_rejected(keep: NDArray[np.bool_] | NDArray[np.int64]):
    # Given a mask with wrong dtype or row count.
    # When selecting the source population.
    with pytest.raises(PipelineInputError):
        # Then it cannot be interpreted as row indices or broadcast.
        _ = occupied_voxels(np.zeros((2, 3)), np.asarray(keep))


def test_cli_exports_exact_before_after_csv_json_and_chart(tmp_path: Path):
    # Given an archived SOR source with counts 1, 2 and 100, and two retained duplicates.
    _ = pytest.importorskip('matplotlib')
    source, destination = tmp_path / 'sor', tmp_path / 'statistics'
    source.mkdir()
    records = np.zeros(103, dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8')])
    records['x'][1:3], records['x'][3:] = 1, 2
    keep = np.zeros(103, dtype=np.bool_)
    keep[1:3] = True
    np.save(source / 'records.npy', records)
    np.savez(source / 'decisions.npz', ids=np.arange(103, dtype=np.int64), keep=keep)
    # When driving the actual standalone CLI.
    status = main(['--filter-output', str(source), '--destination', str(destination)])
    # Then both exact distributions and independent summaries are published, without tails lost.
    assert status == 0
    for name, expected in (('before', [(1, 1), (2, 1), (100, 1)]), ('after', [(2, 1)])):
        with (destination / f'{name}.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert [(int(r['points_per_voxel']), int(r['occupied_voxels'])) for r in rows] == expected
    summary = json.loads((destination / 'summary.json').read_text())
    assert summary['grid']['origin_m'] == [0, 0, 0]
    assert summary['grid']['voxel_size_m'] == 0.1
    assert summary['before']['total_points'] == 103
    assert summary['before']['median'] == 2
    assert summary['after']['occupied_voxels'] == 1
    assert summary['after']['total_points'] == 2
    assert (destination / 'distribution.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n')


def test_empty_comparison_renders_without_log_zero_failure(tmp_path: Path):
    # Given two completely empty clouds.
    _ = pytest.importorskip('matplotlib')
    empty = occupied_voxels(np.empty((0, 3)))
    path = tmp_path / 'empty.png'
    # When rendering the full and low-count panels.
    plot_distribution(empty, empty, path)
    # Then a real image is produced without fictitious occupied voxels.
    assert path.stat().st_size > 0


def test_cli_exports_three_sequential_selections(tmp_path: Path):
    # Given source rows, a weak-SOR mask and a stricter geometric selection.
    _ = pytest.importorskip('matplotlib')
    source, destination = tmp_path / 'staged', tmp_path / 'statistics'
    source.mkdir()
    records = np.zeros(6, dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8')])
    records['x'] = [0, 1, 1, 2, 2, 2]
    sor_keep = np.array([False, True, True, True, True, True])
    keep = np.array([False, True, True, False, False, False])
    np.save(source / 'records.npy', records)
    np.savez(
        source / 'decisions.npz',
        ids=np.arange(6, dtype=np.int64),
        sor_keep=sor_keep,
        keep=keep,
    )
    # When exporting the same grid for every processing stage.
    status = main(['--filter-output', str(source), '--destination', str(destination)])
    # Then all three summaries and exact frequency files describe their own occupied cells.
    summary = json.loads((destination / 'summary.json').read_text())
    assert status == 0
    assert summary['before']['total_points'] == 6
    assert summary['after_sor']['total_points'] == 5
    assert summary['after']['total_points'] == 2
    assert summary['after_sor']['median'] == 2.5
    with (destination / 'after_sor.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert [(int(r['points_per_voxel']), int(r['occupied_voxels'])) for r in rows] == [
        (2, 1),
        (3, 1),
    ]
