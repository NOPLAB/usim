"""Independent inclusive thresholds, source-row mapping and CPU CLI publication."""

from dataclasses import FrozenInstanceError
from collections.abc import Sequence
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from numpy.typing import NDArray
import pytest

from usim_env_pipeline.models import PipelineInputError
from usim_env_pipeline.voxel_statistics import occupied_voxels
from usim_env_pipeline.voxel_thresholds import THRESHOLDS, analyze_thresholds, publish_thresholds


def records_at(x: Sequence[float] | NDArray[np.float64]) -> NDArray[np.void]:
    records = np.zeros(
        len(x), dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('intensity', '<f4')]
    )
    records['x'] = x
    records['intensity'] = np.arange(len(x))
    return records


def test_inclusive_thresholds_use_same_sor_population_and_one_sparse_grouping(monkeypatch):
    # Each exact threshold and its next count occupy separate cells.
    counts = np.array([5, 6, 10, 11, 20, 21, 40, 41])
    records = records_at(np.repeat(np.arange(len(counts), dtype=np.float64), counts))
    sor_keep = np.ones(len(records), dtype=np.bool_)
    original_records, original_keep = records.copy(), sor_keep.copy()
    unique = np.unique
    grouping_calls = []

    def tracked_unique(*args, **kwargs):
        if kwargs.get('axis') == 0:
            grouping_calls.append(kwargs.copy())
        return unique(*args, **kwargs)

    monkeypatch.setattr(np, 'unique', tracked_unique)
    result = analyze_thresholds(records, sor_keep)
    assert len(grouping_calls) == 1
    assert grouping_calls[0]['return_inverse'] is True
    assert grouping_calls[0]['return_counts'] is True
    assert result.counts_per_record.dtype == np.int64
    np.testing.assert_array_equal(result.counts_per_record, np.repeat(counts, counts))
    assert isinstance(result.selections, tuple)
    assert [item.threshold for item in result.selections] == list(THRESHOLDS)
    for selection in result.selections:
        expected = counts > selection.threshold
        assert selection.keep.dtype == np.bool_
        np.testing.assert_array_equal(selection.keep, np.repeat(expected, counts))
        np.testing.assert_array_equal(selection.stats.counts, counts[expected])
        assert selection.stats.summary['total_points'] == int(counts[expected].sum())
        assert selection.stats.summary['occupied_voxels'] == int(expected.sum())
        assert int(np.dot(selection.stats.point_counts, selection.stats.frequencies)) == int(
            counts[expected].sum()
        )
    np.testing.assert_array_equal(records, original_records)
    np.testing.assert_array_equal(sor_keep, original_keep)
    with pytest.raises(FrozenInstanceError):
        setattr(result, 'sor_stats', result.sor_stats)
    with pytest.raises(FrozenInstanceError):
        setattr(result.selections[0], 'threshold', 0)


def test_sor_exclusions_duplicate_counts_and_original_row_mapping():
    # Interleave cells and remove records both inside surviving cells and whole cells.
    records = records_at([1, 0, 2, 0, 1, 0, 0, 0, 0, 1, 0])
    sor_keep = np.array([True, True, False, True, False, True, True, True, True, True, False])
    result = analyze_thresholds(records, sor_keep)
    np.testing.assert_array_equal(result.counts_per_record, [2, 6, 0, 6, 0, 6, 6, 6, 6, 2, 0])
    np.testing.assert_array_equal(
        result.selections[0].keep,
        [False, True, False, True, False, True, True, True, True, False, False],
    )
    np.testing.assert_array_equal(result.sor_stats.counts, [6, 2])
    assert result.selections[0].stats.summary['total_points'] == 6
    assert not any(item.keep.any() for item in result.selections[1:])


@pytest.mark.parametrize('boundary', [0.3, 0.7, -0.3, -0.7, 0.0, -0.1])
def test_boundaries_follow_existing_float64_multiply_floor(boundary: float):
    x = [np.nextafter(boundary, -np.inf)] * 6 + [
        boundary,
        np.nextafter(boundary, np.inf),
    ] * 6
    records = records_at(x)
    result = analyze_thresholds(records, np.ones(len(records), dtype=np.bool_))
    expected = occupied_voxels(records)
    np.testing.assert_array_equal(result.sor_stats.indices, expected.indices)
    np.testing.assert_array_equal(result.sor_stats.counts, expected.counts)
    assert result.sor_stats.summary == expected.summary
    for selection in result.selections:
        expected = occupied_voxels(records, selection.keep)
        np.testing.assert_array_equal(selection.stats.indices, expected.indices)
        np.testing.assert_array_equal(selection.stats.counts, expected.counts)
        np.testing.assert_array_equal(selection.stats.point_counts, expected.point_counts)
        np.testing.assert_array_equal(selection.stats.frequencies, expected.frequencies)
        assert selection.stats.summary == expected.summary


@pytest.mark.parametrize('rows,retained', [(0, False), (7, False), (5, True)])
def test_empty_sor_or_all_voxel_removed_safe(rows: int, retained: bool):
    records = records_at([0.0] * rows)
    result = analyze_thresholds(records, np.full(rows, retained, dtype=np.bool_))
    for selection in result.selections:
        assert not selection.keep.any()
        assert selection.stats.indices.shape == (0, 3)
        assert selection.stats.counts.size == 0
        assert selection.stats.point_counts.size == selection.stats.frequencies.size == 0
        assert selection.stats.summary == {
            'min': None,
            'max': None,
            'mean': None,
            'median': None,
            'occupied_voxels': 0,
            'total_points': 0,
        }
    if not retained:
        np.testing.assert_array_equal(result.counts_per_record, np.zeros(rows, dtype=np.int64))
        assert result.sor_stats.summary['total_points'] == 0


def test_sparse_huge_coordinates_never_allocate_world_bounds():
    records = records_at([-1e12] * 6 + [1e12] * 41)
    records['z'] = -records['x']
    result = analyze_thresholds(records, np.ones(len(records), dtype=np.bool_))
    np.testing.assert_array_equal(
        result.sor_stats.indices, [[-(10**13), 0, 10**13], [10**13, 0, -(10**13)]]
    )
    np.testing.assert_array_equal(result.sor_stats.counts, [6, 41])
    np.testing.assert_array_equal(result.selections[-1].stats.counts, [41])


@pytest.mark.parametrize('x', [[np.nan], [np.inf], [-np.inf], [1e20], [-(2.0**63)]])
def test_invalid_coordinates_rejected(x: list[float]):
    records = records_at(x)
    with pytest.raises(PipelineInputError):
        _ = analyze_thresholds(records, np.ones(len(records), dtype=np.bool_))


@pytest.mark.parametrize('keep', [np.array([1, 0]), np.array([True]), np.ones((2, 1), bool)])
def test_invalid_sor_masks_rejected(keep):
    with pytest.raises(PipelineInputError):
        _ = analyze_thresholds(records_at([0, 1]), keep)


@pytest.mark.parametrize(
    'records',
    [
        np.zeros((2, 3)),
        np.zeros(2, dtype=[('x', '<f8'), ('y', '<f8')]),
        np.zeros(2, dtype=[('x', '<f8', 2), ('y', '<f8'), ('z', '<f8')]),
    ],
)
def test_malformed_source_records_rejected(records):
    with pytest.raises(PipelineInputError):
        _ = analyze_thresholds(records, np.ones(len(records), dtype=np.bool_))


def make_source(source: Path, fallback: bool = False):
    source.mkdir()
    (source / 'source').mkdir()
    _ = (source / 'source' / 'input.pcd').write_bytes(b'exact original archive\x00\xff')
    records = records_at([0] * 6 + [1] * 11 + [2] * 21 + [3] * 41 + [4])
    sor_keep = np.ones(len(records), dtype=np.bool_)
    sor_keep[-1] = False
    geometry_keep = np.zeros(len(records), dtype=np.bool_)
    geometry_keep[:6] = True
    np.save(source / 'records.npy', records, allow_pickle=False)
    decisions: dict[str, NDArray[np.int64] | NDArray[np.bool_]] = {
        'ids': np.arange(len(records), dtype=np.int64)
    }
    if fallback:
        decisions['keep'] = sor_keep
    else:
        decisions.update(
            sor_keep=sor_keep, keep=geometry_keep, geometry_rejected=sor_keep & ~geometry_keep
        )
    np.savez(source / 'decisions.npz', **decisions)
    report = {
        'schema': 1,
        'source': {'path': 'source/input.pcd', 'sha256': 'unchanged'},
        'frame': 'source XYZ; meters; Z-up',
        'color': 'source intensity',
        'config': {'neighbors': 20, 'std_ratio': 4.0},
        'geometry': {'retained_records': 6},
    }
    _ = (source / 'report.json').write_text(json.dumps(report), encoding='utf-8')
    return records, sor_keep, report


@pytest.mark.parametrize('fallback', [False, True])
def test_actual_cli_ignores_geometry_and_preserves_source(tmp_path: Path, fallback: bool):
    _ = pytest.importorskip('matplotlib')
    source, output = tmp_path / 'source_output', tmp_path / 'thresholds'
    records, sor_keep, source_report = make_source(source, fallback)
    output.mkdir()  # An existing empty destination is also valid.
    environment = {
        **os.environ,
        'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src'),
    }
    completed = subprocess.run(
        [
            sys.executable,
            '-m',
            'usim_env_pipeline.voxel_thresholds',
            '--source-output',
            str(source),
            '--output',
            str(output),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (output / 'records.npy').read_bytes() == (source / 'records.npy').read_bytes()
    assert (output / 'source' / 'input.pcd').read_bytes() == (
        source / 'source' / 'input.pcd'
    ).read_bytes()
    analysis = analyze_thresholds(records, sor_keep)
    with np.load(output / 'decisions.npz', allow_pickle=False) as saved:
        assert set(saved.files) == {
            'ids',
            'sor_keep',
            'counts_per_record',
            'keep_5',
            'keep_10',
            'keep_20',
            'keep_40',
        }
        np.testing.assert_array_equal(saved['ids'], np.arange(len(records), dtype=np.int64))
        np.testing.assert_array_equal(saved['sor_keep'], sor_keep)
        np.testing.assert_array_equal(saved['counts_per_record'], analysis.counts_per_record)
        for selection in analysis.selections:
            np.testing.assert_array_equal(saved[f'keep_{selection.threshold}'], selection.keep)
    report = json.loads((output / 'report.json').read_text(encoding='utf-8'))
    assert set(report) == {
        'schema',
        'source',
        'frame',
        'color',
        'source_config',
        'records',
        'sor_removed_records',
        'modes',
    }
    assert report['schema'] == 1
    for field in ('source', 'frame', 'color'):
        assert report[field] == source_report[field]
    assert report['source_config'] == source_report['config']
    assert report['records'] == 80
    assert report['sor_removed_records'] == 1
    assert [mode['name'] for mode in report['modes']] == [
        'SOR',
        'Voxel5',
        'Voxel10',
        'Voxel20',
        'Voxel40',
    ]
    assert [mode['threshold'] for mode in report['modes']] == [0, 5, 10, 20, 40]
    assert [mode['retained_records'] for mode in report['modes']] == [79, 79, 73, 62, 41]
    assert [mode['voxel_removed_records'] for mode in report['modes']] == [0, 0, 6, 17, 38]
    assert [mode['color'] for mode in report['modes']] == [
        [1, 0, 0],
        [1, 0.5, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0.4, 1],
    ]
    stats = [analysis.sor_stats] + [item.stats for item in analysis.selections]
    for mode, expected in zip(report['modes'], stats):
        for field in ('occupied_voxels', 'min', 'max', 'mean', 'median'):
            assert mode[field] == expected.summary[field]
        with (output / f'{mode["name"]}.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert [(int(r['points_per_voxel']), int(r['occupied_voxels'])) for r in rows] == list(
            zip(expected.point_counts.tolist(), expected.frequencies.tolist())
        )
    assert (output / 'distribution.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    np.testing.assert_array_equal(np.load(source / 'records.npy', allow_pickle=False), records)


def test_publisher_rejects_nonempty_destination_and_invalid_source_ids(tmp_path: Path):
    source, output = tmp_path / 'source_output', tmp_path / 'thresholds'
    records, sor_keep, _ = make_source(source)
    output.mkdir()
    sentinel = output / 'do-not-overwrite'
    _ = sentinel.write_bytes(b'untouched')
    with pytest.raises(PipelineInputError, match='absent or empty'):
        _ = publish_thresholds(source, output)
    assert sentinel.read_bytes() == b'untouched'
    np.savez(source / 'decisions.npz', ids=np.arange(len(records))[::-1], sor_keep=sor_keep)
    with pytest.raises(PipelineInputError, match='original source-row IDs'):
        _ = publish_thresholds(source, tmp_path / 'invalid')
    assert not (tmp_path / 'invalid').exists()
