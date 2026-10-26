"""Exact global SOR decisions and lossless source-row provenance."""

import hashlib
import json
from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
import pytest

from test.env_pipeline_fixtures import plane_records, write_pcd
from usim_env_pipeline.models import PipelineInputError, Source
from usim_env_pipeline.point_filter import SORConfig, statistical_outliers
from usim_env_pipeline.point_filter_output import filter_source, main
from usim_env_pipeline.point_filter_retune import retune_filter

pytest.importorskip('scipy')


class ConfigOverrides(TypedDict, total=False):
    neighbors: int
    workers: int
    batch_size: int
    std_ratio: float


def test_global_sor_matches_all_pairs_and_removes_isolated_points():
    # Given a plane crossing x=30 and two distant source observations.
    records = plane_records()
    points = np.vstack(
        [np.column_stack([records[f] for f in ('x', 'y', 'z')]), [[100, 100, 20]]]
    ).astype(np.float64)
    original = points.copy()
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    expected = np.sort(distances, axis=1)[:, :20].mean(axis=1)
    threshold = expected.mean() + 3 * expected.std(ddof=1)
    # When querying global neighborhoods in small batches.
    result = statistical_outliers(points, SORConfig(batch_size=7, workers=1))
    # Then actual distances and decisions agree with an independent all-pairs oracle.
    np.testing.assert_allclose(result.mean_distances, expected, rtol=1e-12)
    np.testing.assert_array_equal(result.keep, expected <= threshold)
    assert result.keep[:-1].all()
    assert not result.keep[-1]
    np.testing.assert_array_equal(points, original)


def test_sor_is_independent_of_batch_size_and_source_order():
    # Given the same geometry in two orders, including a sparse outlier.
    points = np.column_stack([plane_records()[f] for f in ('x', 'y', 'z')]).astype(np.float64)
    points = np.vstack([points, [400, 200, 50]])
    first = statistical_outliers(points, SORConfig(batch_size=1, workers=1))
    # When querying the reversed source with a different batch size.
    second = statistical_outliers(points[::-1], SORConfig(batch_size=31, workers=1))
    # Then order and batching cannot create tile-like density changes.
    np.testing.assert_allclose(second.mean_distances[::-1], first.mean_distances)
    np.testing.assert_array_equal(second.keep[::-1], first.keep)


def test_coincident_records_exclude_only_one_self_and_zero_variance_is_retained():
    # Given coincident source observations, which standard SOR counts separately.
    points = np.zeros((25, 3), dtype=np.float64)
    # When their k-other-record mean distances are evaluated.
    result = statistical_outliers(points)
    # Then zero spread retains all observations, including threshold equality.
    np.testing.assert_array_equal(result.mean_distances, 0)
    assert result.keep.all()
    assert result.threshold == 0


def test_float64_source_spacing_survives_large_world_coordinates():
    # Given submeter spacing that float32 would collapse at a large world offset.
    points = np.column_stack([np.arange(30) * 0.125, np.zeros(30), np.zeros(30)])
    expected = statistical_outliers(points, SORConfig(neighbors=3))
    # When querying the same source after a large translation.
    translated = statistical_outliers(points + 1e9, SORConfig(neighbors=3))
    # Then the actual float64 distances and decisions remain unchanged.
    np.testing.assert_array_equal(translated.mean_distances, expected.mean_distances)
    np.testing.assert_array_equal(translated.keep, expected.keep)


@pytest.mark.parametrize(
    'config',
    [
        {'neighbors': 0},
        {'workers': 0},
        {'batch_size': 0},
        {'std_ratio': 0},
        {'std_ratio': float('nan')},
        {'std_ratio': float('inf')},
    ],
)
def test_invalid_sor_configuration_is_rejected(config: ConfigOverrides):
    # Given an invalid boundary configuration.
    # When parsing the configuration.
    with pytest.raises(PipelineInputError):
        # Then no invalid query can reach native code.
        _ = SORConfig(**config)


@pytest.mark.parametrize(
    'points', [np.zeros((20, 3)), np.zeros((30, 2)), np.array([[np.nan, 0, 0]])]
)
def test_invalid_cloud_is_rejected(points: NDArray[np.float64]):
    # Given too few records, wrong dimensions or nonfinite XYZ.
    # When attempting the filter.
    with pytest.raises(PipelineInputError):
        # Then the input contract is rejected.
        _ = statistical_outliers(points)


def test_source_archive_attributes_and_ids_survive_filtering(tmp_path: Path):
    # Given original normal/intensity/curvature fields plus an isolated last row.
    base = plane_records()
    records = np.concatenate([base, base[:1].copy()])
    records['x'][-1], records['y'][-1], records['z'][-1] = 200, 200, 80
    path = tmp_path / 'survey.pcd'
    original = write_pcd(path, records, encoding='ascii')
    output = tmp_path / 'filtered'
    # When publishing a first-stage selection, without starting meshing.
    report = filter_source(
        Source(path, expected_sha256=hashlib.sha256(original).hexdigest()), output
    )
    # Then all source rows and fields survive, and the mask partitions the original IDs.
    archived = np.load(output / 'records.npy', allow_pickle=False)
    assert archived.dtype == records.dtype
    np.testing.assert_array_equal(archived, records)
    with np.load(output / 'decisions.npz') as decisions:
        np.testing.assert_array_equal(decisions['ids'], np.arange(len(records)))
        assert decisions['keep'][:-1].all()
        assert not decisions['keep'][-1]
        assert len(decisions['mean_distances']) == len(records)
    assert path.read_bytes() == original
    assert (output / 'source' / path.name).read_bytes() == original
    assert report['retained_records'] == len(base)
    assert report['removed_records'] == 1
    assert sum(report['distance_histogram_counts']) == len(records)
    assert report == json.loads((output / 'report.json').read_text())
    assert not (output / 'meshes').exists()


def test_existing_output_is_not_overwritten(tmp_path: Path):
    # Given a directory containing a user's unrelated file.
    output = tmp_path / 'existing'
    output.mkdir()
    retained = output / 'keep.txt'
    retained.write_text('user data')
    # When requesting the same output location.
    with pytest.raises(PipelineInputError):
        _ = filter_source(Source(tmp_path / 'unused.pcd'), output)
    # Then the existing contents are unchanged.
    assert retained.read_text() == 'user data'


def test_cli_filters_without_optional_geometry_or_usd(tmp_path: Path):
    # Given a valid input for the standalone CPU stage.
    path, output = tmp_path / 'survey.pcd', tmp_path / 'out'
    _ = write_pcd(path, plane_records())
    # When invoking the real CLI without requesting comparison publication.
    status = main(['--source', str(path), '--output', str(output)])
    # Then filtering succeeds without starting downstream reconstruction.
    assert status == 0
    assert json.loads((output / 'report.json').read_text())['status'] == 'complete'
    assert not (output / 'world.usda').exists()


def test_retuning_reuses_distances_and_preserves_source_rows(tmp_path: Path):
    # Given a complete original-cloud selection and its immutable artifacts.
    path, previous, output = tmp_path / 'survey.pcd', tmp_path / 'old', tmp_path / 'new'
    records = np.concatenate([plane_records(), plane_records()[:1]])
    records['x'][-1] = 200
    original = write_pcd(path, records)
    _ = filter_source(Source(path), previous)
    old_report = (previous / 'report.json').read_bytes()
    # When strengthening the filter without another neighbor query.
    report = retune_filter(previous, output, 2.0)
    # Then original scores, all attributes and IDs survive; only the cutoff changes.
    with np.load(previous / 'decisions.npz') as old, np.load(output / 'decisions.npz') as new:
        np.testing.assert_array_equal(new['mean_distances'], old['mean_distances'])
        np.testing.assert_array_equal(new['ids'], old['ids'])
        np.testing.assert_array_equal(new['keep'], old['mean_distances'] <= report['threshold_m'])
        assert not np.any(new['keep'] & ~old['keep'])
    np.testing.assert_array_equal(np.load(output / 'records.npy'), records)
    assert (output / 'source' / path.name).read_bytes() == original
    assert (previous / 'report.json').read_bytes() == old_report
    assert report['config']['std_ratio'] == 2.0
