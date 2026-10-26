"""Streamed LOF against an independent dense-distance oracle without an on-disk graph."""

from dataclasses import FrozenInstanceError
from pathlib import Path
import subprocess
import sys

import numpy as np
from numpy.typing import NDArray
import pytest

from usim_env_pipeline.live_lof import compute_lof
from usim_env_pipeline.models import PipelineInputError


def save_records(path: Path, xyz: NDArray[np.float64]) -> Path:
    records = np.zeros(
        len(xyz), dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('intensity', '<f4')]
    )
    for axis, name in enumerate(('x', 'y', 'z')):
        records[name] = xyz[:, axis]
    records['intensity'] = np.arange(len(xyz))
    np.save(path, records)
    return path


def brute_lof(xyz: NDArray[np.float64], neighbors: tuple[int, ...]) -> NDArray[np.float64]:
    distances = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    ids = np.argsort(distances, axis=1, kind='stable')
    output = np.empty((len(xyz), len(neighbors)), dtype=np.float64)
    for column, k in enumerate(neighbors):
        chosen = ids[:, :k]
        kth = np.array([distances[i, ids[i, k - 1]] for i in range(len(xyz))])
        lrd = np.array(
            [
                1.0 / (sum(max(distances[i, j], kth[j]) for j in chosen[i]) / k + 1e-10)
                for i in range(len(xyz))
            ]
        )
        output[:, column] = [sum(lrd[j] / lrd[i] for j in chosen[i]) / k for i in range(len(xyz))]
    return output


@pytest.mark.parametrize('batch_size', [1, 7, 8192])
def test_multi_k_brute_oracle_and_original_order(tmp_path: Path, batch_size: int):
    xyz = np.random.default_rng(571).normal(size=(53, 3))
    neighbors = (9, 2, 17)
    source = save_records(tmp_path / 'records.npy', xyz)
    result = compute_lof(
        source, tmp_path / 'cache', neighbors=neighbors, batch_size=batch_size, workers=1
    )
    np.testing.assert_allclose(result.scores, brute_lof(xyz, neighbors), rtol=2e-14, atol=2e-14)
    assert result.neighbors == neighbors
    assert result.scores.dtype == np.float64 and result.scores.shape == (53, 3)
    assert isinstance(result.scores, np.memmap) and not result.scores.flags.writeable
    assert result.elapsed_seconds > 0 and not result.cached
    with pytest.raises(FrozenInstanceError):
        setattr(result, 'cached', True)
    permutation = np.random.default_rng(89).permutation(len(xyz))
    save_records(source, xyz[permutation])
    shuffled = compute_lof(source, tmp_path / 'cache', neighbors=neighbors, batch_size=batch_size)
    np.testing.assert_allclose(shuffled.scores, result.scores[permutation], rtol=2e-14)


def test_identical_duplicates_are_finite_one(tmp_path: Path):
    source = save_records(tmp_path / 'records.npy', np.full((37, 3), -2.0))
    result = compute_lof(source, tmp_path / 'cache', neighbors=(1, 5, 36), batch_size=4)
    np.testing.assert_array_equal(result.scores, np.ones((37, 3)))
    assert not list((tmp_path / 'cache').glob('*/neighbors.npy'))


def test_self_exclusion_and_tied_zero_and_nonzero_distances(tmp_path: Path):
    xyz = np.zeros((24, 3))
    xyz[20:, 0] = [-1, 1, -4, 4]
    source = save_records(tmp_path / 'records.npy', xyz)
    result = compute_lof(source, tmp_path / 'cache', neighbors=(3, 21), batch_size=3)
    np.testing.assert_allclose(result.scores, brute_lof(xyz, (3, 21)), rtol=2e-14)
    distances = np.load(next((tmp_path / 'cache').glob('*/kdist.npy')), mmap_mode='r')
    np.testing.assert_array_equal(distances[:20, 0], 0)


def test_translation_rotation_invariance(tmp_path: Path):
    xyz = np.random.default_rng(705).normal(size=(72, 3))
    source = save_records(tmp_path / 'records.npy', xyz)
    baseline = compute_lof(source, tmp_path / 'cache', neighbors=(7, 19), batch_size=9)
    rotation, _ = np.linalg.qr(np.random.default_rng(5).normal(size=(3, 3)))
    save_records(source, xyz @ rotation + [134.0, -52.0, 20.0])
    transformed = compute_lof(source, tmp_path / 'cache', neighbors=(7, 19), batch_size=12)
    np.testing.assert_allclose(transformed.scores, baseline.scores, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize('planar', [False, True])
def test_two_normal_sampling_densities_and_detached_outlier(tmp_path: Path, planar: bool):
    random = np.random.default_rng(867)
    # Two independently measured surfaces with a tenfold difference in spacing.
    dense = random.uniform(-1, 1, (350, 3))
    sparse = random.uniform(-10, 10, (350, 3)) + [60, 0, 0]
    if planar:
        dense[:, 2] = 0
        sparse[:, 2] = sparse[:, 1] * 0.5
    xyz = np.vstack((dense, sparse, [[130.0, 75.0, 50.0]]))
    source = save_records(tmp_path / 'records.npy', xyz)
    result = compute_lof(source, tmp_path / 'cache', neighbors=(10, 25), batch_size=41)
    np.testing.assert_allclose(result.scores, brute_lof(xyz, (10, 25)), rtol=2e-14)
    assert np.quantile(result.scores[:350], 0.95) < 1.5
    assert np.quantile(result.scores[350:700], 0.95) < 1.5
    assert np.all(result.scores[-1] > 5)


def test_cache_reuse_source_isolation_and_invalid_completed_scores(tmp_path: Path):
    xyz = np.random.default_rng(68).normal(size=(31, 3))
    source = save_records(tmp_path / 'records.npy', xyz)
    cache = tmp_path / 'cache'
    initial = compute_lof(source, cache, neighbors=(4, 8), batch_size=5)
    again = compute_lof(source, cache, neighbors=(4, 8), batch_size=11, workers=1)
    assert again.cached and not again.scores.flags.writeable
    assert again.source_sha256 == initial.source_sha256
    np.testing.assert_array_equal(again.scores, initial.scores)
    save_records(source, xyz * np.array([1, 2, 3]))
    changed = compute_lof(source, cache, neighbors=(4, 8))
    assert not changed.cached and changed.source_sha256 != initial.source_sha256
    assert len(list(cache.glob('*/complete.json'))) == 2
    reordered = compute_lof(source, cache, neighbors=(8, 4))
    np.testing.assert_array_equal(reordered.scores, changed.scores[:, ::-1])
    scores_path = next(
        path
        for path in cache.glob('*/scores.npy')
        if changed.source_sha256 in str(path) and path.parent.name.endswith('-4-8')
    )
    del changed, reordered
    damaged = np.load(scores_path, mmap_mode='r+')
    damaged[0, 0] = np.nan
    damaged.flush()
    with pytest.raises(PipelineInputError, match='finite'):
        compute_lof(source, cache, neighbors=(4, 8))


@pytest.mark.parametrize('neighbors', [(), (0,), (-1,), (2, 2), (True,), (1.5,), (8,)])
def test_invalid_neighbors(tmp_path: Path, neighbors):
    source = save_records(tmp_path / 'records.npy', np.zeros((8, 3)))
    with pytest.raises(PipelineInputError):
        compute_lof(source, tmp_path / 'cache', neighbors=neighbors)
    assert not (tmp_path / 'cache').exists()


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_nonfinite_source_rejected(tmp_path: Path, value: float):
    xyz = np.zeros((5, 3))
    xyz[3, 2] = value
    source = save_records(tmp_path / 'records.npy', xyz)
    with pytest.raises(PipelineInputError, match='finite'):
        compute_lof(source, tmp_path / 'cache', neighbors=(2,), batch_size=2)
    assert not list(tmp_path.glob('cache/*/complete.json'))


@pytest.mark.parametrize(
    'records',
    [
        np.zeros((5, 3)),
        np.zeros(5, dtype=[('x', 'f8'), ('y', 'f8')]),
        np.zeros(5, dtype=[('x', 'f8', 2), ('y', 'f8'), ('z', 'f8')]),
        np.zeros(5, dtype=[('x', 'c16'), ('y', 'f8'), ('z', 'f8')]),
    ],
)
def test_invalid_record_fields(tmp_path: Path, records):
    source = tmp_path / 'records.npy'
    np.save(source, records)
    with pytest.raises(PipelineInputError, match='scalar numeric'):
        compute_lof(source, tmp_path / 'cache', neighbors=(2,))


@pytest.mark.parametrize('batch_size,workers', [(0, 1), (1, 0), (True, 1), (1, -1)])
def test_invalid_execution_parameters(tmp_path: Path, batch_size: int, workers: int):
    source = save_records(tmp_path / 'records.npy', np.zeros((5, 3)))
    with pytest.raises(PipelineInputError):
        compute_lof(
            source, tmp_path / 'cache', neighbors=(2,), batch_size=batch_size, workers=workers
        )


def test_optional_sklearn_oracle(tmp_path: Path):
    sklearn_neighbors = pytest.importorskip('sklearn.neighbors')
    xyz = np.random.default_rng(750).normal(size=(49, 3))
    source = save_records(tmp_path / 'records.npy', xyz)
    result = compute_lof(source, tmp_path / 'cache', neighbors=(5, 12))
    for column, k in enumerate(result.neighbors):
        oracle = sklearn_neighbors.LocalOutlierFactor(n_neighbors=k)
        oracle.fit(xyz)
        np.testing.assert_allclose(result.scores[:, column], -oracle.negative_outlier_factor_)


def test_module_cli(tmp_path: Path):
    source = save_records(tmp_path / 'records.npy', np.random.default_rng(37).normal(size=(16, 3)))
    completed = subprocess.run(
        [
            sys.executable,
            '-m',
            'usim_env_pipeline.live_lof',
            '--records',
            str(source),
            '--cache',
            str(tmp_path / 'cache'),
            '--k',
            '3',
            '7',
            '--batch-size',
            '2',
            '--workers',
            '1',
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert 'LOF_COMPLETE' in completed.stdout
    assert len(completed.stdout.splitlines()) <= 32
