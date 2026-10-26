"""Survivor-neighborhood recomputation against an independent dense LOF oracle."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from numpy.typing import NDArray
import pytest

from test.test_env_live_lof import brute_lof, save_records
from usim_env_pipeline.iterative_lof import compute_iterative_lof
from usim_env_pipeline.live_lof import compute_lof
from usim_env_pipeline.models import PipelineInputError
from usim_env_pipeline.pcd import sha256


def test_recomputed_survivors_and_original_ids(tmp_path: Path):
    xyz = np.random.default_rng(68).normal(size=(120, 3))
    source = save_records(tmp_path / 'records.npy', xyz)
    before = sha256(source)
    expected = np.zeros(len(xyz), dtype=np.int32)
    ids = np.arange(len(xyz))
    counts: list[int] = []
    for iteration in range(1, 5):
        reject = brute_lof(xyz[ids], (7,))[:, 0] > 1.2
        counts.append(int(np.count_nonzero(reject)))
        expected[ids[reject]] = iteration
        ids = ids[~reject]
    # This fixture must distinguish a real second pass from reusing pass-one scores.
    assert counts[1] > 0 and counts[2] > 0
    result = compute_iterative_lof(
        source, tmp_path / 'cache', neighbors=7, threshold=1.2, iterations=4, workers=1
    )
    np.testing.assert_array_equal(result.removed_at, expected)
    assert [item.rejected for item in result.passes] == counts
    assert result.requested_iterations == 4 and result.stop_reason == 'iteration_limit'
    assert not result.removed_at.flags.writeable
    assert sha256(source) == before == result.source_sha256


def test_iteration_prefix_and_subset_cache_reuse(tmp_path: Path):
    source = save_records(
        tmp_path / 'records.npy', np.random.default_rng(68).normal(size=(120, 3))
    )
    short = compute_iterative_lof(
        source, tmp_path / 'cache', neighbors=7, threshold=1.2, iterations=2
    )
    long = compute_iterative_lof(
        source, tmp_path / 'cache', neighbors=7, threshold=1.2, iterations=4
    )
    np.testing.assert_array_equal(short.removed_at == 1, long.removed_at == 1)
    np.testing.assert_array_equal(short.removed_at == 2, long.removed_at == 2)
    assert all(item.cached for item in long.passes[:2])
    assert not any(item.cached for item in long.passes[2:])


def test_selection_cache_isolation(tmp_path: Path):
    xyz = np.random.default_rng(1).normal(size=(30, 3))
    source = save_records(tmp_path / 'records.npy', xyz)
    for ids in (np.arange(1, 26, dtype=np.int64), np.arange(2, 27, dtype=np.int64)):
        result = compute_lof(source, tmp_path / 'cache', neighbors=(5,), row_ids=ids)
        np.testing.assert_allclose(result.scores, brute_lof(xyz[ids], (5,)), rtol=2e-14)
        assert not result.cached
    assert len(list((tmp_path / 'cache').glob('*/complete.json'))) == 2


@pytest.mark.parametrize(
    'ids',
    [
        np.array([], dtype=np.int64),
        np.array([0, 0], dtype=np.int64),
        np.array([2, 1], dtype=np.int64),
        np.array([-1, 2], dtype=np.int64),
        np.array([1, 9], dtype=np.int64),
        np.array([1, 2], dtype=np.int32),
        np.array([[1, 2]], dtype=np.int64),
    ],
)
def test_invalid_selection(tmp_path: Path, ids: NDArray[np.int64]):
    source = save_records(tmp_path / 'records.npy', np.zeros((8, 3)))
    with pytest.raises(PipelineInputError):
        _ = compute_lof(source, tmp_path / 'cache', neighbors=(1,), row_ids=ids)


def test_early_convergence_with_large_requested_count(tmp_path: Path):
    source = save_records(tmp_path / 'records.npy', np.zeros((30, 3)))
    result = compute_iterative_lof(source, tmp_path / 'cache', neighbors=4, iterations=123)
    assert result.requested_iterations == 123
    assert len(result.passes) == 1 and result.stop_reason == 'converged'
    assert not result.removed_at.any()


def test_insufficient_survivors_stop_without_changing_k(tmp_path: Path):
    source = save_records(tmp_path / 'records.npy', np.zeros((30, 3)))
    result = compute_iterative_lof(
        source, tmp_path / 'cache', neighbors=4, iterations=3, threshold=0.5
    )
    assert result.stop_reason == 'insufficient_points'
    assert len(result.passes) == 1 and result.passes[0].retained == 0
    np.testing.assert_array_equal(result.removed_at, 1)


@pytest.mark.parametrize('iterations', [0, -1, True, 1.5, 2**31])
def test_invalid_iterations(tmp_path: Path, iterations: int):
    with pytest.raises(PipelineInputError):
        _ = compute_iterative_lof(
            tmp_path / 'absent.npy', tmp_path / 'cache', iterations=iterations
        )


@pytest.mark.parametrize('threshold', [0, -1, True, np.nan, np.inf])
def test_invalid_threshold(tmp_path: Path, threshold: float):
    with pytest.raises(PipelineInputError):
        _ = compute_iterative_lof(tmp_path / 'absent.npy', tmp_path / 'cache', threshold=threshold)


def test_module_cli_writes_original_order_removal_iterations(tmp_path: Path):
    source = save_records(
        tmp_path / 'records.npy', np.random.default_rng(68).normal(size=(120, 3))
    )
    _ = subprocess.run(
        [
            sys.executable,
            '-m',
            'usim_env_pipeline.iterative_lof',
            '--records',
            str(source),
            '--cache',
            str(tmp_path / 'cache'),
            '--output',
            str(tmp_path / 'output'),
            '--k',
            '7',
            '--threshold',
            '1.2',
            '--iterations',
            '4',
            '--workers',
            '1',
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads((tmp_path / 'output/result.json').read_text())
    removed = np.load(tmp_path / 'output/removed_at.npy')
    assert report['requested_iterations'] == 4
    assert removed.shape == (120,) and removed.dtype == np.int32
    assert sum(item['rejected'] for item in report['passes']) == np.count_nonzero(removed)
