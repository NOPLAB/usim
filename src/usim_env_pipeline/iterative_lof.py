"""Recompute exact LOF on survivors, preserving the iteration removed for every source row."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .live_lof import compute_lof
from .models import PipelineInputError


@dataclass(frozen=True, slots=True)
class LofPass:
    iteration: int
    input_records: int
    rejected: int
    retained: int
    cached: bool


@dataclass(frozen=True, slots=True)
class IterativeLofResult:
    removed_at: NDArray[np.int32]
    passes: tuple[LofPass, ...]
    requested_iterations: int
    stop_reason: Literal['iteration_limit', 'converged', 'insufficient_points']
    source_sha256: str
    elapsed_seconds: float


def compute_iterative_lof(
    records_path: Path,
    cache_dir: Path,
    *,
    neighbors: int = 20,
    threshold: float = 1.5,
    iterations: int = 3,
    batch_size: int = 8192,
    workers: int = 8,
) -> IterativeLofResult:
    """Start with all observations; each pass uses only the preceding pass's survivors.

    The same k and strict score > threshold rule apply to every pass. Stop early
    on convergence or when fewer than k + 1 observations remain. Cache identities
    include source content and survivor IDs, so changing parameters cannot reuse
    scores from a different neighborhood. Zero removed_at means retained.
    """
    started = perf_counter()
    if type(iterations) is not int or not 1 <= iterations <= np.iinfo(np.int32).max:
        raise PipelineInputError('Iterations must be a positive int32 integer')
    if type(threshold) not in (int, float) or not isfinite(threshold) or threshold <= 0:
        raise PipelineInputError('Threshold must be positive and finite')
    # compute_lof owns source, neighbor and execution-parameter validation.
    scores = compute_lof(
        records_path, cache_dir, neighbors=(neighbors,), batch_size=batch_size, workers=workers
    )
    source_hash = scores.source_sha256
    removed_at = np.zeros(len(scores.scores), dtype=np.int32)
    survivors = np.arange(len(removed_at), dtype=np.int64)
    passes: list[LofPass] = []
    stop_reason: Literal['iteration_limit', 'converged', 'insufficient_points'] = 'iteration_limit'
    for iteration in range(1, iterations + 1):
        if iteration > 1:
            scores = compute_lof(
                records_path,
                cache_dir,
                neighbors=(neighbors,),
                batch_size=batch_size,
                workers=workers,
                row_ids=survivors,
            )
            if scores.source_sha256 != source_hash:
                raise PipelineInputError('Source changed between LOF iterations')
        rejected = scores.scores[:, 0] > threshold
        count = int(np.count_nonzero(rejected))
        removed_at[survivors[rejected]] = iteration
        report = LofPass(iteration, len(survivors), count, len(survivors) - count, scores.cached)
        passes.append(report)
        print('ITERATIVE_LOF_PASS ' + json.dumps(asdict(report)), flush=True)
        survivors = survivors[~rejected]
        if count == 0:
            stop_reason = 'converged'
            break
        if len(survivors) <= neighbors and iteration < iterations:
            stop_reason = 'insufficient_points'
            break
    removed_at.setflags(write=False)
    return IterativeLofResult(
        removed_at, tuple(passes), iterations, stop_reason, source_hash, perf_counter() - started
    )


@dataclass(init=False)
class _Arguments(argparse.Namespace):
    records: Path
    cache: Path
    k: int
    threshold: float
    iterations: int
    batch_size: int
    workers: int
    output: Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument('--records', type=Path, required=True)
    _ = parser.add_argument('--cache', type=Path, required=True)
    _ = parser.add_argument('--k', type=int, default=20)
    _ = parser.add_argument('--threshold', type=float, default=1.5)
    _ = parser.add_argument('--iterations', type=int, default=3)
    _ = parser.add_argument('--batch-size', type=int, default=8192)
    _ = parser.add_argument('--workers', type=int, default=8)
    _ = parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(namespace=_Arguments())
    result = compute_iterative_lof(
        args.records,
        args.cache,
        neighbors=args.k,
        threshold=args.threshold,
        iterations=args.iterations,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / 'removed_at.npy', result.removed_at)
    report = {
        'neighbors': args.k,
        'threshold': args.threshold,
        'requested_iterations': result.requested_iterations,
        'stop_reason': result.stop_reason,
        'source_sha256': result.source_sha256,
        'source_records': len(result.removed_at),
        'passes': [asdict(item) for item in result.passes],
        'elapsed_seconds': result.elapsed_seconds,
    }
    _ = (args.output / 'result.json').write_text(
        json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8'
    )
    print('ITERATIVE_LOF_COMPLETE ' + json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
