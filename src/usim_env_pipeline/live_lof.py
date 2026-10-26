"""Exact CPU LOF over every source row, with bounded streamed neighbor queries."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256 as hash_bytes
import json
from pathlib import Path
from time import perf_counter
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .models import PipelineInputError
from .pcd import sha256
from .storage import write_json

ALGORITHM_VERSION: Final = 'exact-lof-self-excluded-v2'


@dataclass(frozen=True, slots=True)
class LiveLofResult:
    scores: NDArray[np.float64]
    neighbors: tuple[int, ...]
    elapsed_seconds: float
    cached: bool
    source_sha256: str


def _progress(stage: str, end: int, total: int, previous: int) -> int:
    bucket = end * 10 // total
    if bucket != previous:
        print(f'LOF_STAGE {stage} {end}/{total}', flush=True)
    return bucket


def compute_lof(
    records_path: Path,
    cache_dir: Path,
    *,
    neighbors: tuple[int, ...] = (20, 40, 80),
    batch_size: int = 8192,
    workers: int = 8,
    row_ids: NDArray[np.int64] | None = None,
) -> LiveLofResult:
    """Return readonly original-order LOF columns in the requested k order.

    RAM holds float64 XYZ, the tree and O(N * len(k)) scalar density tables,
    plus O(batch_size * max(k)) buffers. Neighbors are re-queried, not stored.
    Only float64 scores and float32 inspection distances are saved to disk.
    Source content, ordered k values and the algorithm version identify the cache.
    Optional increasing int64 row_ids select source observations without copying
    their records to disk. Scores then follow row_ids, and its content hash is
    included in the cache identity.
    """
    started = perf_counter()
    if (
        not isinstance(neighbors, tuple)
        or not neighbors
        or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in neighbors)
        or len(set(neighbors)) != len(neighbors)
    ):
        raise PipelineInputError(
            'Neighbors must be a nonempty tuple of distinct positive integers'
        )
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
        or isinstance(workers, bool)
        or not isinstance(workers, int)
        or workers < 1
    ):
        raise PipelineInputError('Batch size and workers must be positive integers')
    source_sha256 = sha256(records_path)
    try:
        records: NDArray[np.void] = np.load(records_path, mmap_mode='r', allow_pickle=False)
    except (ValueError, OSError) as exc:
        raise PipelineInputError(f'Cannot load source records: {records_path}') from exc
    fields = records.dtype.fields
    if (
        records.ndim != 1
        or fields is None
        or not {'x', 'y', 'z'}.issubset(fields)
        or any(
            fields[name][0].shape or fields[name][0].kind not in 'biuf' for name in ('x', 'y', 'z')
        )
    ):
        raise PipelineInputError('Source records must contain scalar numeric x/y/z fields')
    selection_hash = None
    if row_ids is not None:
        if (
            row_ids.ndim != 1
            or row_ids.dtype != np.int64
            or not len(row_ids)
            or row_ids[0] < 0
            or row_ids[-1] >= len(records)
            or np.any(row_ids[1:] <= row_ids[:-1])
        ):
            raise PipelineInputError('Row IDs must be increasing unique int64 source indices')
        row_ids = row_ids.copy()
        digest = hash_bytes()
        for start in range(0, len(row_ids), batch_size):
            digest.update(row_ids[start : start + batch_size].astype('<i8').tobytes())
        selection_hash = digest.hexdigest()
    count = len(records) if row_ids is None else len(row_ids)
    if count >= np.iinfo(np.int32).max or max(neighbors) >= count:
        raise PipelineInputError('Neighbors must be less than N, and N must fit the int32 graph')
    xyz: NDArray[np.float64] = np.empty((count, 3), dtype=np.float64)
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        for axis, name in enumerate(('x', 'y', 'z')):
            xyz[start:end, axis] = (
                records[name][start:end] if row_ids is None else records[name][row_ids[start:end]]
            )
        if not np.isfinite(xyz[start:end]).all():
            raise PipelineInputError('All source XYZ must be finite float64')
    del records
    shape = (count, len(neighbors))
    metadata = {
        'algorithm_version': ALGORITHM_VERSION,
        'source_sha256': source_sha256,
        'neighbors': list(neighbors),
        'shape': list(shape),
    }
    owned = cache_dir / (
        f'{ALGORITHM_VERSION}-{source_sha256}-' + '-'.join(str(k) for k in neighbors)
    )
    if selection_hash is not None:
        metadata['selection_sha256'] = selection_hash
        owned = owned.with_name(owned.name + '-rows-' + selection_hash)
    completion = owned / 'complete.json'
    scores_path = owned / 'scores.npy'
    if completion.exists():
        expected = json.dumps(metadata, indent=2, allow_nan=False) + '\n'
        if completion.read_text(encoding='utf-8') != expected:
            raise PipelineInputError('LOF cache metadata does not match the source and parameters')
        try:
            scores: NDArray[np.float64] = np.load(scores_path, mmap_mode='r', allow_pickle=False)
        except (ValueError, OSError) as exc:
            raise PipelineInputError('Cannot load completed LOF scores') from exc
        if scores.shape != shape or scores.dtype != np.float64:
            raise PipelineInputError('Completed LOF scores have invalid shape or dtype')
        for start in range(0, count, batch_size):
            if not np.isfinite(scores[start : start + batch_size]).all():
                raise PipelineInputError('Completed LOF scores must be finite')
        if sha256(records_path) != source_sha256:
            raise PipelineInputError('Source changed while loading LOF scores')
        return LiveLofResult(scores, neighbors, perf_counter() - started, True, source_sha256)

    owned.mkdir(parents=True, exist_ok=True)
    maxk = max(neighbors)
    kdist = np.empty(shape, dtype=np.float64)
    lrd = np.empty(shape, dtype=np.float64)
    output: np.memmap[tuple[int, ...], np.dtype[np.float64]] = np.lib.format.open_memmap(
        scores_path, mode='w+', dtype=np.float64, shape=shape
    )
    print(f'LOF_STAGE tree {count}', flush=True)
    tree = cKDTree(xyz)
    columns = np.arange(maxk)[None, :]

    def query(start: int, end: int) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Exclude row self even when coincident observations reorder the zero-distance ties."""
        distances: NDArray[np.float64]
        ids: NDArray[np.int64]
        distances, ids = tree.query(xyz[start:end], k=maxk + 1, eps=0, p=2, workers=workers)
        self_mask = ids == np.arange(start, end)[:, None]
        has_self = self_mask.any(axis=1)
        # A missing self can only be displaced by tied coincident points. Drop one
        # zero-distance tie, never the nearest nonzero neighbor.
        if np.any(~has_self & (distances[:, 0] != 0)):
            raise PipelineInputError(
                'Exact neighbor query omitted self without a zero-distance tie'
            )
        drop = self_mask.argmax(axis=1)
        take = columns + (columns >= drop[:, None])
        selected = np.take_along_axis(ids, take, axis=1)
        selected_distances = np.take_along_axis(distances, take, axis=1)
        return selected_distances, selected

    previous = -1
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        selected_distances, _ = query(start, end)
        for column, k in enumerate(neighbors):
            kdist[start:end, column] = selected_distances[:, k - 1]
        previous = _progress('neighbors', end, count, previous)
        del selected_distances

    previous = -1
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        distance, neighbor_ids = query(start, end)
        for column, k in enumerate(neighbors):
            reachability = np.maximum(distance[:, :k], kdist[neighbor_ids[:, :k], column])
            lrd[start:end, column] = 1.0 / (reachability.mean(axis=1) + 1e-10)
            del reachability
        previous = _progress('lrd', end, count, previous)
        del distance

    previous = -1
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        _, neighbor_ids = query(start, end)
        for column, k in enumerate(neighbors):
            ratios = lrd[neighbor_ids[:, :k], column] / lrd[start:end, column, None]
            output[start:end, column] = ratios.mean(axis=1)
        if not np.isfinite(output[start:end]).all():
            raise PipelineInputError('Computed LOF scores must be finite')
        previous = _progress('scores', end, count, previous)
    output.flush()
    # Inspection precision does not affect the float64 LOF computation.
    np.save(owned / 'kdist.npy', kdist.astype(np.float32))
    if sha256(records_path) != source_sha256:
        raise PipelineInputError('Source changed while computing LOF scores')
    write_json(completion, metadata)
    del output
    scores = np.load(scores_path, mmap_mode='r', allow_pickle=False)
    return LiveLofResult(scores, neighbors, perf_counter() - started, False, source_sha256)


@dataclass(init=False)
class _Arguments(argparse.Namespace):
    records: Path
    cache: Path
    k: list[int]
    batch_size: int
    workers: int


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument('--records', type=Path, required=True)
    _ = parser.add_argument('--cache', type=Path, required=True)
    _ = parser.add_argument('--k', type=int, nargs='+', default=[20, 40, 80])
    _ = parser.add_argument('--batch-size', type=int, default=8192)
    _ = parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args(namespace=_Arguments())
    result = compute_lof(
        args.records,
        args.cache,
        neighbors=tuple(args.k),
        batch_size=args.batch_size,
        workers=args.workers,
    )
    print(
        f'LOF_COMPLETE rows={len(result.scores)} cached={result.cached} '
        f'elapsed_seconds={result.elapsed_seconds:.3f}',
        flush=True,
    )


if __name__ == '__main__':
    main()
