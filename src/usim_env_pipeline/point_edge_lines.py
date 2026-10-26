"""Cached fitted edge segments with finite original-observation witnesses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256 as hash_bytes
import json
from pathlib import Path
from time import perf_counter
from typing import Final

import numpy as np
from numpy.typing import NDArray

from .edge_line_geometry import EdgeLineConfig, extract_edge_lines
from .models import PipelineInputError
from .pcd import sha256
from .storage import write_json

ALGORITHM_VERSION: Final = 'observed-face-edge-segments-v4'


@dataclass(frozen=True, slots=True)
class EdgeLineResult:
    source_ids: NDArray[np.int64]
    xyz: NDArray[np.float64]
    segments: NDArray[np.float64]
    kinds: NDArray[np.uint8]
    support_counts: NDArray[np.int64]
    coverage: NDArray[np.float32]
    max_gaps: NDArray[np.float32]
    equations: NDArray[np.float64]
    witness_offsets: NDArray[np.int64]
    witness_ids: NDArray[np.int64]
    bin_origins: NDArray[np.float64]
    config: EdgeLineConfig
    source_sha256: str
    selected_ids_sha256: str
    elapsed_seconds: float
    cached: bool
    cache_dir: Path


def compute_edge_lines(
    records_path: Path,
    source_ids_path: Path,
    cache_dir: Path,
    *,
    boundary_flags_path: Path | None = None,
    config: EdgeLineConfig = EdgeLineConfig(),
) -> EdgeLineResult:
    """Fit lines from unchanged LOF XYZ; flags only propose boundary candidates."""
    started = perf_counter()
    source_hash, ids_hash = sha256(records_path), sha256(source_ids_path)
    flags_hash = sha256(boundary_flags_path) if boundary_flags_path is not None else None
    try:
        raw = np.load(records_path, mmap_mode='r', allow_pickle=False)
        ids = np.load(source_ids_path, allow_pickle=False)
        flags = (
            np.load(boundary_flags_path, allow_pickle=False)
            if boundary_flags_path is not None
            else np.ones(len(ids), dtype=np.uint8)
        )
    except (OSError, ValueError) as error:
        raise PipelineInputError(
            'Cannot load original LOF records or boundary proposals'
        ) from error
    if (
        raw.ndim != 1
        or raw.dtype.fields is None
        or not {'x', 'y', 'z'}.issubset(raw.dtype.fields)
        or any(raw.dtype[name].shape or raw.dtype[name].kind not in 'biuf' for name in 'xyz')
        or ids.ndim != 1
        or ids.dtype != np.int64
        or not len(ids)
        or ids[0] < 0
        or ids[-1] >= len(raw)
        or np.any(ids[1:] <= ids[:-1])
        or flags.shape != ids.shape
        or flags.dtype != np.uint8
        or np.any(flags > 3)
    ):
        raise PipelineInputError('Increasing original IDs and matching boundary flags required')
    xyz = np.column_stack([raw[name][ids] for name in 'xyz']).astype(np.float64)
    del raw
    if not np.isfinite(xyz).all():
        raise PipelineInputError('Original LOF XYZ is not finite')
    metadata = {
        'algorithm': ALGORITHM_VERSION,
        'source_sha256': source_hash,
        'selected_ids_sha256': ids_hash,
        'boundary_flags_sha256': flags_hash,
        'input_records': len(ids),
        'island_filter_applied': False,
        'configuration': asdict(config),
    }
    identity = hash_bytes(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    folder = cache_dir / f'{ALGORITHM_VERSION}-{identity}'
    complete = folder / 'complete.json'
    cached = complete.exists()
    if not cached:
        folder.mkdir(parents=True, exist_ok=True)
        edges = list(extract_edge_lines(xyz, (flags & 1) != 0, config))
        witnesses = []
        offsets = np.zeros((len(edges), 2, 2), dtype=np.int64)
        counts = np.zeros((len(edges), 2), dtype=np.int64)
        cursor = 0
        for number, edge in enumerate(edges):
            for face, rows in enumerate(edge.line.witnesses):
                witnesses.append(ids[rows])
                offsets[number, face] = [cursor, cursor + len(rows)]
                counts[number, face] = len(rows)
                cursor += len(rows)
            if edge.kind == 1:
                offsets[number, 1] = [cursor, cursor]
        np.save(
            folder / 'segments.npy',
            np.array([e.line.endpoints for e in edges], dtype=np.float64).reshape(-1, 2, 3),
        )
        np.save(folder / 'kinds.npy', np.array([e.kind for e in edges], dtype=np.uint8))
        np.save(folder / 'support_counts.npy', counts)
        np.save(
            folder / 'coverage.npy', np.array([e.line.coverage for e in edges], dtype=np.float32)
        )
        np.save(
            folder / 'max_gaps.npy',
            np.array([e.line.max_unsupported_gap for e in edges], dtype=np.float32),
        )
        np.save(
            folder / 'equations.npy',
            np.array([e.equations for e in edges], dtype=np.float64).reshape(-1, 2, 4),
        )
        np.save(folder / 'witness_offsets.npy', offsets)
        np.save(
            folder / 'witness_ids.npy',
            np.concatenate(witnesses) if witnesses else np.empty(0, dtype=np.int64),
        )
        origins = []
        for edge in edges:
            if edge.line.bin_origin is None:
                raise PipelineInputError('Completed line has no support-bin origin')
            origins.append(edge.line.bin_origin)
        np.save(folder / 'bin_origins.npy', np.array(origins, dtype=np.float64).reshape(-1, 3))
        if sha256(records_path) != source_hash or sha256(source_ids_path) != ids_hash:
            raise PipelineInputError('Original input changed during edge-line fitting')
        if boundary_flags_path is not None and sha256(boundary_flags_path) != flags_hash:
            raise PipelineInputError('Boundary proposals changed during edge-line fitting')
        write_json(complete, metadata)
    elif complete.read_text(encoding='utf-8') != json.dumps(metadata, indent=2) + '\n':
        raise PipelineInputError('Edge-line cache metadata disagrees with input')
    arrays = [
        np.load(folder / f'{name}.npy', mmap_mode='r', allow_pickle=False)
        for name in (
            'segments',
            'kinds',
            'support_counts',
            'coverage',
            'max_gaps',
            'equations',
            'witness_offsets',
            'witness_ids',
            'bin_origins',
        )
    ]
    segments, kinds, counts, coverage, gaps, equations, offsets, witnesses, bin_origins = arrays
    n = len(kinds)
    dtypes = (
        np.float64,
        np.uint8,
        np.int64,
        np.float32,
        np.float32,
        np.float64,
        np.int64,
        np.int64,
        np.float64,
    )
    if (
        segments.shape != (n, 2, 3)
        or kinds.shape != (n,)
        or counts.shape != (n, 2)
        or coverage.shape != (n,)
        or gaps.shape != (n,)
        or equations.shape != (n, 2, 4)
        or offsets.shape != (n, 2, 2)
        or witnesses.ndim != 1
        or bin_origins.shape != (n, 3)
        or any(value.dtype != dtype for value, dtype in zip(arrays, dtypes))
        or any(not np.isfinite(value).all() for value in arrays)
        or np.any((kinds != 1) & (kinds != 2))
        or np.any(coverage < config.support.min_coverage - 1e-7)
        or np.any(coverage > 1)
        or np.any(gaps < 0)
        or np.any(gaps > config.support.max_gap + 1e-7)
        or np.any(offsets < 0)
        or np.any(offsets > len(witnesses))
        or not np.array_equal(offsets[:, :, 1] - offsets[:, :, 0], counts)
        or np.any(counts[:, 0] < config.support.min_points)
        or np.any(counts[kinds == 2, 1] < config.support.min_points)
        or not np.isin(witnesses, ids).all()
    ):
        raise PipelineInputError('Completed edge-line support arrays invalid')
    if sha256(records_path) != source_hash or sha256(source_ids_path) != ids_hash:
        raise PipelineInputError('Original input changed while loading supported lines')
    ids.setflags(write=False)
    xyz.setflags(write=False)
    return EdgeLineResult(
        ids,
        xyz,
        segments,
        kinds,
        counts,
        coverage,
        gaps,
        equations,
        offsets,
        witnesses,
        bin_origins,
        config,
        source_hash,
        ids_hash,
        perf_counter() - started,
        cached,
        folder,
    )
