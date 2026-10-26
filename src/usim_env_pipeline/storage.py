"""Archival hashing, immutable-source publication and regional record partitioning."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import TypeVar

import numpy as np

from .cloud import Cloud, read_cloud
from .models import PipelineInputError, Source
from .pcd import require_xyzinormal, sha256, xyz

T = TypeVar('T')


def write_json(path: Path, value: T) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def tile_name(key: tuple[int, int]) -> str:
    return f't_{key[0] + 10000}_{key[1] + 10000}'.replace('-', 'm')


def archive_source(source: Source, output: Path) -> Cloud:
    """Verify before and after copying; never reuse a stale file with the same name."""
    original_hash = sha256(source.path)
    if source.expected_sha256 is not None and original_hash != source.expected_sha256:
        raise PipelineInputError(f'Original SHA256 mismatch: {source.path}')
    archive = output / 'source' / source.path.name
    if archive.exists():
        raise PipelineInputError(f'Duplicate source basename: {source.path.name}')
    shutil.copyfile(source.path, archive)
    if sha256(archive) != original_hash:
        raise PipelineInputError('Archive hash disagrees with source')
    cloud = read_cloud(archive)
    if source.strict_xyzinormal:
        require_xyzinormal(cloud.records, cloud.metadata)
    cloud.metadata.update(
        {
            'sha256': original_hash,
            'path': f'source/{source.path.name}',
            'official_url': source.url,
            'license': source.license,
            'attribution': source.attribution,
        }
    )
    return cloud


def partition(cloud: Cloud, output: Path, size: float) -> frozenset[tuple[int, int]]:
    positions = xyz(cloud.records)
    floating_keys = np.floor(positions[:, :2].astype(np.float64) / size)
    if (np.abs(floating_keys) > np.iinfo(np.int32).max).any():
        raise PipelineInputError('XY extent/tile size exceeds supported regional key range')
    keys = floating_keys.astype(np.int32)
    unique, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind='stable')
    occupied = frozenset((int(key[0]), int(key[1])) for key in unique)
    start = 0
    for key, count in zip(unique, counts):
        ids = order[start : start + count].astype(np.int64)
        np.savez_compressed(
            output / 'records' / f'{tile_name((int(key[0]), int(key[1])))}.npz',
            records=cloud.records[ids],
            ids=ids,
        )
        start += count
    return occupied
