"""Independent source ownership and local-density audit for removed noise cells."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .models import PipelineInputError
from .simulation_obstacles import ObstacleConfig


def verify_cleanup(output: Path, points: NDArray[np.float32], config: ObstacleConfig) -> int:
    """Rebuild candidates from source IDs and check every retained/removed decision."""
    from scipy.spatial import cKDTree

    folder = output / 'simulation-models'
    with np.load(folder / 'source-partition.npz') as archive:
        labels = archive['labels']
    ids = np.flatnonzero(labels >= 3)
    expected, groups = np.unique(
        np.floor(points[ids].astype(np.float64) / config.cell).astype(np.int64),
        axis=0,
        return_inverse=True,
    )
    with np.load(folder / 'cleanup.npz') as archive:
        keys, counts, keep = archive['keys'], archive['support_counts'], archive['keep']
    if (
        not np.array_equal(keys, expected)
        or counts.shape != keep.shape
        or keep.shape != (len(keys),)
        or keep.dtype != np.bool_
        or not np.array_equal(keep[groups], labels[ids] != 5)
    ):
        raise PipelineInputError('Noise cleanup source ownership or candidate cells changed')
    if len(keys):
        tree = cKDTree(expected)
        radius = config.cleanup_radius / config.cell + 1e-10
        for start in range(0, len(keys), 8192):
            stop = min(start + 8192, len(keys))
            actual = tree.query_ball_point(
                expected[start:stop], radius, return_length=True, workers=4
            )
            if not (
                np.array_equal(counts[start:stop], actual)
                and np.array_equal(keep[start:stop], actual >= config.cleanup_min_cells)
            ):
                raise PipelineInputError('Noise cleanup local-density evidence changed')
    return int((~keep).sum())
