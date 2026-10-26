"""Audit source-backed voxel occupancy and its closed canonical boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .models import PipelineInputError
from .simulation_obstacles import ObstacleConfig, voxel_surface


@dataclass(frozen=True, slots=True)
class ObstacleAudit:
    positions: NDArray[np.float32]
    faces: NDArray[np.int32]
    volume_m3: float
    edge_contacts: int


def verify_obstacles(
    output: Path, points: NDArray[np.float32], config: ObstacleConfig
) -> ObstacleAudit:
    """Validate measured ownership before checking the exact occupied-volume shell."""
    folder = output / 'simulation-models'
    with np.load(folder / 'source-partition.npz') as archive:
        labels = archive['labels']
    with np.load(folder / 'obstacles.npz') as archive:
        keys, ids, groups, counts = (
            archive['keys'],
            archive['ids'],
            archive['groups'],
            archive['counts'],
        )
    if not (
        np.array_equal(np.sort(ids), np.flatnonzero(labels == 3))
        and np.array_equal(np.bincount(groups, minlength=len(keys)), counts)
        and np.array_equal(
            np.floor(points[ids].astype(np.float64) / config.cell).astype(np.int64),
            keys[groups],
        )
        and (counts >= 1).all()
    ):
        raise PipelineInputError('Obstacle proxy occupancy or original record ownership changed')
    positions, faces, _ = voxel_surface(keys, config.cell)
    triangles = positions[faces].astype(np.float64)
    volume = float(
        np.einsum('ij,ij->i', triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum()
        / 6
    )
    if not np.isclose(volume, len(keys) * config.cell**3, rtol=1e-4):
        raise PipelineInputError('Obstacle boundary is not the closed occupied volume')
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    _, edge_counts = np.unique(edges, axis=0, return_counts=True)
    if (edge_counts % 2).any():
        raise PipelineInputError('Obstacle boundary has open edges')
    return ObstacleAudit(positions, faces, volume, int((edge_counts > 2).sum()))
