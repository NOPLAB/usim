"""Single-pass statistical outliers from global, source-order point neighborhoods."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

import numpy as np
from numpy.typing import NDArray

from .models import PipelineInputError


@dataclass(frozen=True, slots=True)
class SORConfig:
    neighbors: int = 20
    std_ratio: float = 3.0
    batch_size: int = 32768
    workers: int = 4

    def __post_init__(self) -> None:
        if self.neighbors < 1 or self.batch_size < 1 or self.workers < 1:
            raise PipelineInputError('SOR neighbors, batch size and workers must be positive')
        if not math.isfinite(self.std_ratio) or self.std_ratio <= 0:
            raise PipelineInputError(
                'SOR standard deviation multiplier must be finite and positive'
            )


@dataclass(frozen=True, slots=True)
class SORResult:
    mean_distances: NDArray[np.float64]
    keep: NDArray[np.bool_]
    global_mean: float
    global_std: float
    threshold: float


DEFAULT_SOR: Final = SORConfig()


def statistical_outliers(
    points: NDArray[np.float64], config: SORConfig = DEFAULT_SOR
) -> SORResult:
    """Compare k-other-record mean distances with the global mean plus sample deviation.

    Query the complete original cloud, not individual tiles or a progressively
    eroded selection. Coincident records are distinct observations, as in standard
    record-level SOR. Dropping one zero-distance neighbor excludes the query
    record even when coincident records tie with it. Float64 distances retain the
    precision of supplied coordinates; no normals or surface model are inferred.
    """
    from scipy.spatial import KDTree

    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise PipelineInputError('SOR requires finite XYZ coordinates')
    if len(points) <= config.neighbors:
        raise PipelineInputError('SOR requires more records than its neighbor count')
    positions = np.asarray(points, dtype=np.float64)
    # KDTree is SciPy's public Python wrapper over the same compiled cKDTree engine.
    tree = KDTree(positions, leafsize=16)
    means = np.empty(len(positions), dtype=np.float64)
    for start in range(0, len(positions), config.batch_size):
        stop = min(start + config.batch_size, len(positions))
        distances, _ = tree.query(
            positions[start:stop], k=config.neighbors + 1, workers=config.workers
        )
        means[start:stop] = distances[:, 1:].mean(axis=1)
    mean = float(np.mean(means))
    deviation = float(np.std(means, ddof=1))
    threshold = mean + config.std_ratio * deviation
    return SORResult(means, means <= threshold, mean, deviation, threshold)
