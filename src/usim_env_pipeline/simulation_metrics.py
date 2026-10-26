"""Geometric metrics using one common origin for independently published mesh tiles."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def signed_volume(
    positions: NDArray[np.float32] | NDArray[np.float64], faces: NDArray[np.int32]
) -> float:
    """Keep the global origin so open tile contributions sum to the closed map volume."""
    triangles = positions[faces].astype(np.float64)
    return float(
        np.einsum('ij,ij->i', triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum()
        / 6
    )
