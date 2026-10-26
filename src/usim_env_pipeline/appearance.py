"""Original procedural colors on measured surfaces; never move or fill geometry."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def surface_colors(
    positions: NDArray[np.float32],
    normals: NDArray[np.float64],
    heights: NDArray[np.float64],
) -> NDArray[np.float32]:
    """Use a matte ground/wall/roof palette with explicitly synthetic window bands."""
    colors = np.tile(np.array([0.52, 0.54, 0.55], dtype=np.float32), (len(positions), 1))
    horizontal = np.abs(normals[:, 2]) >= 0.7
    ground = horizontal & (heights < 0.45)
    roof = horizontal & ~ground
    wall = (np.abs(normals[:, 2]) < 0.35) & (heights > 1.0)
    colors[ground] = (0.30, 0.32, 0.33)
    colors[roof] = (0.65, 0.66, 0.64)
    colors[wall] = (0.78, 0.76, 0.68)
    along = np.where(
        np.abs(normals[:, 0]) > np.abs(normals[:, 1]), positions[:, 1], positions[:, 0]
    )
    windows = wall & (heights % 3 > 0.75) & (heights % 3 < 2.3) & (along % 2.0 < 1.5)
    colors[windows] = (0.16, 0.28, 0.37)
    return colors
