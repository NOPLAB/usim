# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Observation conversion for Genesis visualization cameras."""

from typing import Any

import numpy as np
from numpy.typing import NDArray


def frame_robot(camera: Any, bounds: NDArray[np.float32 | np.float64]) -> None:
    """Frame external models from native bounds rather than CRANE workspace coordinates."""
    lower = bounds[:, 0].min(axis=0)
    upper = bounds[:, 1].max(axis=0)
    center = (lower + upper) / 2
    radius = max(float(np.linalg.norm(upper - lower)) / 2, 0.05)
    distance = 1.4 * radius / np.sin(np.deg2rad(69 / 2))
    direction = np.array([1.0, -1.0, 0.8])
    camera.set_pose(pos=center + direction / np.linalg.norm(direction) * distance, lookat=center)


def render_images(
    camera: Any, n_envs: int, with_depth: bool
) -> tuple[NDArray[np.uint8], NDArray[np.float32] | None]:
    """Return separate environment frames, never copies of one merged scene image."""
    rgb, depth, _, _ = camera.render(rgb=True, depth=with_depth)
    rgb = np.asarray(rgb)
    if rgb.ndim == 3 and n_envs == 1:
        rgb = rgb[None, ...]
    if rgb.ndim != 4 or rgb.shape[0] != n_envs:
        raise RuntimeError('Genesis camera did not render every environment separately')
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    if depth is not None:
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 2 and n_envs == 1:
            depth = depth[None, ...]
        if depth.ndim != 3 or depth.shape[0] != n_envs:
            raise RuntimeError('Genesis depth batch does not match environment count')
    return rgb, depth
