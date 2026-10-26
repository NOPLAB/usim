"""Shared measured-plane, repair and Gaussian-join contracts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class PlaneConfig:
    voxel: float = 0.12
    tolerance: float = 0.08
    cell: float = 0.15
    bridge_radius: float = 0.45
    hole_area: float = 256.0
    min_support: int = 128
    min_area: float = 3.0
    max_planes: int = 12
    iterations: int = 350
    seed: int = 7
    distance_sigma: float = 0.04
    gap_sigma: float = 1.0
    angle_sigma: float = 3.0
    join_cost: float = 4.5
    workers: int = 8


@dataclass(frozen=True, slots=True)
class Plane:
    normal: NDArray[np.float64]
    offset: float
    ids: NDArray[np.int64]
    links: NDArray[np.int64] | None = None
    link_models: NDArray[np.int32] | None = None


@dataclass(frozen=True, slots=True)
class Patch:
    plane: Plane
    u: NDArray[np.float64]
    v: NDArray[np.float64]
    lower: NDArray[np.int64]
    mask: NDArray[np.bool_]
    observed: NDArray[np.bool_]
    protected: NDArray[np.bool_]
    enclosed: NDArray[np.bool_]
    bridged: NDArray[np.bool_]
    positions: NDArray[np.float32]
    faces: NDArray[np.int32]


def plane_basis(normal: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    reference = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(reference, normal)
    u /= np.linalg.norm(u)
    return u, np.cross(normal, u)
