"""Shared measured-edge values and extraction settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .edge_line_support import LineSupport, SupportedLine
from .models import PipelineInputError


@dataclass(frozen=True, slots=True)
class EdgeLineConfig:
    tile_size: float = 8.0
    # Ownership only: moving it must not change the source-anchored fitting grid.
    tile_origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    halo: float = 0.6
    voxel: float = 0.1
    plane_tolerance: float = 0.05
    min_plane_points: int = 96
    min_plane_area: float = 1.0
    max_planes: int = 8
    iterations: int = 350
    min_dihedral: float = 25.0
    normal_advantage: float = 0.01
    min_normal_coherence: float = 0.45
    surface_radius: float = 0.6
    normal_probes: int = 128
    max_refit_angle: float = 2.0
    seed: int = 7
    support: LineSupport = field(default_factory=lambda: LineSupport(min_length=0.75))

    def __post_init__(self) -> None:
        if (
            any(
                type(v) not in (int, float) or not isfinite(v) or v <= 0
                for v in (
                    self.tile_size,
                    self.halo,
                    self.voxel,
                    self.plane_tolerance,
                    self.min_plane_area,
                    self.min_dihedral,
                    self.normal_advantage,
                    self.min_normal_coherence,
                    self.surface_radius,
                    self.max_refit_angle,
                )
            )
            or type(self.tile_origin) is not tuple
            or len(self.tile_origin) != 3
            or any(
                type(value) not in (int, float) or not isfinite(value)
                for value in self.tile_origin
            )
            or not 0.35 <= self.halo < self.tile_size
            or self.min_dihedral >= 90
            or self.max_refit_angle >= 90
            or not 0 < self.min_normal_coherence <= 1
            or any(
                type(v) is not int or v < 1
                for v in (self.max_planes, self.iterations, self.normal_probes)
            )
            or type(self.min_plane_points) is not int
            or self.min_plane_points < 16
            or type(self.seed) is not int
            or self.seed < 0
        ):
            raise PipelineInputError('Invalid observed-plane edge-line configuration')


@dataclass(frozen=True, slots=True)
class MeasuredEdge:
    line: SupportedLine
    kind: Literal[1, 2]
    equations: NDArray[np.float64]
    # Complete original parent-face observations, when supplied by discovery.
    face_members: tuple[NDArray[np.int64], ...] = ()
