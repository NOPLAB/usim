"""Measured facade support and closed building proxy contracts, in meters/Z-up."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

from .models import PipelineInputError


@dataclass(frozen=True, slots=True)
class BuildingConfig:
    cell: float = 0.4
    gap: float = 1.0
    wall_thickness: float = 0.3
    minimum_height: float = 2.5
    minimum_width: float = 3.0
    minimum_area: float = 12.0
    minimum_coverage: float = 0.25
    minimum_points: int = 100
    vertical_angle: float = 10.0
    support_distance: float = 0.15
    corner_gap: float = 0.75
    corner_angle: float = 5.0
    height_overlap: float = 0.7

    def __post_init__(self) -> None:
        positive = (
            self.cell,
            self.wall_thickness,
            self.minimum_height,
            self.minimum_width,
            self.minimum_area,
            self.support_distance,
            self.corner_gap,
        )
        if (
            not all(np.isfinite(value) and value > 0 for value in positive)
            or not np.isfinite(self.gap)
            or self.gap < 0
            or self.minimum_points < 3
            or not 0 < self.minimum_coverage <= 1
            or not 0 < self.height_overlap <= 1
            or not 0 < self.vertical_angle < 45
            or not 0 < self.corner_angle < 45
        ):
            raise PipelineInputError('Invalid building geometry configuration')


@dataclass(frozen=True, slots=True)
class Facade:
    source_planes: tuple[int, ...]
    normal: NDArray[np.float64]
    offset: float
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    support_ids: NDArray[np.int64]
    coverage: float

    @property
    def tangent(self) -> NDArray[np.float64]:
        return np.array([self.normal[1], -self.normal[0], 0.0])


@dataclass(frozen=True, slots=True)
class Cuboid:
    center: NDArray[np.float64]
    size: NDArray[np.float64]
    axes: NDArray[np.float64]
    facade_indices: tuple[int, ...]
    kind: Literal['building', 'facade']


class BuildingReport(TypedDict):
    source_sha256: str
    source_records: int
    input_planes: int
    facades: int
    building_cuboids: int
    facade_cuboids: int
    supported_records: int
    world: str
    authority: str
    ground_modified: bool
    residual_triangles: int
    replaced_triangles: int
