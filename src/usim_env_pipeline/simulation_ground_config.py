"""Bounds for fitting a measured single-valued simulation terrain."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .models import PipelineInputError


@dataclass(frozen=True, slots=True)
class GroundConfig:
    cell: float = 0.5
    radius: float = 0.75
    nearest_support: float = 0.6
    max_error: float = 0.12
    max_slope: float = 0.35
    neighbors: int = 48
    min_neighbors: int = 6
    continuity_tolerance: float = 0.04

    def __post_init__(self) -> None:
        for value in (
            self.cell,
            self.radius,
            self.nearest_support,
            self.max_error,
            self.max_slope,
            self.continuity_tolerance,
        ):
            if not math.isfinite(value) or value <= 0:
                raise PipelineInputError('Ground thresholds must be finite and positive')
        if self.nearest_support > self.radius:
            raise PipelineInputError('Nearest ground support must not exceed the search radius')
        if (
            isinstance(self.neighbors, bool)
            or not isinstance(self.neighbors, int)
            or isinstance(self.min_neighbors, bool)
            or not isinstance(self.min_neighbors, int)
            or not 3 <= self.min_neighbors <= self.neighbors <= 64
        ):
            raise PipelineInputError('Require 3 <= min_neighbors <= neighbors <= 64')
