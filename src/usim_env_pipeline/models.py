"""Public reconstruction inputs and quantitative output contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Literal, NotRequired, TypedDict


class PipelineInputError(ValueError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class Source:
    path: Path
    key: str = 'primary'
    license: str = 'unspecified'
    attribution: str = ''
    url: str = ''
    catalog: str = ''
    expected_sha256: str | None = None
    strict_xyzinormal: bool = False


@dataclass(frozen=True, slots=True)
class Support:
    radii: tuple[float, ...] = (0.12, 0.20, 0.35)
    max_edge: float = 0.65
    min_face_normal_dot: float = 0.35
    min_pair_normal_dot: float = 0.0
    max_edge_normal_dot: float = 0.65
    midpoint_distance: float = 0.16
    centroid_distance: float = 0.18
    min_double_area: float = 1e-10

    def __post_init__(self) -> None:
        distances = (*self.radii, self.max_edge, self.midpoint_distance, self.centroid_distance)
        if not self.radii or any(not math.isfinite(v) or v <= 0 for v in distances):
            raise PipelineInputError('Support distances must be finite and positive')
        if tuple(sorted(set(self.radii))) != self.radii:
            raise PipelineInputError('Pivot radii must be strictly increasing')
        if not math.isfinite(self.min_double_area) or self.min_double_area <= 0:
            raise PipelineInputError('Minimum double area must be finite and positive')
        for v in (self.min_face_normal_dot, self.min_pair_normal_dot, self.max_edge_normal_dot):
            if not math.isfinite(v) or not 0 <= v <= 1:
                raise PipelineInputError('Normal support thresholds must be in [0, 1]')


@dataclass(frozen=True, slots=True)
class MapConfig:
    tile_size: float = 30.0
    halo: float = 1.5
    workers: int = 4
    normal_policy: Literal['auto', 'supplied'] = 'auto'
    frame: str = 'source XYZ'
    support: Support = field(default_factory=Support)
    references: tuple[Source, ...] = ()
    resume: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.tile_size) or not math.isfinite(self.halo):
            raise PipelineInputError('Tile size and halo must be finite')
        if not 2 * max(self.support.radii) <= self.halo < self.tile_size:
            raise PipelineInputError('Require 2*largest pivot radius <= halo < tile size')
        if self.workers < 1:
            raise PipelineInputError('Workers must be positive')
        if self.normal_policy not in ('auto', 'supplied'):
            raise PipelineInputError('Unknown normal policy')


class SourceMetadata(TypedDict):
    points: int
    bytes: int
    fields: list[str]
    dtype: str
    data_offset: NotRequired[int]
    trailing_zero_bytes: NotRequired[int]
    header: NotRequired[str]
    sha256: NotRequired[str]
    path: NotRequired[str]
    official_url: NotRequired[str]
    license: NotRequired[str]
    attribution: NotRequired[str]
    role: NotRequired[str]


class Counts(TypedDict):
    records: int
    normal_valid: int
    normal_invalid: int
    triangles: int
    mesh_vertices: int
    core_meshed_records: int
    unmeshed_records: int
    horizontal_triangles: int
    other_triangles: int
    open_boundary_edges: int
    nonmanifold_edges: int
    rejected_support: int
    estimated_normal_records: int


class RegionReport(Counts):
    name: str
    key: list[int]
    halo_records: int
    candidate_owned: int
    min: list[float]
    max: list[float]
    mesh_min: NotRequired[list[float]]
    mesh_max: NotRequired[list[float]]


class Summary(Counts):
    occupied_tiles: int
    meshed_tiles: int
    unmeshed_tiles: list[str]


class Bounds(TypedDict):
    path: str
    min: list[float]
    max: list[float]


class Payload(Bounds):
    payload: str
