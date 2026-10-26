"""CAD reconstruction configuration and source-supported board provenance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from OCP.TopoDS import TopoDS_Shape


@dataclass(frozen=True, slots=True)
class CadConfig:
    hole_area: float = 1.0
    boundary_tolerance: float = 0.0
    minimum_face_area: float = 1.0
    parallel_distance: float = 0.15
    parallel_angle: float = 3.0
    containment: float = 0.995
    deflection: float = 0.05
    workers: int = 8


class CadArchive(TypedDict):
    equation: NDArray[np.float64]
    u: NDArray[np.float64]
    v: NDArray[np.float64]
    lower: NDArray[np.int64]
    mask: NDArray[np.bool_]


class CadShapeStats(TypedDict):
    area_m2: float
    face_count: int
    hole_count: int
    valid: bool


class CadJoinReport(TypedDict):
    smaller: int
    larger: int
    kind: Literal['contained', 'fused']
    max_projection_m: float
    overlap_m2: float
    projected_m2: float


@dataclass(frozen=True, slots=True)
class CadBoard:
    shape: TopoDS_Shape
    equation: NDArray[np.float64]
    source_planes: tuple[int, ...]
    support_ids: NDArray[np.int64]


class CadBoardReport(TypedDict):
    name: str
    source_planes: list[int]
    support_records: int
    area: float
    faces: int
    holes: int
    triangles: int


class CadReport(TypedDict):
    kernel: str
    authority: str
    step_units: str
    brep_units: str
    source_sha256: str
    source_records: int
    input_planes: int
    output_boards: int
    output_faces: int
    output_holes: int
    small_holes_removed: int
    small_hole_area: float
    contained_boards_removed: int
    overlapping_boards_fused: int
    omitted_small_face_area: float
    surface_area: float
    display_triangles: int
    world: str
