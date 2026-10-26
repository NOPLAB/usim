"""Analytic planar OpenCASCADE topology. Coordinates and areas are meters and m²."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.BRepTools import BRepTools, BRepTools_WireExplorer
from OCP.GeomAbs import GeomAbs_Plane
from OCP.GProp import GProp_GProps
from OCP.ShapeFix import ShapeFix_Shape
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.TopAbs import TopAbs_FACE, TopAbs_FORWARD, TopAbs_REVERSED, TopAbs_WIRE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Compound, TopoDS_Face, TopoDS_Shape, TopoDS_Wire
from OCP.gp import gp_Dir, gp_Pln, gp_Pnt

from .cad_types import CadShapeStats
from .models import PipelineInputError

ShapeT = TypeVar('ShapeT', bound=TopoDS_Shape)


def faces(shape: TopoDS_Shape) -> list[TopoDS_Face]:
    """Return all topological faces, including a face passed as the shape."""
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    result = []
    while explorer.More():
        result.append(TopoDS.Face_s(explorer.Current()))
        explorer.Next()
    return result


def validate(shape: ShapeT) -> ShapeT:
    """Reject null, empty, invalid or non-planar B-Reps; return the valid shape."""
    if shape.IsNull() or not BRepCheck_Analyzer(shape).IsValid() or not faces(shape):
        raise PipelineInputError('Invalid or empty CAD face topology')
    for face in faces(shape):
        if BRepAdaptor_Surface(face).GetType() != GeomAbs_Plane:
            raise PipelineInputError('CAD presentation requires analytic planar faces')
    return shape


def surface_area(shape: TopoDS_Shape) -> float:
    """Exact B-Rep surface area in square meters."""
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, props)
    return props.Mass()


def face_normal(face: TopoDS_Face) -> NDArray[np.float64]:
    """Oriented world-space unit normal."""
    direction = BRepAdaptor_Surface(face).Plane().Axis().Direction()
    normal = np.array([direction.X(), direction.Y(), direction.Z()])
    return -normal if face.Orientation() == TopAbs_REVERSED else normal


def _wire_points(wire: TopoDS_Wire, face: TopoDS_Face) -> NDArray[np.float64]:
    explorer = BRepTools_WireExplorer(wire, face)
    points = []
    while explorer.More():
        point = BRep_Tool.Pnt_s(explorer.CurrentVertex())
        points.append((point.X(), point.Y(), point.Z()))
        explorer.Next()
    return np.asarray(points, dtype=np.float64)


def _wires(face: TopoDS_Face) -> tuple[TopoDS_Wire, list[TopoDS_Wire]]:
    outer = BRepTools.OuterWire_s(face)
    inner = []
    explorer = TopExp_Explorer(face, TopAbs_WIRE)
    while explorer.More():
        wire = TopoDS.Wire_s(explorer.Current())
        if not wire.IsSame(outer):
            inner.append(wire)
        explorer.Next()
    return outer, inner


def face_loops(face: TopoDS_Face) -> list[NDArray[np.float64]]:
    """Return implicit-closure Nx3 loops, outer CCW then inner CW about face normal."""
    outer, inner = _wires(face)
    normal = face_normal(face)
    loops = []
    for index, wire in enumerate([outer, *inner]):
        points = _wire_points(wire, face)
        signed = np.dot(np.cross(points, np.roll(points, -1, axis=0)).sum(axis=0), normal)
        if (signed > 0) != (index == 0):
            points = points[::-1].copy()
        loops.append(points)
    return loops


def _points(loop: NDArray[np.float64]) -> NDArray[np.float64]:
    points = np.asarray(loop, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise PipelineInputError('Polygon loops must contain finite world-space Nx3 points')
    if len(points) > 1 and np.array_equal(points[0], points[-1]):
        points = points[:-1]
    if len(points) < 3 or np.any(
        np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1) < 1e-9
    ):
        raise PipelineInputError('Polygon loops need three distinct non-coincident vertices')
    return points


def _polygon(points: NDArray[np.float64], normal: NDArray[np.float64], outer: bool) -> TopoDS_Wire:
    signed = np.dot(np.cross(points, np.roll(points, -1, axis=0)).sum(axis=0), normal)
    if abs(signed) < 1e-12:
        raise PipelineInputError('Degenerate polygon loop')
    if (signed > 0) != outer:
        points = points[::-1]
    maker = BRepBuilderAPI_MakePolygon()
    for point in points:
        maker.Add(gp_Pnt(*point))
    maker.Close()
    if not maker.IsDone():
        raise PipelineInputError('CAD polygon wire construction failed')
    return maker.Wire()


def make_planar_face(
    outer: NDArray[np.float64],
    holes: Sequence[NDArray[np.float64]] = (),
    *,
    normal: NDArray[np.float64] | None = None,
) -> TopoDS_Face:
    """Build straight-edge analytic face; normalize winding relative to normal."""
    points = _points(outer)
    if normal is None:
        normal = np.cross(points - points[0], np.roll(points, -1, axis=0) - points[0]).sum(0)
    normal = np.asarray(normal, dtype=np.float64)
    if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-12:
        raise PipelineInputError('Plane normal must be finite and nonzero')
    normal = normal / np.linalg.norm(normal)
    loops = [points, *[_points(hole) for hole in holes]]
    if any(np.max(np.abs((loop - points[0]) @ normal)) > 1e-7 for loop in loops):
        raise PipelineInputError('Polygon loops are not coplanar')
    plane = gp_Pln(gp_Pnt(*points[0]), gp_Dir(*normal))
    maker = BRepBuilderAPI_MakeFace(plane, _polygon(points, normal, True), True)
    for loop in loops[1:]:
        maker.Add(_polygon(loop, normal, False))
    if not maker.IsDone():
        raise PipelineInputError('CAD planar face construction failed')
    return validate(maker.Face())


def compound(shapes: Iterable[TopoDS_Shape]) -> TopoDS_Shape:
    """Collect shapes without changing their topology or orientation."""
    result = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(result)
    for shape in shapes:
        builder.Add(result, validate(shape))
    return validate(result)


def make_planar_faces(
    loops: Sequence[NDArray[np.float64]],
    normal: NDArray[np.float64],
    *,
    hole_threshold_m2: float = 0.0,
) -> list[TopoDS_Face]:
    """Remove tiny CAD wires before validation, then assign surviving hole wires."""
    if not np.isfinite(hole_threshold_m2) or hole_threshold_m2 < 0:
        raise PipelineInputError('Hole threshold must be a finite nonnegative square-meter area')
    loops = [_points(loop) for loop in loops]
    normal = np.asarray(normal, dtype=float)
    outer, holes = [], []
    for loop in loops:
        signed = np.dot(np.cross(loop, np.roll(loop, -1, axis=0)).sum(0), normal)
        if signed < 0 and surface_area(make_planar_face(loop, normal=normal)) < hole_threshold_m2:
            continue
        (outer if signed > 0 else holes).append(loop)
    bases: list[TopoDS_Shape] = [make_planar_face(loop, normal=normal) for loop in outer]
    for hole in holes:
        void = make_planar_face(hole, normal=normal)
        overlap = [common_area(face, void) for face in bases]
        owners = [index for index, area in enumerate(overlap) if area > 1e-9]
        if not owners:
            raise PipelineInputError('CAD hole has no intersecting outer component')
        enclosed_area = surface_area(void)
        containing = [index for index in owners if overlap[index] >= enclosed_area * (1 - 1e-7)]
        owner = (
            min(containing, key=lambda index: surface_area(bases[index]))
            if containing
            else max(owners, key=lambda index: overlap[index])
        )
        operation = BRepAlgoAPI_Cut(bases[owner], void)
        operation.SetFuzzyValue(1e-8)
        operation.Build()
        if not operation.IsDone():
            raise PipelineInputError('CAD internal-boundary subtraction failed')
        bases[owner] = validate(operation.Shape())
    if not bases:
        raise PipelineInputError('Contours contain no outer CAD components')
    return [face for base in bases for face in faces(base)]


def remove_small_holes(shape: TopoDS_Shape, threshold_m2: float) -> TopoDS_Shape:
    """Heal inner wires whose exact CAD enclosed area is strictly below threshold."""
    validate(shape)
    if not np.isfinite(threshold_m2) or threshold_m2 < 0:
        raise PipelineInputError('Hole threshold must be a finite nonnegative square-meter area')
    result = []
    for face in faces(shape):
        orientation = face.Orientation()
        face = TopoDS.Face_s(face.Oriented(TopAbs_FORWARD))
        outer, inner = _wires(face)
        plane = BRepAdaptor_Surface(face).Plane()
        maker = BRepBuilderAPI_MakeFace(plane, outer, True)
        for wire in inner:
            enclosed = BRepBuilderAPI_MakeFace(plane, wire, True).Face()
            if abs(surface_area(enclosed)) >= threshold_m2:
                maker.Add(wire)
        rebuilt = maker.Face()
        rebuilt.Orientation(orientation)
        result.append(validate(rebuilt))
    return result[0] if len(result) == 1 else compound(result)


def common_area(left: TopoDS_Shape, right: TopoDS_Shape) -> float:
    """Exact common surface area in m²; disjoint faces return zero."""
    operation = BRepAlgoAPI_Common(validate(left), validate(right))
    operation.Build()
    if not operation.IsDone():
        raise PipelineInputError('CAD common operation failed')
    return surface_area(operation.Shape())


def fuse_unify(shapes: Iterable[TopoDS_Shape]) -> TopoDS_Shape:
    """Fuse overlapping shapes and remove same-domain internal CAD boundaries."""
    iterator = iter(shapes)
    result = validate(next(iterator))
    for shape in iterator:
        operation = BRepAlgoAPI_Fuse(result, validate(shape))
        operation.SetFuzzyValue(1e-6)
        operation.SetNonDestructive(True)
        operation.Build()
        if not operation.IsDone():
            raise PipelineInputError('CAD fuse operation failed')
        repair = ShapeFix_Shape(operation.Shape())
        repair.SetPrecision(1e-6)
        repair.SetMaxTolerance(1e-4)
        repair.Perform()
        result = validate(repair.Shape())
    unify = ShapeUpgrade_UnifySameDomain(result, True, True, False)
    unify.Build()
    return validate(unify.Shape())


def project_to_plane(
    shape: TopoDS_Shape, origin: NDArray[np.float64], normal: NDArray[np.float64]
) -> TopoDS_Shape:
    """Orthogonally project CAD polygon loops onto an analytic target plane."""
    validate(shape)
    origin, normal = np.asarray(origin, dtype=float), np.asarray(normal, dtype=float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise PipelineInputError('Projection origin must be a finite 3-vector')
    if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-12:
        raise PipelineInputError('Projection normal must be finite and nonzero')
    normal = normal / np.linalg.norm(normal)
    result = []
    for face in faces(shape):
        loops = [loop - np.outer((loop - origin) @ normal, normal) for loop in face_loops(face)]
        result.append(make_planar_face(loops[0], loops[1:], normal=normal))
    return result[0] if len(result) == 1 else compound(result)


def shape_stats(shape: TopoDS_Shape) -> CadShapeStats:
    """Return validated exact surface area and topological face/hole counts."""
    validate(shape)
    items = faces(shape)
    return {
        'area_m2': surface_area(shape),
        'face_count': len(items),
        'hole_count': sum(len(_wires(face)[1]) for face in items),
        'valid': True,
    }
