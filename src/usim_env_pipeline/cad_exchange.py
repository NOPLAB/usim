"""Authoritative CAD exchange and tessellation derived only from B-Rep faces."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepTools import BRepTools
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
from OCP.TopAbs import TopAbs_REVERSED, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS_Shape
from OCP.gp import gp_Pnt, gp_Trsf

from .cad_kernel import compound, face_loops, face_normal, faces, make_planar_faces, validate
from .models import PipelineInputError


def _scaled(shape: TopoDS_Shape, factor: float) -> TopoDS_Shape:
    transform = gp_Trsf()
    transform.SetScale(gp_Pnt(0, 0, 0), factor)
    return validate(BRepBuilderAPI_Transform(shape, transform, True).Shape())


def write_brep(shape: TopoDS_Shape, path: Path) -> None:
    """Write the canonical meter-valued B-Rep, without changing geometry."""
    validate(shape)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not BRepTools.Write_s(shape, str(path)):
        raise PipelineInputError(f'BRep write failed: {path}')


def read_brep(path: Path) -> TopoDS_Shape:
    """Read canonical meter-valued B-Rep."""
    shape = TopoDS_Shape()
    if not BRepTools.Read_s(shape, str(path), BRep_Builder()):
        raise PipelineInputError(f'BRep read failed: {path}')
    return validate(shape)


def write_step(shape: TopoDS_Shape, path: Path) -> None:
    """Export a copy scaled to millimeters; STEP explicitly declares millimeters."""
    validate(shape)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = STEPControl_Writer()
    # Reconstruct surviving trims: copied inner wires can touch their outer boundary
    # after unit conversion, while CAD subtraction preserves the same surface exactly.
    exported = []
    for face in faces(shape):
        loops = face_loops(face)
        if len(loops) == 1:
            exported.append(face)
            continue
        loops = [
            loop[np.linalg.norm(loop - np.roll(loop, 1, axis=0), axis=1) > 1e-7] for loop in loops
        ]
        exported.extend(make_planar_faces(loops, face_normal(face)))
    Interface_Static.SetCVal_s('write.step.unit', 'MM')
    # Fused faces carry local repair tolerances; an average understates their uncertainty.
    Interface_Static.SetIVal_s('write.precision.mode', 1)
    # Analytic planes reconstruct exact pcurves from their 3D edges on import.
    Interface_Static.SetIVal_s('write.surfacecurve.mode', 0)
    authority = shape if TopExp_Explorer(shape, TopAbs_SOLID).More() else compound(exported)
    if writer.Transfer(_scaled(authority, 1000.0), STEPControl_AsIs) != IFSelect_RetDone:
        raise PipelineInputError('STEP transfer failed')
    if writer.Write(str(path)) != IFSelect_RetDone:
        raise PipelineInputError(f'STEP write failed: {path}')


def read_step(path: Path) -> TopoDS_Shape:
    """Read STEP into millimeters explicitly, then recover canonical meters."""
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise PipelineInputError(f'STEP read failed: {path}')
    reader.SetSystemLengthUnit(1.0)
    if reader.TransferRoots() == 0:
        raise PipelineInputError('STEP contains no transferable roots')
    transform = gp_Trsf()
    transform.SetScale(gp_Pnt(0, 0, 0), 0.001)
    imported = BRepBuilderAPI_Transform(reader.OneShape(), transform, True).Shape()
    if TopExp_Explorer(imported, TopAbs_SOLID).More():
        return validate(imported)
    restored = []
    for face in faces(imported):
        if BRepCheck_Analyzer(face).IsValid():
            restored.append(face)
            continue
        loops = [
            loop[np.linalg.norm(loop - np.roll(loop, 1, axis=0), axis=1) > 1e-7]
            for loop in face_loops(face)
        ]
        restored.extend(make_planar_faces(loops, face_normal(face)))
    return compound(restored)


def tessellate(
    shape: TopoDS_Shape, deflection: float = 0.01
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Return world-meter vertices and oriented triangle indices from actual B-Rep."""
    validate(shape)
    if not np.isfinite(deflection) or deflection <= 0:
        raise PipelineInputError('Tessellation deflection must be positive meters')
    mesh = BRepMesh_IncrementalMesh(shape, deflection, False, 0.1, False)
    if not mesh.IsDone():
        raise PipelineInputError('B-Rep tessellation failed')
    vertices, triangles = [], []
    for face in faces(shape):
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        if triangulation is None:
            raise PipelineInputError('CAD face has no kernel triangulation')
        offset = len(vertices)
        for index in range(1, triangulation.NbNodes() + 1):
            point = triangulation.Node(index).Transformed(location.Transformation())
            vertices.append((point.X(), point.Y(), point.Z()))
        for index in range(1, triangulation.NbTriangles() + 1):
            triangle = list(triangulation.Triangle(index).Get())
            if face.Orientation() == TopAbs_REVERSED:
                triangle[1], triangle[2] = triangle[2], triangle[1]
            triangles.append([offset + item - 1 for item in triangle])
    return np.asarray(vertices, dtype=np.float64), np.asarray(triangles, dtype=np.int32)
