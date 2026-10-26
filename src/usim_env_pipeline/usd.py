"""Regional USD geometry, triangle collisions, and unloaded streaming roots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .models import Payload, RegionReport

if TYPE_CHECKING:
    from pxr import Usd


@dataclass(frozen=True, slots=True)
class LayerMetadata:
    license: str
    attribution: str
    source_catalog: str
    frame: str
    color: str


@dataclass(frozen=True, slots=True)
class Geometry:
    positions: NDArray[np.float32]
    ids: NDArray[np.int64]
    colors: NDArray[np.float32]
    faces: NDArray[np.int32] | None = None


def stage_base(path: Path, root: str, metadata: LayerMetadata) -> Usd.Stage:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateNew(str(path))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, root).GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.GetRootLayer().customLayerData = {
        'license': metadata.license,
        'attribution': metadata.attribution,
        'sourceCatalog': metadata.source_catalog,
        'frame': metadata.frame,
        'color': metadata.color,
    }
    return stage


def author_geometry(path: Path, data: Geometry, metadata: LayerMetadata) -> None:
    from pxr import Sdf, UsdGeom, UsdPhysics, Vt

    stage = stage_base(path, '/Tile', metadata)
    if data.faces is None:
        geometry = UsdGeom.Points.Define(stage, '/Tile/Points')
        geometry.CreateWidthsAttr(Vt.FloatArray([0.055]))
        geometry.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        geometry.CreateIdsAttr(Vt.Int64Array.FromNumpy(data.ids))
    else:
        geometry = UsdGeom.Mesh.Define(stage, '/Tile/Surface')
        geometry.CreateFaceVertexCountsAttr(
            Vt.IntArray.FromNumpy(np.full(len(data.faces), 3, dtype=np.int32))
        )
        geometry.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(data.faces.ravel()))
        geometry.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        geometry.CreateDoubleSidedAttr(True)
        geometry.GetPrim().CreateAttribute(
            'source:recordIds', Sdf.ValueTypeNames.Int64Array, custom=True
        ).Set(Vt.Int64Array.FromNumpy(data.ids))
        UsdPhysics.CollisionAPI.Apply(geometry.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(geometry.GetPrim()).CreateApproximationAttr('none')
    geometry.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(data.positions))
    geometry.CreateExtentAttr(
        Vt.Vec3fArray.FromNumpy(np.stack([data.positions.min(axis=0), data.positions.max(axis=0)]))
    )
    geometry.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(
        Vt.Vec3fArray.FromNumpy(data.colors)
    )
    stage.GetRootLayer().Save()


def make_layers(output: Path, reports: list[RegionReport], metadata: LayerMetadata) -> None:
    from pxr import UsdGeom

    from .storage import write_json

    entries: list[Payload] = []
    point_entries: list[Payload] = []
    for filename, folder, only_mesh in [
        ('streamed.usda', 'meshes', True),
        ('points.usda', 'points', False),
    ]:
        stage = stage_base(output / filename, '/World', metadata)
        group = 'Regions' if only_mesh else 'PointRegions'
        UsdGeom.Xform.Define(stage, f'/World/{group}')
        for report in reports:
            if only_mesh and not report['triangles']:
                continue
            path = f'/World/{group}/{report["name"]}'
            prim = UsdGeom.Xform.Define(stage, path).GetPrim()
            prim.GetPayloads().AddPayload(f'./{folder}/{report["name"]}.usdc', '/Tile')
            entry: Payload = {
                'path': path,
                'min': report.get('mesh_min', report['min']) if only_mesh else report['min'],
                'max': report.get('mesh_max', report['max']) if only_mesh else report['max'],
                'payload': f'{folder}/{report["name"]}.usdc',
            }
            (entries if only_mesh else point_entries).append(entry)
        stage.GetRootLayer().Save()
    write_json(output / 'index.json', entries)
    write_json(output / 'point-index.json', point_entries)
    write_json(output / 'world-index.json', entries + point_entries)
    world = stage_base(output / 'world.usda', '/World', metadata)
    world.GetRootLayer().subLayerPaths = ['./streamed.usda', './points.usda']
    world.GetRootLayer().Save()
    archive = stage_base(output / 'archival-points.usda', '/World', metadata)
    archive.GetRootLayer().subLayerPaths = ['./points.usda']
    archive.GetRootLayer().Save()
