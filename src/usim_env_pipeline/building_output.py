"""Publish closed CAD proxies while retaining unrelated measured source triangles."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

from .building_types import BuildingConfig, Cuboid, Facade
from .cad_exchange import read_brep, tessellate, write_brep, write_step
from .cad_kernel import compound
from .storage import write_json
from .usd import Geometry, LayerMetadata, author_geometry, stage_base


def publish(
    output: Path,
    facades: list[Facade],
    models: list[Cuboid],
    config: BuildingConfig,
    metadata: LayerMetadata,
) -> tuple[int, int]:
    """Return surviving and replaced original Regions triangles, excluding Ground."""
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    folder = output / 'building-models'
    (folder / 'residual').mkdir(parents=True, exist_ok=True)
    world = stage_base(output / 'buildings.usda', '/World', metadata)
    world.GetRootLayer().subLayerPaths = ['./authored.usda']
    UsdGeom.Imageable(world.OverridePrim('/World/PointRegions')).CreateVisibilityAttr(
        UsdGeom.Tokens.invisible
    )
    shapes, rows = [], []
    for number, model in enumerate(models):
        name = f'model_{number:04}'
        path = folder / f'{name}.usdc'
        origin = model.center - model.axes @ (model.size / 2)
        frame = gp_Ax2(gp_Pnt(*origin), gp_Dir(*model.axes[:, 2]), gp_Dir(*model.axes[:, 0]))
        shape = BRepPrimAPI_MakeBox(frame, *model.size).Solid()
        write_brep(shape, path.with_suffix('.brep'))
        # Derive both visible and collidable geometry from the published authority.
        shape = read_brep(path.with_suffix('.brep'))
        shapes.append(shape)
        positions, triangles = tessellate(shape)
        author_geometry(
            path,
            Geometry(
                positions.astype(np.float32),
                np.empty(0, dtype=np.int64),
                np.tile(np.array([[0.72, 0.72, 0.68]], dtype=np.float32), (len(positions), 1)),
                triangles,
            ),
            metadata,
        )
        stage = Usd.Stage.Open(str(path))
        prim = stage.GetPrimAtPath('/Tile/Surface')
        prim.RemoveProperty('source:recordIds')
        prim.CreateAttribute('building:authority', Sdf.ValueTypeNames.Asset, custom=True).Set(
            Sdf.AssetPath(f'./{name}.brep')
        )
        prim.CreateAttribute('building:kind', Sdf.ValueTypeNames.String, custom=True).Set(
            model.kind
        )
        inference = (
            'Unobserved volume and faces are inferred approximation, not measured architecture.'
        )
        prim.CreateAttribute('building:inference', Sdf.ValueTypeNames.String, custom=True).Set(
            inference
        )
        UsdPhysics.MeshCollisionAPI(prim).CreateApproximationAttr('convexHull')
        stage.GetRootLayer().Save()
        source_planes = sorted(
            {plane for index in model.facade_indices for plane in facades[index].source_planes}
        )
        support_ids = (
            np.unique(
                np.concatenate([facades[index].support_ids for index in model.facade_indices])
            )
            if model.facade_indices
            else np.empty(0, dtype=np.int64)
        )
        np.savez_compressed(
            path.with_suffix('.npz'),
            center=model.center,
            size=model.size,
            axes=model.axes,
            facade_indices=np.asarray(model.facade_indices, dtype=np.int32),
            source_planes=np.asarray(source_planes, dtype=np.int32),
            support_ids=support_ids,
        )
        rows.append(
            {
                'name': name,
                'kind': model.kind,
                'center': model.center.tolist(),
                'size': model.size.tolist(),
                'axes': model.axes.tolist(),
                'facade_indices': list(model.facade_indices),
                'source_planes': source_planes,
                'support_records': len(support_ids),
                'volume_m3': float(np.prod(model.size)),
                'inference': inference,
            }
        )
        UsdGeom.Xform.Define(
            world, f'/World/BuildingModels/{name}'
        ).GetPrim().GetPayloads().AddPayload(f'./building-models/{name}.usdc', '/Tile')
    authority = compound(shapes)
    write_brep(authority, folder / 'model.brep')
    write_step(authority, folder / 'model.step')
    write_json(folder / 'models.json', rows)

    fitted = sorted({index for model in models for index in model.facade_indices})
    residual, replaced = 0, 0
    entries = json.loads((output / 'authored-index.json').read_text(encoding='utf-8'))
    for entry in entries:
        if not entry['path'].startswith('/World/Regions/'):
            continue
        stage = Usd.Stage.Open(str(output / entry['payload']))
        mesh = UsdGeom.Mesh.Get(stage, '/Tile/Surface')
        positions = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3)
        triangles = positions[faces].astype(np.float64)
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        vertical = (lengths > 0) & (
            np.abs(normals[:, 2]) <= lengths * np.sin(np.deg2rad(config.vertical_angle))
        )
        remove = np.zeros(len(faces), dtype=bool)
        tolerance = 1e-6
        tile_lower, tile_upper = positions.min(0), positions.max(0)
        for index in fitted:
            facade = facades[index]
            ends = np.outer([facade.lower[0], facade.upper[0]], facade.tangent)
            ends -= facade.normal * facade.offset
            lower, upper = ends.min(0), ends.max(0)
            lower[2], upper[2] = facade.lower[1], facade.upper[1]
            if (tile_upper < lower - config.support_distance).any() or (
                tile_lower > upper + config.support_distance
            ).any():
                continue
            along = triangles @ facade.tangent
            distance = np.abs(triangles @ facade.normal + facade.offset)
            inside = (distance <= config.support_distance + tolerance).all(axis=1)
            inside &= (along >= facade.lower[0] - tolerance).all(axis=1)
            inside &= (along <= facade.upper[0] + tolerance).all(axis=1)
            inside &= (triangles[:, :, 2] >= facade.lower[1] - tolerance).all(axis=1)
            inside &= (triangles[:, :, 2] <= facade.upper[1] + tolerance).all(axis=1)
            remove |= vertical & inside
        keep = ~remove
        residual += int(keep.sum())
        replaced += int(remove.sum())
        if not remove.any():
            continue
        world.OverridePrim(entry['path']).SetActive(False)
        if not keep.any():
            continue
        name = entry['path'].rsplit('/', 1)[1]
        ids = np.asarray(mesh.GetPrim().GetAttribute('source:recordIds').Get(), dtype=np.int64)
        colors = np.asarray(mesh.GetDisplayColorPrimvar().Get(), dtype=np.float32)
        author_geometry(
            folder / 'residual' / f'{name}.usdc',
            Geometry(positions, ids, colors, faces[keep]),
            metadata,
        )
        np.savez_compressed(
            folder / 'residual' / f'{name}.npz',
            source_face_indices=np.flatnonzero(keep),
        )
        UsdGeom.Xform.Define(
            world, f'/World/BuildingResidual/{name}'
        ).GetPrim().GetPayloads().AddPayload(f'./building-models/residual/{name}.usdc', '/Tile')
    world.GetRootLayer().Save()
    return residual, replaced
