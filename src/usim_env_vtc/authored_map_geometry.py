"""Retain measured and VTC-authored geometry without inheriting imported appearance."""

from pathlib import Path

from pxr import Usd, UsdGeom, UsdPhysics

from .authored_terrain import TerrainEntry


def retain_posts(source: Usd.Stage, world: Usd.Stage, out: Path) -> list[TerrainEntry]:
    """Re-author the seven measured proxies into bounded, independently owned cell payloads."""
    entries = []
    parent = source.GetPrimAtPath('/World/InferredStructures')
    if not parent:
        return entries
    for old_cell in parent.GetChildren():
        filename = 'posts_' + old_cell.GetName() + '.usdc'
        cell = Usd.Stage.CreateNew(str((out / filename).resolve()))
        root = UsdGeom.Xform.Define(cell, '/Cell')
        cell.SetDefaultPrim(root.GetPrim())
        UsdGeom.SetStageUpAxis(cell, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(cell, 1)
        for prim in old_cell.GetChildren():
            old = UsdGeom.Cylinder(prim)
            new = UsdGeom.Cylinder.Define(cell, '/Cell/' + prim.GetName())
            for attr in ('height', 'radius', 'axis', 'extent', 'primvars:displayColor'):
                original = prim.GetAttribute(attr)
                new.GetPrim().CreateAttribute(attr, original.GetTypeName()).Set(original.Get())
            UsdGeom.Xformable(new).MakeMatrixXform().Set(
                UsdGeom.Xformable(old).GetLocalTransformation()
            )
            UsdPhysics.CollisionAPI.Apply(new.GetPrim())
        cell.GetRootLayer().Save()
        path = str(old_cell.GetPath())
        world.DefinePrim(path, 'Xform').GetPayloads().AddPayload(filename, '/Cell')
        bounds = (
            UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            .ComputeWorldBound(root.GetPrim())
            .ComputeAlignedRange()
        )
        entries.append(
            {
                'path': path,
                'asset': filename,
                'prim': '/Cell',
                'min': list(bounds.GetMin()),
                'max': list(bounds.GetMax()),
            }
        )
    return entries


def retain_bsp(source: Usd.Stage, world: Usd.Stage, out: Path) -> TerrainEntry | None:
    """Keep VTC-authored map triangles, stripping all inherited UVs/materials/artwork."""
    original = source.GetPrimAtPath('/World/BSP')
    if not original:
        return None
    cell = Usd.Stage.CreateNew(str((out / 'bsp.usdc').resolve()))
    root = UsdGeom.Xform.Define(cell, '/BSP')
    cell.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(cell, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(cell, 1)
    count = 0
    for prim in Usd.PrimRange(original):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        old = UsdGeom.Mesh(prim)
        new = UsdGeom.Mesh.Define(cell, f'/BSP/Surface_{count}')
        new.CreatePointsAttr(old.GetPointsAttr().Get())
        new.CreateFaceVertexCountsAttr(old.GetFaceVertexCountsAttr().Get())
        new.CreateFaceVertexIndicesAttr(old.GetFaceVertexIndicesAttr().Get())
        new.CreateExtentAttr(old.GetExtentAttr().Get())
        new.CreateOrientationAttr(old.GetOrientationAttr().Get())
        new.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        new.CreateDisplayColorPrimvar().Set([(0.27, 0.28, 0.25)])
        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        UsdGeom.Xformable(new).MakeMatrixXform().Set(matrix)
        UsdPhysics.CollisionAPI.Apply(new.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(new.GetPrim()).CreateApproximationAttr('none')
        count += 1
    cell.GetRootLayer().Save()
    world.DefinePrim('/World/BSP', 'Xform').GetPayloads().AddPayload('bsp.usdc', '/BSP')
    bounds = (
        UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        .ComputeWorldBound(root.GetPrim())
        .ComputeAlignedRange()
    )
    return {
        'path': '/World/BSP',
        'asset': 'bsp.usdc',
        'prim': '/BSP',
        'min': list(bounds.GetMin()),
        'max': list(bounds.GetMax()),
    }
