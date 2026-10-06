"""Keep corrected map geometry, replacing imported surface art with original colors."""

from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt


class TerrainEntry(TypedDict):
    path: str
    asset: str
    prim: str
    min: list[float]
    max: list[float]


def surface_colors(
    weights: NDArray[np.uint8], extra: NDArray[np.uint8], seed: int
) -> NDArray[np.float32]:
    """Paint five map-data layers using independent procedural surface recipes."""
    rng = np.random.default_rng(seed)
    h, w = weights.shape[:2]
    y, x = np.mgrid[:h, :w]
    grit = rng.uniform(-0.025, 0.025, (h, w, 1))
    gravel = np.array([0.30, 0.27, 0.21]) + grit * 2
    asphalt = np.array([0.065, 0.073, 0.080]) + grit * 0.45
    joints = ((x % 6 == 0) | (y % 6 == 0))[..., None]
    tile = np.where(joints, np.array([0.17, 0.17, 0.16]), np.array([0.44, 0.40, 0.33]))
    tile = tile + grit
    grass = np.array([0.13, 0.22, 0.055]) + grit
    white = np.array([0.68, 0.69, 0.64]) + grit * 0.5
    masks = np.concatenate((weights[..., :3], extra[..., :2]), axis=2).astype(np.float32)
    total = np.maximum(masks.sum(axis=2, keepdims=True), 1)
    result = (
        sum(
            masks[..., number, None] * color
            for number, color in enumerate((gravel, asphalt, tile, grass, white))
        )
        / total
    )
    return np.clip(result, 0, 1).astype(np.float32)


def author_tile(
    source: Usd.Stage,
    entry: TerrainEntry,
    masks: Path,
    out: Path,
    seed: int,
) -> TerrainEntry:
    """Export only repaired topology, placement and physics, with no upstream artwork arcs."""
    path = entry['path']
    source.Load(path)
    original = source.GetPrimAtPath(path)
    name = original.GetName()
    filename = name + '.usdc'
    tile = Usd.Stage.CreateNew(str((out / filename).resolve()))
    root = UsdGeom.Xform.Define(tile, '/Tile')
    tile.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(tile, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(tile, 1)
    root.MakeMatrixXform().Set(UsdGeom.Xformable(original).GetLocalTransformation())
    with (
        Image.open(masks / f'{name}-0.png') as first,
        Image.open(masks / f'{name}-1.png') as second,
    ):
        colors = surface_colors(np.asarray(first), np.asarray(second), seed)
    material = UsdShade.Material.Define(tile, '/Tile/Looks/Ground')
    shader = UsdShade.Shader.Define(tile, '/Tile/Looks/Ground/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    reader = UsdShade.Shader.Define(tile, '/Tile/Looks/Ground/Color')
    reader.CreateIdAttr('UsdPrimvarReader_float3')
    reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('displayColor')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(
        reader.CreateOutput('result', Sdf.ValueTypeNames.Float3)
    )
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.88)
    material.CreateSurfaceOutput().ConnectToSource(
        shader.CreateOutput('surface', Sdf.ValueTypeNames.Token)
    )
    vertices = 0
    for prim in original.GetChildren():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        old = UsdGeom.Mesh(prim)
        mesh = UsdGeom.Mesh.Define(tile, '/Tile/' + prim.GetName())
        points = np.asarray(old.GetPointsAttr().Get(), dtype=np.float32)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
        mesh.CreateFaceVertexCountsAttr(old.GetFaceVertexCountsAttr().Get())
        mesh.CreateFaceVertexIndicesAttr(old.GetFaceVertexIndicesAttr().Get())
        mesh.CreateOrientationAttr(old.GetOrientationAttr().Get())
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.stack((points.min(0), points.max(0)))))
        component_matrix = UsdGeom.Xformable(prim).GetLocalTransformation()
        UsdGeom.Xformable(mesh).MakeMatrixXform().Set(component_matrix)
        # Recompute normals from the repaired lattice rather than copying imported art.
        lattice = points.reshape(127, 127, 3)
        dy, dx = np.gradient(lattice[..., 2], 0.1)
        normals = np.stack((-dx, dy, np.ones_like(dx)), axis=2)
        normals /= np.linalg.norm(normals, axis=2, keepdims=True)
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals.reshape(-1, 3)))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        offset = component_matrix.ExtractTranslation()
        col = np.clip(np.rint((points[:, 0] + offset[0]) * 10).astype(int), 0, 1008)
        row = np.clip(np.rint(-(points[:, 1] + offset[1]) * 10).astype(int), 0, 1008)
        mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(
            Vt.Vec3fArray.FromNumpy(colors[row, col])
        )
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('none')
        vertices += len(points)
    tile.GetRootLayer().customLayerData = {
        'geometryProvenance': 'VTC landscape with retained road-relief correction',
        'surfaceArtwork': 'original procedural colors; no imported images or vertex colors',
        'sourceArtworkCopied': False,
    }
    tile.GetRootLayer().Save()
    bounds = (
        UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        .ComputeWorldBound(root.GetPrim())
        .ComputeAlignedRange()
    )
    source.Unload(path)
    print(f'AUTHORED_TERRAIN {name} vertices={vertices}', flush=True)
    return {
        'path': path,
        'asset': 'terrain/' + filename,
        'prim': '/Tile',
        'min': list(bounds.GetMin()),
        'max': list(bounds.GetMax()),
    }
