"""Original-art composition must preserve placement and corrected map geometry."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('pxr')
pytest.importorskip('PIL')
from PIL import Image
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

from usim_env_vtc import author_scene
from usim_env_vtc.authored_terrain import surface_colors


def test_surface_recipes_keep_map_classes_and_repeatable_original_colors():
    # Given the five pure map-data surface weights.
    weights = np.zeros((1, 5, 4), dtype=np.uint8)
    extra = weights.copy()
    for i in range(3):
        weights[0, i, i] = 255
    extra[0, 3, 0] = 255
    extra[0, 4, 1] = 255
    # When painting original surface pigments twice.
    colors = surface_colors(weights, extra, 11)
    repeated = surface_colors(weights, extra, 11)
    # Then labels are distinguishable, deterministic, finite and in the linear color domain.
    assert np.array_equal(colors, repeated)
    assert len(np.unique(colors.reshape(-1, 3), axis=0)) == 5
    assert colors[0, 1].mean() < colors[0, 0].mean() < colors[0, 4].mean()
    assert colors[0, 3, 1] > colors[0, 3, 0]
    assert np.isfinite(colors).all() and np.all((colors >= 0) & (colors <= 1))


def test_independent_scene_retains_repaired_geometry_placement_and_no_imported_art(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Given an imported cube, a hidden helper and a shifted repaired terrain component.
    prototype = Usd.Stage.CreateNew(str(tmp_path / 'Engine_BasicShapes_Cube.usdc'))
    prototype.SetDefaultPrim(UsdGeom.Xform.Define(prototype, '/Model').GetPrim())
    UsdGeom.Cube.Define(prototype, '/Model/ImportedCube').CreateSizeAttr(1)
    prototype.GetRootLayer().Save()
    source_path = tmp_path / 'source.usda'
    source = Usd.Stage.CreateNew(str(source_path))
    source.SetDefaultPrim(UsdGeom.Xform.Define(source, '/World').GetPrim())
    parent = UsdGeom.Xform.Define(source, '/World/Structures/VTC')
    parent.AddTranslateOp().Set((20, -30, 1))
    cube = UsdGeom.Xform.Define(source, '/World/Structures/VTC/Cube')
    cube.AddTranslateOp().Set((4, 5, 2))
    cube.AddRotateZOp().Set(29)
    cube.AddScaleOp().Set((2, 3, 4))
    cube.GetPrim().GetReferences().AddReference(prototype.GetRootLayer().identifier, '/Model')
    hidden = UsdGeom.Xform.Define(source, '/World/Structures/Reference/Helper')
    hidden.GetPrim().GetReferences().AddReference(prototype.GetRootLayer().identifier, '/Model')
    tile = UsdGeom.Xform.Define(source, '/World/Terrain/TC_x0_y0')
    tile.AddTranslateOp().Set((-40, 10, 0))
    mesh = UsdGeom.Mesh.Define(source, '/World/Terrain/TC_x0_y0/LandscapeComponent_1')
    UsdGeom.Xformable(mesh).AddTranslateOp().Set((12.6, 0, 0))
    y, x = np.mgrid[:127, :127]
    points = np.column_stack((x.ravel() / 10, -y.ravel() / 10, x.ravel() * 0.003))
    mesh.CreatePointsAttr(points.tolist())
    mesh.CreateFaceVertexCountsAttr([3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 127, 1, 1, 127, 128])
    source.GetRootLayer().Save()
    source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    inventory = tmp_path / 'inventory.json'
    inventory.write_text(
        json.dumps(
            {
                'models': [
                    {'name': 'Engine/BasicShapes/Cube', 'min': [-0.5] * 3, 'max': [0.5] * 3}
                ],
            }
        )
    )
    index = tmp_path / 'index.json'
    index.write_text(
        json.dumps(
            [
                {
                    'path': str(tile.GetPath()),
                    'min': [-40, -3, 0],
                    'max': [-14, 10, 1],
                }
            ]
        )
    )
    masks = tmp_path / 'masks'
    masks.mkdir()
    weights = np.zeros((1009, 1009, 4), dtype=np.uint8)
    weights[:, :126, 0] = 255
    weights[:, 126:, 1] = 255
    Image.fromarray(weights).save(masks / 'TC_x0_y0-0.png')
    Image.fromarray(np.zeros_like(weights)).save(masks / 'TC_x0_y0-1.png')
    out = tmp_path / 'output'
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'author_scene',
            '--world',
            str(source_path),
            '--index',
            str(index),
            '--inventory',
            str(inventory),
            '--masks',
            str(masks),
            '--out',
            str(out),
        ],
    )
    # When generating and reopening the standalone world, not an upstream override.
    author_scene.main()
    actual = Usd.Stage.Open(str(out / 'world.usda'))
    # Then hierarchy and exact world placement survive, but imported artwork does not.
    placed = actual.GetPrimAtPath(str(cube.GetPath()))
    assert UsdGeom.XformCache().GetLocalToWorldTransform(placed) == (
        UsdGeom.XformCache().GetLocalToWorldTransform(cube.GetPrim())
    )
    assert placed.IsInstance()
    assert not actual.GetPrimAtPath(str(hidden.GetPath()))
    ground = UsdGeom.Mesh(actual.GetPrimAtPath(str(mesh.GetPath())))
    assert np.array_equal(np.asarray(ground.GetPointsAttr().Get()), points.astype(np.float32))
    assert ground.GetFaceVertexIndicesAttr().Get() == mesh.GetFaceVertexIndicesAttr().Get()
    assert ground.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
    assert UsdGeom.XformCache().GetLocalToWorldTransform(ground.GetPrim()) == (
        UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim())
    )
    # The shifted component samples asphalt at x=126, not gravel at its local x=0.
    assert np.asarray(ground.GetDisplayColorPrimvar().Get())[0].mean() < 0.1
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(
        Sdf.AssetPath(str(out / 'world.usda'))
    )
    assert not unresolved and not assets
    assert all(Path(layer.realPath).is_relative_to(out) for layer in layers)
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_digest
    report = json.loads((out / 'authorship.json').read_text())
    assert report['placementCount'] == 1
    assert report['omittedNonDisplayHelpers'] == [str(hidden.GetPath())]
    entries = json.loads((out / 'world.index.json').read_text())
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default'])
    for entry in entries:
        bounds = cache.ComputeWorldBound(actual.GetPrimAtPath(entry['path'])).ComputeAlignedRange()
        assert all(
            entry['min'][i] <= bounds.GetMin()[i] and entry['max'][i] >= bounds.GetMax()[i]
            for i in range(3)
        )
