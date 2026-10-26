"""A bounded terrain correction must compose without replacing native geometry contracts."""

import json
from typing import Any

import numpy as np
import pytest

pytest.importorskip('scipy')

from usim_env_vtc import corridor_repair as repair


def profile_data() -> dict[str, Any]:
    x = np.linspace(0, 10, 101)
    z = -1 - 0.03 * x + 0.6 * np.exp(-(((x - 5) / 0.3) ** 2))
    return {
        'start': [10, -5],
        'end': [20, -5],
        'samples_xy_z': np.column_stack((x + 10, np.full(len(x), -5), z)).tolist(),
    }


def test_source_profile_retains_end_levels_and_relief_without_using_pcd_quantiles():
    profile = profile_data()
    profile['official_2019_ground_profile'] = [{'height_quantiles_m': [-10, -9, -8]}]
    along, z = repair.smooth_profile(profile, 1)
    assert z[0] == pytest.approx(profile['samples_xy_z'][0][2])
    assert z[-1] == pytest.approx(profile['samples_xy_z'][-1][2])
    assert np.max(np.abs(np.diff(z) / np.diff(along))) < 0.25
    assert z.min() > -2
    assert np.ptp(z) > 0.2
    assert np.max(z - (-1 - 0.03 * along)) > 0.05


def test_finite_boundary_and_exterior_are_bitwise_preserved():
    profile = profile_data()
    along, z = repair.smooth_profile(profile, 1)
    points = np.array(
        [
            [7.9, -5, -1],
            [8, -5, -1],
            [22, -5, -1],
            [22.1, -5, -1],
            [15, -8, -1],
            [15, -8.1, -1],
            [15, -5, -1],
        ],
        dtype=np.float64,
    )
    before = points.copy()
    fixed, _, active = repair.repair_points(points, profile, along, z, width=2, blend=2)
    assert np.array_equal(fixed[:6], before[:6])
    assert active.tolist() == [False] * 6 + [True]
    assert fixed[-1, 2] != points[-1, 2]
    assert np.array_equal(points, before)
    assert np.array_equal(fixed[:, :2], before[:, :2])


def test_rejects_incomplete_native_profile():
    profile = profile_data()
    profile['samples_xy_z'] = profile['samples_xy_z'][::3]
    with pytest.raises(ValueError, match='cover'):
        repair.smooth_profile(profile, 1)


def test_original_payloads_topology_collision_and_seams_survive_composition(tmp_path):
    pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    profile = profile_data()
    world_path = tmp_path / 'base.usda'
    stage = Usd.Stage.CreateNew(str(world_path))
    UsdGeom.Xform.Define(stage, '/World')
    entries, originals = [], {}
    for number, (left, right) in enumerate(((0, 5), (5, 10))):
        tile_path = tmp_path / f'tile{number}.usda'
        tile = Usd.Stage.CreateNew(str(tile_path))
        root = UsdGeom.Xform.Define(tile, '/Tile')
        root.AddTranslateOp().Set((10, -5, 2))
        tile.SetDefaultPrim(root.GetPrim())
        x, y = np.meshgrid(np.linspace(left, right, 51), np.linspace(-4, 4, 81))
        z = -3 - 0.03 * x + 0.6 * np.exp(-(((x - 5) / 0.3) ** 2))
        points = np.column_stack((x.ravel(), y.ravel(), z.ravel())).astype(np.float32)
        mesh = UsdGeom.Mesh.Define(tile, '/Tile/Ground')
        mesh.CreatePointsAttr(points.tolist())
        a = (np.arange(80)[:, None] * 51 + np.arange(50)).ravel()
        faces = np.concatenate(
            (
                np.column_stack((a, a + 1, a + 51)),
                np.column_stack((a + 1, a + 52, a + 51)),
            )
        )
        mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        material = UsdShade.Material.Define(tile, '/Tile/Looks/Ground')
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        tile.GetRootLayer().Save()
        path = f'/World/Terrain/T{number}'
        stage.DefinePrim(path).GetPayloads().AddPayload(tile_path.name, '/Tile')
        entries.append(
            {
                'path': path,
                'asset': tile_path.name,
                'prim': '/Tile',
                'min': [10 + left, -9, -2],
                'max': [10 + right, -1, 0],
            }
        )
        originals[path] = (points.copy(), faces.ravel().copy(), tile_path.read_bytes())
    stage.GetRootLayer().Save()
    original_world = world_path.read_bytes()
    index_path = tmp_path / 'index.json'
    index_path.write_text(json.dumps(entries))
    profile['source'] = str(world_path)
    profile_path = tmp_path / 'profile.json'
    profile_path.write_text(json.dumps(profile))
    out = tmp_path / 'derived'
    report = repair.repair_world(
        world_path, index_path, profile_path, out, width=2, blend=2, sigma=1
    )
    assert report['changed_vertices'] > 0
    assert report['core_limits_pass']
    assert report['payload_roots'] == 2
    assert report['seams_after']['duplicate_xy_groups'] > 0
    assert report['seams_after']['maximum_seam_gap_m'] == 0
    assert world_path.read_bytes() == original_world
    composed = Usd.Stage.Open(str(out / 'world.usda'))
    correction = Usd.Stage.Open(str(out / 'corridor-correction.usdc'))
    for prim in correction.TraverseAll():
        assert {attr.GetName() for attr in prim.GetAttributes()} <= {'points', 'extent'}
        assert not prim.GetRelationships()
    actual_changed = 0
    for number, entry in enumerate(entries):
        path = entry['path']
        source, faces, original_bytes = originals[path]
        actual = UsdGeom.Mesh(composed.GetPrimAtPath(path + '/Ground'))
        points = np.asarray(actual.GetPointsAttr().Get())
        actual_changed += int(np.count_nonzero(points[:, 2] != source[:, 2]))
        assert np.array_equal(points[:, :2], source[:, :2])
        outside = np.abs(source[:, 1]) >= 3
        assert np.array_equal(points[outside], source[outside])
        assert np.array_equal(np.asarray(actual.GetFaceVertexIndicesAttr().Get()), faces)
        assert actual.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
        assert UsdShade.MaterialBindingAPI(actual).GetDirectBinding().GetMaterialPath()
        assert (tmp_path / f'tile{number}.usda').read_bytes() == original_bytes
    assert report['changed_vertices'] == actual_changed
    derived_index = json.loads((out / 'world.index.json').read_text())
    assert [entry['path'] for entry in derived_index] == [entry['path'] for entry in entries]
    assert all((tmp_path / entry['asset']).exists() for entry in derived_index)
