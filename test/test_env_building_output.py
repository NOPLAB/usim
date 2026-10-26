"""Real closed CAD proxies and selective, immutable-source USD composition."""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('OCP')
pytest.importorskip('pxr')

from OCP.BRepCheck import BRepCheck_Analyzer, BRepCheck_NoError, BRepCheck_Shell
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_SHELL, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from pxr import Usd, UsdGeom, UsdPhysics

from usim_env_pipeline.building_output import publish
from usim_env_pipeline.building_types import BuildingConfig, Cuboid, Facade
from usim_env_pipeline.cad_exchange import read_brep, read_step, tessellate
from usim_env_pipeline.cad_kernel import faces
from usim_env_pipeline.storage import write_json
from usim_env_pipeline.usd import Geometry, LayerMetadata, author_geometry, stage_base

METADATA = LayerMetadata('CC-BY-4.0', 'fixture', '', 'XYZ', 'measured source')


def source_world(output, tiles):
    world = stage_base(output / 'authored.usda', '/World', METADATA)
    entries = []
    for root, name, positions, triangles in tiles:
        path = output / f'{root}_{name}.usdc'
        ids = np.arange(len(positions), dtype=np.int64) + 100
        colors = np.arange(len(positions) * 3, dtype=np.float32).reshape(-1, 3) / 100
        author_geometry(path, Geometry(positions, ids, colors, triangles), METADATA)
        prim_path = f'/World/{root}/{name}'
        UsdGeom.Xform.Define(world, prim_path).GetPrim().GetPayloads().AddPayload(
            './' + path.name, '/Tile'
        )
        entries.append({'path': prim_path, 'payload': path.name})
    UsdGeom.Xform.Define(world, '/World/PointRegions')
    world.GetRootLayer().Save()
    write_json(output / 'authored-index.json', entries)


def facade():
    return Facade(
        (7, 9),
        np.array([0.0, 1.0, 0.0]),
        0.0,
        np.array([0.0, 0.0]),
        np.array([2.0, 3.0]),
        np.array([11, 12, 13], dtype=np.int64),
        1.0,
    )


@pytest.mark.parametrize('angle', [0.0, 0.63])
def test_closed_positive_volume_rotated_box_and_kernel_mesh(tmp_path, angle):
    source_world(tmp_path, [])
    c, s = np.cos(angle), np.sin(angle)
    axes = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    model = Cuboid(np.array([23.0, -7.0, 5.0]), np.array([4.0, 2.0, 6.0]), axes, (0,), 'building')
    assert publish(tmp_path, [facade()], [model], BuildingConfig(), METADATA) == (0, 0)
    folder = tmp_path / 'building-models'
    shape = read_brep(folder / 'model_0000.brep')
    assert shape.ShapeType() == TopAbs_SOLID
    assert BRepCheck_Analyzer(shape).IsValid()
    shell = TopExp_Explorer(shape, TopAbs_SHELL)
    assert shell.More()
    assert BRepCheck_Shell(TopoDS.Shell_s(shell.Current())).Closed() == BRepCheck_NoError
    assert len(faces(shape)) == 6
    properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, properties)
    assert properties.Mass() == pytest.approx(48)
    center = properties.CentreOfMass()
    np.testing.assert_allclose([center.X(), center.Y(), center.Z()], model.center, atol=1e-9)
    explorer = TopExp_Explorer(read_brep(folder / 'model.brep'), TopAbs_SOLID)
    assert explorer.More()
    explorer.Next()
    assert not explorer.More()
    positions, triangles = tessellate(shape)
    local = (positions - model.center) @ axes
    np.testing.assert_allclose(np.abs(local), np.tile(model.size / 2, (len(local), 1)))
    stage = Usd.Stage.Open(str(tmp_path / 'buildings.usda'))
    mesh = UsdGeom.Mesh.Get(stage, '/World/BuildingModels/model_0000/Surface')
    np.testing.assert_array_equal(mesh.GetPointsAttr().Get(), positions.astype(np.float32))
    np.testing.assert_array_equal(
        np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3), triangles
    )
    prim = mesh.GetPrim()
    assert not prim.HasAttribute('source:recordIds')
    assert prim.GetAttribute('building:kind').Get() == 'building'
    assert (
        Path(prim.GetAttribute('building:authority').Get().resolvedPath).resolve()
        == (folder / 'model_0000.brep').resolve()
    )
    assert UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
    assert UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == 'convexHull'
    with np.load(folder / 'model_0000.npz') as data:
        assert set(data.files) == {
            'center',
            'size',
            'axes',
            'facade_indices',
            'source_planes',
            'support_ids',
        }
        np.testing.assert_array_equal(data['axes'], axes)
        np.testing.assert_array_equal(data['source_planes'], [7, 9])
        np.testing.assert_array_equal(data['support_ids'], [11, 12, 13])
    row = json.loads((folder / 'models.json').read_text())[0]
    assert row['volume_m3'] == 48
    assert row['support_records'] == 3
    assert row['source_planes'] == [7, 9]
    exchanged = read_step(folder / 'model.step')
    assert TopExp_Explorer(exchanged, TopAbs_SOLID).More()
    step_properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(exchanged, step_properties)
    assert step_properties.Mass() == pytest.approx(48)


def test_selective_wall_removal_preserves_ground_unrelated_walls_and_source_bytes(tmp_path):
    positions = np.array(
        [
            [0, 0, 0],
            [2, 0, 0],
            [2, 0, 3],
            [0, 0, 3],
            [3, 0, 0],
            [3, 0, 3],
            [0, 2, 0],
            [2, 2, 0],
            [0, 0.3, 0],
            [2, 0.3, 0],
            [2, 0.3, 3],
        ],
        dtype=np.float32,
    )
    triangles = np.array(
        [[0, 1, 2], [0, 2, 3], [1, 4, 5], [0, 1, 6], [1, 7, 6], [8, 9, 10]],
        dtype=np.int32,
    )
    source_world(
        tmp_path,
        [
            ('Regions', 'mixed', positions, triangles),
            ('Regions', 'unrelated', positions + [20, 0, 0], triangles),
            ('Regions', 'fully_replaced', positions[:4], triangles[:2]),
            ('Ground', 'original', positions, triangles[3:5]),
        ],
    )
    originals = {path: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
    model = Cuboid(
        np.array([1.0, -0.15, 1.5]), np.array([2.0, 0.3, 3.0]), np.eye(3), (0,), 'facade'
    )
    assert publish(tmp_path, [facade()], [model], BuildingConfig(), METADATA) == (10, 4)
    for path, original in originals.items():
        assert path.read_bytes() == original
    stage = Usd.Stage.Open(str(tmp_path / 'buildings.usda'))
    assert stage.GetPrimAtPath('/World/Ground').IsActive()
    assert stage.GetPrimAtPath('/World/Ground/original').IsActive()
    assert stage.GetPrimAtPath('/World/Regions').IsActive()
    assert stage.GetPrimAtPath('/World/Regions/unrelated').IsActive()
    assert not stage.GetPrimAtPath('/World/Regions/mixed').IsActive()
    assert not stage.GetPrimAtPath('/World/Regions/fully_replaced').IsActive()
    assert not stage.GetPrimAtPath('/World/BuildingResidual/fully_replaced')
    assert (
        UsdGeom.Imageable(stage.GetPrimAtPath('/World/PointRegions')).GetVisibilityAttr().Get()
        == 'invisible'
    )
    source = Usd.Stage.Open(str(tmp_path / 'Regions_mixed.usdc'))
    original_mesh = UsdGeom.Mesh.Get(source, '/Tile/Surface')
    residual = UsdGeom.Mesh.Get(stage, '/World/BuildingResidual/mixed/Surface')
    np.testing.assert_array_equal(residual.GetPointsAttr().Get(), positions)
    np.testing.assert_array_equal(
        np.asarray(residual.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3), triangles[2:]
    )
    np.testing.assert_array_equal(
        residual.GetPrim().GetAttribute('source:recordIds').Get(),
        original_mesh.GetPrim().GetAttribute('source:recordIds').Get(),
    )
    np.testing.assert_array_equal(
        residual.GetDisplayColorPrimvar().Get(), original_mesh.GetDisplayColorPrimvar().Get()
    )
    assert UsdPhysics.MeshCollisionAPI(residual.GetPrim()).GetApproximationAttr().Get() == 'none'
