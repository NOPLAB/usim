"""The fitted presentation keeps baseline authority while replacing voxel rendering."""

import json

import numpy as np
import pytest

pytest.importorskip('shapely')
pytest.importorskip('skimage')
pytest.importorskip('pxr')
pytest.importorskip('open3d')

import open3d as o3d
from pxr import Usd, UsdGeom, UsdPhysics, Vt

from usim_env_pipeline.pcd import DTYPE, sha256
from usim_env_pipeline.simulation_fit import fit
from usim_env_pipeline.simulation_fit_republish import republish
from usim_env_pipeline.usd import LayerMetadata, stage_base
from test.env_pipeline_fixtures import write_pcd


@pytest.mark.parametrize('refit_curves', [False, True])
def test_fitted_scene_preserves_baseline_floor_and_uses_actual_triangle_collision(
    tmp_path, refit_curves
):
    # Given an immutable baseline floor and a measured irregular occupied object.
    metadata = LayerMetadata('test', 'test', 'test', 'Z metres', 'test')
    stage = stage_base(tmp_path / 'simulation.usda', '/World', metadata)
    floor = UsdGeom.Mesh.Define(stage, '/World/SimulationGround/Surface')
    positions = np.array([[0, 0, 0], [5, 0, 0], [5, 5, 0]], np.float32)
    floor.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(positions))
    floor.CreateFaceVertexCountsAttr([3])
    floor.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdGeom.Xform.Define(stage, '/World/SimulationObstacles')
    wall = UsdGeom.Mesh.Define(stage, '/World/BuildingModels/Rectangle/Surface')
    wall.CreatePointsAttr(
        Vt.Vec3fArray.FromNumpy(
            np.array([[0, 0, 1], [0, 4.2, 1], [0, 4.2, 3.2], [0, 0, 3.2]], np.float32)
        )
    )
    wall.CreateFaceVertexCountsAttr([3, 3])
    wall.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
    UsdPhysics.CollisionAPI.Apply(wall.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(wall.GetPrim()).CreateApproximationAttr('none')
    stage.GetRootLayer().Save()
    before = sha256(tmp_path / 'simulation.usda')
    model = tmp_path / 'simulation-models'
    (model / 'planes').mkdir(parents=True)
    y, z = np.meshgrid(
        np.r_[np.arange(0.05, 1.06, 0.1), np.arange(3.05, 4.06, 0.1)],
        np.arange(1.05, 3.06, 0.1),
    )
    source_points = np.column_stack((np.zeros(y.size), y.ravel(), z.ravel()))
    records = np.zeros(len(source_points), dtype=DTYPE)
    for axis, name in enumerate(('x', 'y', 'z')):
        records[name] = source_points[:, axis]
    write_pcd(tmp_path / 'source.pcd', records)
    facades = tmp_path / 'building-models/facades'
    facades.mkdir(parents=True)
    np.savez_compressed(
        facades / 'facade_0000.npz',
        normal=[1.0, 0, 0],
        offset=0,
        support_ids=np.arange(len(records), dtype=np.int64),
    )
    (facades.parent / 'facades.json').write_text(json.dumps([{'name': 'facade_0000'}]))
    (tmp_path / 'manifest.json').write_text(
        json.dumps(
            {
                'primary_source': 'source',
                'sources': {
                    'source': {
                        'license': 'test',
                        'attribution': 'test',
                        'path': 'source.pcd',
                        'sha256': sha256(tmp_path / 'source.pcd'),
                    }
                },
                'source_catalog': 'test',
                'frame': 'Z metres',
            }
        )
    )
    (model / 'config.json').write_text(json.dumps({'obstacles': {'cell': 0.1}}))
    (model / 'planes/index.json').write_text('[]')
    keys = (np.indices((4, 4, 4)).reshape(3, -1).T + 10).astype(np.int64)
    np.savez_compressed(model / 'obstacles.npz', keys=keys)
    # When producing the straight/curved presentation as a separate layer.
    fit(tmp_path)
    if refit_curves:
        # Interrupted derived cache writes must not block source-only regeneration.
        for archive in (tmp_path / 'geometry-fit/curves').glob('*.npz'):
            archive.write_bytes(b'incomplete derived cache')
    republish(tmp_path, refit_curves=refit_curves)
    fitted = Usd.Stage.Open(str(tmp_path / 'simulation-fitted.usda'))
    # Then the original file and floor survive, while curved rendering has identical collision.
    assert sha256(tmp_path / 'simulation.usda') == before
    actual_floor = UsdGeom.Mesh.Get(fitted, '/World/SimulationGround/Surface')
    np.testing.assert_array_equal(actual_floor.GetPointsAttr().Get(), positions)
    assert not fitted.GetPrimAtPath('/World/SimulationObstacles').IsActive()
    curves = [
        p
        for p in fitted.Traverse()
        if p.IsA(UsdGeom.Mesh) and p.GetPath().HasPrefix('/World/FittedCurves')
    ]
    assert curves
    for prim in curves:
        assert UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        assert UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == 'none'
        assert UsdGeom.Mesh(prim).GetNormalsInterpolation() == 'vertex'
    # The measured wall stays collidable, while its unmeasured street opening stays empty.
    scene = o3d.t.geometry.RaycastingScene()
    for prim in fitted.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            scene.add_triangles(
                o3d.core.Tensor(
                    np.asarray(mesh.GetPointsAttr().Get()), dtype=o3d.core.Dtype.Float32
                ),
                o3d.core.Tensor(
                    np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
                    .reshape(-1, 3)
                    .astype(np.uint32),
                    dtype=o3d.core.Dtype.UInt32,
                ),
            )
    hits = scene.cast_rays(
        o3d.core.Tensor(
            [[-1, 2, 2, 1, 0, 0], [-1, 0.55, 2, 1, 0, 0]], dtype=o3d.core.Dtype.Float32
        )
    )['t_hit'].numpy()
    assert not np.isfinite(hits[0]), 'An unsupported rectangular wall still blocks the street'
    assert np.isfinite(hits[1]), 'Removing invented walls also removed measured wall support'
