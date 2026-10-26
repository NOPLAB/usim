"""Audit fitted geometry against unchanged source authorities and actual USD collisions."""

from __future__ import annotations

import json
from pathlib import Path
from enum import Enum
from typing import TYPE_CHECKING, assert_never

import numpy as np
from scipy.spatial import cKDTree

from .models import PipelineInputError
from .pcd import open_records, sha256, xyz
from .simulation_metrics import signed_volume
from .storage import write_json

if TYPE_CHECKING:
    from pxr import Usd


class SurfaceKind(Enum):
    PLANE = 'plane'
    CURVE = 'curve'


def verify_ground(output: Path, stage: Usd.Stage) -> None:
    """Check the inherited floor's vertices, topology and actual collision."""
    from pxr import UsdGeom, UsdPhysics

    ground = [
        UsdGeom.Mesh(p)
        for p in stage.Traverse()
        if p.IsA(UsdGeom.Mesh) and p.GetPath().HasPrefix('/World/SimulationGround')
    ]
    with np.load(output / 'simulation-models/terrain.npz') as terrain:
        if len(ground) != 1 or not np.array_equal(
            ground[0].GetPointsAttr().Get(), terrain['positions']
        ):
            raise PipelineInputError('The one-mesh continuous floor changed')
        if not np.array_equal(
            np.asarray(ground[0].GetFaceVertexIndicesAttr().Get()).reshape(-1, 3),
            terrain['faces'],
        ) or not (
            UsdPhysics.CollisionAPI(ground[0].GetPrim()).GetCollisionEnabledAttr().Get()
            and UsdPhysics.MeshCollisionAPI(ground[0].GetPrim()).GetApproximationAttr().Get()
            == 'none'
        ):
            raise PipelineInputError('Continuous floor topology or triangle collision changed')


def verify(output: Path) -> None:
    """Check baseline preservation, geometric support, source coverage and collision equality."""
    import open3d as o3d
    from pxr import Usd, UsdGeom, UsdPhysics

    from .simulation_polygon_slab import polygon_slab

    folder = output / 'geometry-fit'
    report = json.loads((folder / 'report.json').read_text())
    if sha256(output / 'simulation.usda') != report['baseline_sha256']:
        raise PipelineInputError('Baseline simulation layer changed during geometric fitting')
    for relative, fingerprint in json.loads(
        (output / 'simulation-models/source-hashes.json').read_text()
    ).items():
        if sha256(output / relative) != fingerprint:
            raise PipelineInputError(f'Original authority changed: {relative}')
    stage = Usd.Stage.Open(str(output / 'simulation-fitted.usda'), Usd.Stage.LoadNone)
    for root in (
        'SimulationObstacles',
        'SimulationPlanes',
        'SimulationUnresolved',
        'BuildingModels',
    ):
        if stage.GetPrimAtPath('/World/' + root).IsActive():
            raise PipelineInputError('Unfitted duplicate geometry remains active')
    stage.Load('/World/SimulationGround')
    verify_ground(output, stage)
    stage.Unload('/World/SimulationGround')
    source_points = np.empty((0, 3), np.float32)
    if report['measured_facades']:
        manifest = json.loads((output / 'manifest.json').read_text())
        original_source = manifest['sources'][manifest['primary_source']]
        if sha256(output / original_source['path']) != original_source['sha256']:
            raise PipelineInputError('Measured facade point source changed')
        records, _ = open_records(output / original_source['path'])
        source_points = xyz(records)
    with np.load(output / 'simulation-models/obstacles.npz') as data:
        centers = (data['keys'] + 0.5) * float(report['cell_m'])
    tree = cKDTree(centers)
    scene = o3d.t.geometry.RaycastingScene()
    faces_count = 0
    maximum_support = 0.0
    for row in json.loads((folder / 'index.json').read_text()):
        stage.Load(row['path'])
        mesh = UsdGeom.Mesh.Get(stage, row['path'] + '/Surface')
        prim = mesh.GetPrim()
        actual_positions = np.asarray(mesh.GetPointsAttr().Get())
        actual_faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        with np.load((output / row['payload']).with_suffix('.npz')) as saved:
            positions, faces = saved['positions'], saved['faces']
        if not (
            np.array_equal(actual_positions, positions)
            and np.array_equal(actual_faces, faces)
            and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            and UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == 'none'
        ):
            raise PipelineInputError('Fitted render/collision geometry differs from its authority')
        triangles = positions[faces].astype(np.float64)
        area = np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
        )
        if not np.isfinite(positions).all() or (area <= 0).any():
            raise PipelineInputError(
                f'Fitted surface contains invalid triangles: {row["payload"]}'
            )
        try:
            kind = SurfaceKind(row['kind'])
        except ValueError as error:
            raise PipelineInputError('Unknown fitted surface kind') from error
        match kind:
            case SurfaceKind.PLANE:
                source = output / row['authority']
                if sha256(source) != row['authority_sha256']:
                    raise PipelineInputError('Measured planar authority changed')
                with np.load(source) as authority:
                    if 'source_facade' in row:
                        original_facade = output / row['source_facade']
                        if sha256(original_facade) != row['source_facade_sha256']:
                            raise PipelineInputError('Original facade support changed')
                        with np.load(original_facade) as measured:
                            support = source_points[measured['support_ids']].astype(np.float64)
                        local = np.column_stack(
                            (support @ authority['u'], support @ authority['v'])
                        )
                        measured_keys = np.unique(
                            np.floor(local / float(report['cell_m'])).astype(np.int64), axis=0
                        )
                        if not np.array_equal(measured_keys, authority['keys']):
                            raise PipelineInputError(
                                'Facade filled cells without measured support'
                            )
                        projected = np.column_stack(
                            (positions @ authority['u'], positions @ authority['v'])
                        )
                        samples = np.concatenate(
                            (
                                projected,
                                projected[faces].mean(1),
                                (projected[faces[:, 0]] + projected[faces[:, 1]]) / 2,
                                (projected[faces[:, 1]] + projected[faces[:, 2]]) / 2,
                                (projected[faces[:, 2]] + projected[faces[:, 0]]) / 2,
                            )
                        )
                        if (cKDTree(local).query(samples, workers=4)[0] > 0.3).any():
                            raise PipelineInputError('Published facade spans unsupported space')
                    slab = polygon_slab(authority['keys'], float(report['cell_m']), 0.16)
                    expected = (
                        slab.positions[:, :1] * authority['u']
                        + slab.positions[:, 1:2] * authority['v']
                        + (slab.positions[:, 2:3] - float(authority['offset']))
                        * authority['normal']
                    ).astype(np.float32)
                if not np.array_equal(expected, positions) or not np.array_equal(
                    slab.faces, faces
                ):
                    raise PipelineInputError('Straight outline differs from supported fitting')
                if not np.isclose(signed_volume(positions, faces), slab.area_m2 * 0.16, rtol=1e-3):
                    raise PipelineInputError('Fitted planar slab lost closure or positive volume')
            case SurfaceKind.CURVE:
                distance = tree.query(positions, workers=4)[0]
                maximum_support = max(maximum_support, float(distance.max()))
                if (distance > 0.3).any():
                    raise PipelineInputError(
                        f'Curved envelope extends beyond supported fitting: '
                        f'{row["payload"]}; maximum={distance.max()} m'
                    )
                scene.add_triangles(
                    o3d.core.Tensor(positions, dtype=o3d.core.Dtype.Float32),
                    o3d.core.Tensor(faces.astype(np.uint32), dtype=o3d.core.Dtype.UInt32),
                )
            case unreachable:
                assert_never(unreachable)
        faces_count += len(faces)
        stage.Unload(row['path'])
        del mesh, prim, actual_positions, actual_faces, positions, faces, triangles, area
    outside = 0
    maximum_outside = 0.0
    for start in range(0, len(centers), 100000):
        distance = scene.compute_signed_distance(
            o3d.core.Tensor(centers[start : start + 100000].astype(np.float32)),
            nthreads=4,
            nsamples=3,
        ).numpy()
        outside += int((distance > 1e-4).sum())
        maximum_outside = max(maximum_outside, float(distance.max()))
    if outside:
        raise PipelineInputError(
            f'Curved envelopes failed to represent {outside} occupied source cells; error={maximum_outside}'
        )
    result = {
        'baseline_unchanged': True,
        'original_authorities_unchanged': True,
        'ground_meshes': 1,
        'fitted_faces': faces_count,
        'curve_surface_support_max_m': maximum_support,
        'source_cells_outside_envelopes': outside,
        'render_collision_identical': True,
        'inferred_building_walls_active': False,
        'measured_facades': report['measured_facades'],
    }
    write_json(folder / 'verification.json', result)
    print('GEOMETRY_FIT_VERIFIED ' + json.dumps(result), flush=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', type=Path, required=True)
    parser.add_argument('--ground-only', action='store_true')
    args = parser.parse_args()
    if args.ground_only:
        from pxr import Usd

        stage = Usd.Stage.Open(str(args.map / 'simulation-fitted.usda'), Usd.Stage.LoadNone)
        stage.Load('/World/SimulationGround')
        verify_ground(args.map, stage)
        print('GEOMETRY_FIT_GROUND_VERIFIED', flush=True)
    else:
        verify(args.map)


if __name__ == '__main__':
    main()
