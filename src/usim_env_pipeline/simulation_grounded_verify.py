"""Audit filtered geometry against original faces and actual USD triangle collisions."""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .models import PipelineInputError
from .pcd import sha256
from .storage import write_json


def verify(output: Path, *, angular: bool = False, angular_name: str = 'angular') -> None:
    """Check exact face subsets, immutable sources, and the inherited one-mesh floor."""
    from pxr import Usd, UsdGeom, UsdPhysics

    from .simulation_fit_verify import verify_ground
    from .simulation_ground_connectivity import native_intersection
    from .simulation_grounded_output import AttributeInterpolation, subset_attribute

    folder = output / (f'{angular_name}-grounded-models' if angular else 'grounded-models')
    report = json.loads((folder / 'report.json').read_text(encoding='utf-8'))
    for path, expected in report['source_hashes'].items():
        if sha256(Path(path)) != expected:
            raise PipelineInputError(f'Original floating-filter authority changed: {path}')
    result_file = (
        f'simulation-{angular_name}-grounded.usda' if angular else 'simulation-grounded.usda'
    )
    source_file = f'simulation-{angular_name}.usda' if angular else 'simulation-fitted.usda'
    stage = Usd.Stage.Open(str(output / result_file), Usd.Stage.LoadNone)
    stage.Load('/World/SimulationGround')
    verify_ground(output, stage)
    stage.Unload('/World/SimulationGround')
    originals = Usd.Stage.Open(str(output / source_file), Usd.Stage.LoadNone)
    assets = json.loads((folder / 'sources.json').read_text(encoding='utf-8'))
    asset_by_path = {asset['path']: asset for asset in assets}

    @lru_cache(maxsize=2)
    def load_original(
        index: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.int64]]:
        asset = assets[index]
        originals.Load(asset['path'])
        mesh = UsdGeom.Mesh.Get(originals, asset['path'] + '/Surface')
        matrix = np.array(
            UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim()), np.float64
        )
        if not np.array_equal(matrix, asset['matrix']):
            raise PipelineInputError('Cached world transform differs from the original geometry')
        originals.Unload(asset['path'])
        with np.load(output / asset['archive']) as data:
            points, faces = data['positions'], data['faces']
        points = points.astype(np.float64) @ matrix[:3, :3] + matrix[3, :3]
        order = np.load(output / asset['order'], mmap_mode='r')
        return points, faces, order

    with np.load(folder / 'connections.npz') as graph:
        keep, merged, witnesses = graph['keep'], graph['merged'], graph['witnesses']
        chunk_owners, chunk_assets = graph['chunk_owners'], graph['chunk_assets']
        starts, stops = graph['chunk_starts'], graph['chunk_stops']
        if not keep[merged[0]]:
            raise PipelineInputError('The floor component was not retained')
        reached = {int(merged[0])}
        for first, second in witnesses:
            source, target = int(chunk_owners[first]), int(chunk_owners[second])
            if source not in reached or target in reached or not keep[target]:
                raise PipelineInputError('Contact witnesses do not form a floor-rooted forest')
            pair = []
            for chunk in (first, second):
                points, faces, order = load_original(int(chunk_assets[chunk]))
                pair.append((points, faces[order[starts[chunk] : stops[chunk]]]))
            if not native_intersection(pair[0], pair[1]):
                raise PipelineInputError(
                    'A claimed grounding contact has no triangle intersection'
                )
            reached.add(target)
        if reached != set(np.flatnonzero(keep).tolist()):
            raise PipelineInputError('A retained component lacks a floor-connected witness')
    rows = json.loads((folder / 'index.json').read_text(encoding='utf-8'))
    retained, removed = 0, 0
    for row in rows:
        mask = np.load(output / row['mask'])
        labels = np.load(output / row['labels'], mmap_mode='r')
        if not np.array_equal(mask, keep[merged[labels]]):
            raise PipelineInputError('Published mask differs from whole-component reachability')
        if len(mask) != row['source_faces'] or int(mask.sum()) != row['kept_faces']:
            raise PipelineInputError('Filtered triangle accounting is inconsistent')
        with np.load((output / row['source_payload']).with_suffix('.npz')) as original:
            positions, faces = original['positions'], original['faces']
        if not mask.any():
            if stage.GetPrimAtPath(row['path']).IsActive():
                raise PipelineInputError('A floating model remains active')
        else:
            stage.Load(row['path'])
            mesh = UsdGeom.Mesh.Get(stage, row['path'] + '/Surface')
            actual_positions = np.asarray(mesh.GetPointsAttr().Get())
            actual_faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
            if len(actual_faces) != int(mask.sum()):
                raise PipelineInputError('Published retained face count differs from the mask')
            expected_faces = faces[mask]
            for start in range(0, len(actual_faces), 100000):
                if not np.array_equal(
                    actual_positions[actual_faces[start : start + 100000]],
                    positions[expected_faces[start : start + 100000]],
                ):
                    raise PipelineInputError('Filtering changed retained triangle coordinates')
            prim = mesh.GetPrim()
            matrix = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(prim), np.float64)
            if not np.array_equal(matrix, asset_by_path[row['path']]['matrix']):
                raise PipelineInputError('Filtering changed a retained world transform')
            used = np.arange(len(positions)) if mask.all() else np.unique(expected_faces)
            with np.load((output / row['labels']).with_suffix('.appearance.npz')) as attributes:
                normals_type = str(attributes['normals_interpolation']) if angular else 'vertex'
                colors_type = str(attributes['colors_interpolation']) if angular else 'vertex'
                if not (
                    np.array_equal(
                        mesh.GetNormalsAttr().Get(),
                        subset_attribute(
                            attributes['normals'], AttributeInterpolation(normals_type), used, mask
                        ),
                    )
                    and np.array_equal(
                        mesh.GetDisplayColorPrimvar().ComputeFlattened(),
                        subset_attribute(
                            attributes['colors'], AttributeInterpolation(colors_type), used, mask
                        ),
                    )
                    and mesh.GetNormalsInterpolation() == normals_type
                    and mesh.GetDisplayColorPrimvar().GetInterpolation() == colors_type
                ):
                    raise PipelineInputError('Filtering changed retained normals or colors')
            if not (
                UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                and UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == 'none'
            ):
                raise PipelineInputError('Filtered render and collision geometry differ')
            stage.Unload(row['path'])
        retained += int(mask.sum())
        removed += int((~mask).sum())
    if retained != report['retained_faces'] or removed != report['removed_faces']:
        raise PipelineInputError('Global removal totals differ from published geometry')
    write_json(
        folder / 'verification.json',
        {
            'source_files_unchanged': True,
            'ground_meshes': 1,
            'retained_faces': retained,
            'removed_faces': removed,
            'retained_coordinates_unchanged': True,
            'retained_appearance_unchanged': True,
            'retained_transforms_unchanged': True,
            'render_collision_identical': True,
            'whole_components_only': True,
            'floor_connected_witnesses': len(witnesses),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', type=Path, required=True)
    parser.add_argument('--angular', action='store_true')
    parser.add_argument('--angular-name', default='angular')
    args = parser.parse_args()
    verify(args.map.resolve(), angular=args.angular, angular_name=args.angular_name)


if __name__ == '__main__':
    main()
