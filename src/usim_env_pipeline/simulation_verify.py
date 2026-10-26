"""Independently validate supported simulation geometry and immutable source archives."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .models import PipelineInputError
from .pcd import open_records, sha256, xyz
from .simulation_ground import GroundConfig, Terrain, terrain_heights
from .simulation_obstacles import ObstacleConfig
from .simulation_verify_cleanup import verify_cleanup
from .simulation_verify_faces import verify_faces
from .simulation_verify_ground import verify_ground
from .simulation_verify_obstacles import verify_obstacles
from .simulation_verify_planes import verify_planes
from .simulation_surface_smoothing import smooth_surface
from .simulation_metrics import signed_volume
from .storage import write_json


def verify(output: Path) -> None:
    """Audit canonical support/volume, source partition and composed collision geometry."""
    from pxr import Usd, UsdGeom, UsdPhysics
    from scipy.spatial import cKDTree

    folder = output / 'simulation-models'
    report = json.loads((output / 'simulation-report.json').read_text(encoding='utf-8'))
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    source = manifest['sources'][manifest['primary_source']]
    for path, expected in json.loads((folder / 'source-hashes.json').read_text()).items():
        if sha256(output / path) != expected:
            raise PipelineInputError(f'Archived source changed: {path}')
    if source['sha256'] != report['source_sha256']:
        raise PipelineInputError('Simulation provenance does not match original source')
    records, _ = open_records(output / source['path'])
    points = xyz(records)
    config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    ground_config = GroundConfig(**config['ground'])
    obstacle_config = ObstacleConfig(**config['obstacles'])
    with np.load(folder / 'terrain.npz') as archive:
        terrain = Terrain(
            archive['positions'],
            archive['faces'],
            archive['keys'],
            archive['cells'],
            archive['corner_indices'],
            archive['support_ids'],
            float(archive['smoothing_strength']),
        )
        measured_ids = archive['measured_ground_ids']
    from .simulation_map import ground_support

    if not np.array_equal(measured_ids, ground_support(output)):
        raise PipelineInputError('Ground support is not the authoritative measured floor')
    ground_audit = verify_ground(terrain, points, measured_ids, ground_config)
    if (
        terrain.smoothing_strength != report['ground_smoothing_strength']
        or ground_audit.inferred_nodes != report['ground_inferred_nodes']
    ):
        raise PipelineInputError('Ground inferred-node or smoothing metadata changed')
    with np.load(folder / 'source-partition.npz') as archive:
        labels = archive['labels']
    if len(labels) != len(points) or (labels > 5).any():
        raise PipelineInputError('Simulation source partition is incomplete or invalid')
    if not np.isin(labels[measured_ids], [1, 2]).all():
        raise PipelineInputError('Authoritative ground was converted back into obstacles or noise')
    counts = np.bincount(labels, minlength=6)
    for label, name in enumerate(
        (
            'unresolved_records',
            'ground_records',
            'building_records',
            'obstacle_records',
            'planar_records',
            'noise_records',
        )
    ):
        if int(counts[label]) != report[name]:
            raise PipelineInputError(f'Source partition count changed: {name}')
    removed_voxels = verify_cleanup(output, points, obstacle_config)
    if removed_voxels != report['removed_noise_voxels']:
        raise PipelineInputError('Removed noise voxel count changed')
    ground_ids = np.flatnonzero(labels == 1)
    measured_floor = cKDTree(points[measured_ids, :2])
    maximum_ground_error = 0.0
    for start in range(0, len(ground_ids), 250000):
        selected = ground_ids[start : start + 250000]
        heights = terrain_heights(points[selected, :2], terrain, ground_config)
        errors = np.abs(points[selected, 2] - heights)
        original_distance, neighbor = measured_floor.query(points[selected, :2], workers=4)
        original_error = np.abs(points[selected, 2] - points[measured_ids[neighbor], 2])
        if (
            not np.isfinite(errors).all()
            or (
                (errors > obstacle_config.ground_tolerance + 1e-5)
                & (
                    (original_distance > ground_config.nearest_support + 1e-5)
                    | (original_error > obstacle_config.ground_tolerance + 1e-5)
                )
            ).any()
        ):
            raise PipelineInputError('Source points classified as ground lack surface support')
        maximum_ground_error = max(maximum_ground_error, float(errors.max(initial=0)))
    building_ids = np.flatnonzero(labels == 2)
    claimed = np.zeros(len(building_ids), dtype=bool)
    for row in json.loads((output / 'building-models/models.json').read_text()):
        center, size, axes = map(np.asarray, (row['center'], row['size'], row['axes']))
        signed = np.abs((points[building_ids] - center) @ axes) - size / 2
        claimed |= (
            np.linalg.norm(np.maximum(signed, 0), axis=1) <= obstacle_config.building_margin + 1e-5
        )
    if not claimed.all():
        raise PipelineInputError('Building source partition extends beyond bounded models')
    obstacle_mesh = verify_obstacles(output, points, obstacle_config)
    positions, faces = obstacle_mesh.positions, obstacle_mesh.faces
    stage = Usd.Stage.Open(str(output / report['world']))
    ground_meshes = [
        prim
        for prim in stage.Traverse()
        if prim.GetPath().HasPrefix('/World/SimulationGround') and prim.IsA(UsdGeom.Mesh)
    ]
    if len(ground_meshes) != 1:
        raise PipelineInputError('Simulation ground must be authored as exactly one mesh')
    ground_mesh = UsdGeom.Mesh(ground_meshes[0])
    normals = np.asarray(ground_mesh.GetNormalsAttr().Get())
    if (
        ground_mesh.GetNormalsInterpolation() != UsdGeom.Tokens.vertex
        or normals.shape != terrain.positions.shape
        or not np.isfinite(normals).all()
        or not np.allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-5)
        or (normals[:, 2] <= 0).any()
    ):
        raise PipelineInputError('Continuous ground lacks smooth upward vertex normals')
    plane_ids, plane_faces = verify_planes(output, stage, points, obstacle_config.cell)
    if not np.array_equal(plane_ids, np.flatnonzero(labels == 4)):
        raise PipelineInputError('Residual slab record partition disagrees with source ownership')
    for name in (
        'Regions',
        'PointRegions',
        'Ground',
        'BuildingResidual',
        'InferredStructures',
        'Vegetation',
    ):
        if stage.GetPrimAtPath('/World/' + name).IsActive():
            raise PipelineInputError(f'Duplicate archival geometry/collision is active: {name}')
    if not stage.GetPrimAtPath('/World/BuildingModels').IsActive():
        raise PipelineInputError('Independent closed building models disappeared')
    original_faces, replaced_faces, remaining_faces, noise_faces = verify_faces(
        output,
        stage,
        np.flatnonzero(labels == 5),
        np.union1d(measured_ids, ground_ids),
    )
    if (original_faces, replaced_faces, remaining_faces, noise_faces) != (
        report['source_triangles'],
        report['replaced_source_triangles'],
        report['retained_source_triangles'],
        report['removed_noise_triangles'],
    ):
        raise PipelineInputError('Source face accounting counts disagree')
    actual_faces = {
        'ground': 0,
        'obstacles': 0,
        'unresolved': remaining_faces,
        'planes': plane_faces,
    }
    canonical = {'ground': (terrain.positions, terrain.faces), 'obstacles': (positions, faces)}
    smoothed_volume = 0.0
    for row in json.loads((folder / 'index.json').read_text()):
        mesh = UsdGeom.Mesh.Get(stage, row['path'] + '/Surface')
        prim = mesh.GetPrim()
        kind = row['kind']
        if kind in ('unresolved', 'planes'):
            continue
        expected_positions, expected_faces = canonical[kind]
        with np.load((output / row['payload']).with_suffix('.npz')) as indices:
            used, owned = indices['vertex_indices'], indices['face_indices']
        rendered = np.asarray(mesh.GetPointsAttr().Get())
        rendered_faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        expected_local_positions = expected_positions[used]
        smoothing_limit = float(prim.GetAttribute('simulation:smoothingLimit').Get())
        if kind == 'obstacles':
            if not np.isclose(smoothing_limit, 0.2 * obstacle_config.cell):
                raise PipelineInputError('Obstacle smoothing bound changed')
            expected_local_positions = smooth_surface(
                expected_local_positions, rendered_faces, obstacle_config.cell
            )
            distance = np.linalg.norm(rendered - expected_positions[used], axis=1)
            if (distance > smoothing_limit + 1e-6).any():
                raise PipelineInputError('Obstacle approximation exceeds its displacement bound')
            smoothed_volume += signed_volume(rendered, rendered_faces)
        if not (
            np.array_equal(rendered, expected_local_positions)
            and np.array_equal(used[rendered_faces], expected_faces[owned])
            and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            and UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == 'none'
            and not prim.HasAttribute('source:recordIds')
            and prim.GetAttribute('simulation:kind').Get() == kind
        ):
            raise PipelineInputError(
                f'Composed simulation geometry/collision mismatch: {row["path"]}'
            )
        actual_faces[kind] += len(rendered_faces)
    if actual_faces != {
        'ground': len(terrain.faces),
        'obstacles': len(faces),
        'unresolved': report['retained_source_triangles'],
        'planes': plane_faces,
    }:
        raise PipelineInputError('Tile partition lost or duplicated generated faces')
    result = {
        'source_unchanged': True,
        'source_partition_complete': True,
        'ground_area_m2': ground_audit.area_m2,
        'ground_source_error_max_m': maximum_ground_error,
        'ground_slope_max': ground_audit.slope_max,
        'ground_center_support_max_m': ground_audit.center_support_max_m,
        'ground_inferred_nodes': ground_audit.inferred_nodes,
        'ground_boundary_edges': ground_audit.boundary_edges,
        'ground_anchor_error_max_m': ground_audit.anchor_error_max_m,
        'ground_smoothing_strength': terrain.smoothing_strength,
        'ground_meshes': 1,
        'ground_components': 1,
        'ground_interior_holes': 0,
        'obstacle_volume_m3': obstacle_mesh.volume_m3,
        'smoothed_obstacle_volume_m3': smoothed_volume,
        'obstacle_open_edges': 0,
        'obstacle_edge_contacts': obstacle_mesh.edge_contacts,
        'original_mesh_active': False,
        'source_faces_accounted': original_faces,
        'retained_source_faces': remaining_faces,
        'residual_plane_faces': plane_faces,
        'noise_records_removed': int(counts[5]),
        'noise_voxels_removed': removed_voxels,
        'noise_source_faces_removed': noise_faces,
    }
    write_json(folder / 'verification.json', result)
    print('SIMULATION_VERIFICATION_PASSED ' + json.dumps(result), flush=True)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        verify(args.map)
    except PipelineInputError as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
