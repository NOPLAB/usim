"""MIT tooling: complete, exact-source-vertex CPU ball-pivoted public-cloud map.

No estimated normals, downsampling, smoothing, projection, gap filling or decimation.
All original records survive in copied PCD and source-indexed regional NPZ archives.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from numpy.typing import NDArray

if not __package__:
    # Direct-file CLI and module CLI share the same local tooling package.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'usim_env_vtc'
from .pcd import PUBLIC_HASHES, normals, open_records, sha256, xyz

LICENSE = 'CC-BY-NC-SA-4.0'
ATTRIBUTION = (
    'fuRo; Yoshitaka Hara and Masahiro Tomono, Moving Object Removal and Surface Mesh '
    'Mapping for Path Planning on 3D Terrain, Advanced Robotics 34(6), 375-387, 2020; '
    'https://doi.org/10.1080/01691864.2020.1717375'
)
CATALOG = 'https://github.com/tsukubachallenge/tc-datasets'
SOURCE_URLS = {
    '2019': 'https://drive.google.com/file/d/1mH20dXpnBBlQ6hMKJZqdVhphrffsvWK_/view',
    '2018': 'https://drive.google.com/file/d/1c7Vd4vkMudAHyxc0ZOZCbTgx8ZFZ_Slx/view',
}
SUPPORT = {
    'radii': [0.12, 0.20, 0.35],
    'max_edge': 0.65,
    'min_face_normal_dot': 0.35,
    'min_pair_normal_dot': 0.0,
    'max_edge_normal_dot': 0.65,
    'midpoint_distance': 0.16,
    'centroid_distance': 0.18,
    'min_double_area': 1e-10,
}


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def tile_name(key: tuple[int, int]) -> str:
    return f't_{key[0] + 10000}_{key[1] + 10000}'


def stage_base(path: Path, root: str):
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateNew(str(path))
    prim = UsdGeom.Xform.Define(stage, root).GetPrim()
    stage.SetDefaultPrim(prim)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.GetRootLayer().customLayerData = {
        'license': LICENSE,
        'attribution': ATTRIBUTION,
        'sourceCatalog': CATALOG,
        'frame': 'raw 2019 XYZ; correspondence-based meter/Z-up established separately',
        'color': 'intensity grayscale, not photographic RGB',
    }
    return stage


def author_geometry(
    path: Path,
    positions: NDArray[np.float32],
    ids: NDArray[np.int64],
    intensity: NDArray[np.float32],
    faces: NDArray[np.int32] | None,
) -> None:
    from pxr import Sdf, UsdGeom, UsdPhysics, Vt

    stage = stage_base(path, '/Tile')
    if faces is None:
        geometry = UsdGeom.Points.Define(stage, '/Tile/Points')
        geometry.CreateWidthsAttr(Vt.FloatArray([0.055]))
        geometry.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        geometry.CreateIdsAttr(Vt.Int64Array.FromNumpy(ids.astype(np.int64)))
    else:
        geometry = UsdGeom.Mesh.Define(stage, '/Tile/Surface')
        geometry.CreateFaceVertexCountsAttr(
            Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32))
        )
        geometry.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.astype(np.int32).ravel()))
        geometry.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        geometry.CreateDoubleSidedAttr(True)
        geometry.GetPrim().CreateAttribute(
            'source:recordIds', Sdf.ValueTypeNames.Int64Array, custom=True
        ).Set(Vt.Int64Array.FromNumpy(ids.astype(np.int64)))
        UsdPhysics.CollisionAPI.Apply(geometry.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(geometry.GetPrim()).CreateApproximationAttr('none')
    geometry.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(positions.astype(np.float32)))
    geometry.CreateExtentAttr(
        Vt.Vec3fArray.FromNumpy(
            np.stack([positions.min(axis=0), positions.max(axis=0)]).astype(np.float32)
        )
    )
    gray = np.repeat((intensity.astype(np.float32) / 255)[:, None], 3, axis=1)
    geometry.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(Vt.Vec3fArray.FromNumpy(gray))
    stage.GetRootLayer().Save()


def region_job(
    job: tuple[str, tuple[int, int], float, float, set[tuple[int, int]]],
) -> dict[str, Any]:
    import open3d as o3d
    from scipy.spatial import cKDTree

    output, key, size, halo, occupied = job
    output = Path(output)
    name = tile_name(key)
    low = np.array(key) * size
    high = low + size
    own = np.load(output / 'records' / f'{name}.npz')
    own_records, own_ids = own['records'], own['ids']
    own_xyz = xyz(own_records)
    own_normals, own_valid = normals(own_records)
    author_geometry(
        output / 'points' / f'{name}.usdc', own_xyz, own_ids, own_records['intensity'], None
    )
    records, ids = [], []
    for x in range(key[0] - 1, key[0] + 2):
        for y in range(key[1] - 1, key[1] + 2):
            if (x, y) not in occupied:
                continue
            archive = np.load(output / 'records' / f'{tile_name((x, y))}.npz')
            positions = xyz(archive['records'])
            inside = ((positions[:, :2] >= low - halo) & (positions[:, :2] < high + halo)).all(
                axis=1
            )
            records.append(archive['records'][inside])
            ids.append(archive['ids'][inside])
    records, ids = np.concatenate(records), np.concatenate(ids)
    order = np.argsort(ids)
    records, ids = records[order], ids[order]
    positions = xyz(records)
    working_normals, valid = normals(records)
    # Supplied normals are unoriented plane directions. Choose a deterministic
    # hemisphere for pivoting only; archival normal fields remain byte-identical.
    dominant = np.argmax(np.abs(working_normals), axis=1)
    negative = working_normals[np.arange(len(working_normals)), dominant] < 0
    working_normals[negative] *= -1
    mesh_positions, mesh_normals, mesh_ids = positions[valid], working_normals[valid], ids[valid]
    candidate = np.empty((0, 3), dtype=np.int64)
    if len(mesh_positions) >= 3:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(mesh_positions.astype(np.float64))
        cloud.normals = o3d.utility.Vector3dVector(mesh_normals)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            cloud, o3d.utility.DoubleVector(SUPPORT['radii'])
        )
        if not np.array_equal(np.asarray(mesh.vertices), mesh_positions):
            raise AssertionError('Ball pivoting changed original vertices')
        candidate = np.asarray(mesh.triangles).copy()
    triangles = mesh_positions[candidate].astype(np.float64)
    centroid = triangles.mean(axis=1)
    owned = ((centroid[:, :2] >= low) & (centroid[:, :2] < high)).all(axis=1)
    candidate, triangles = candidate[owned], triangles[owned]
    edges = triangles[:, [1, 2, 0]] - triangles
    lengths = np.linalg.norm(edges, axis=2)
    cross = np.cross(edges[:, 0], -edges[:, 2])
    area = np.linalg.norm(cross, axis=1)
    face_normal = cross / np.maximum(area[:, None], 1e-30)
    tri_normals = mesh_normals[candidate]
    keep = (area > SUPPORT['min_double_area']) & (lengths.max(axis=1) <= SUPPORT['max_edge'])
    keep &= (
        np.einsum('fi,fvi->fv', face_normal, tri_normals).min(axis=1)
        >= SUPPORT['min_face_normal_dot']
    )
    keep &= (
        np.einsum('fvi,fvi->fv', tri_normals, tri_normals[:, [1, 2, 0]]).min(axis=1)
        >= SUPPORT['min_pair_normal_dot']
    )
    direction = edges / np.maximum(lengths[:, :, None], 1e-30)
    tangent = np.maximum(
        np.abs(np.einsum('fvi,fvi->fv', direction, tri_normals)),
        np.abs(np.einsum('fvi,fvi->fv', direction, tri_normals[:, [1, 2, 0]])),
    )
    keep &= tangent.max(axis=1) <= SUPPORT['max_edge_normal_dot']
    if len(candidate):
        tree = cKDTree(mesh_positions)
        midpoint = (triangles + triangles[:, [1, 2, 0]]) * 0.5
        distances = tree.query(midpoint.reshape(-1, 3), workers=1)[0].reshape(-1, 3)
        keep &= distances.max(axis=1) <= SUPPORT['midpoint_distance']
        keep &= tree.query(triangles.mean(axis=1), workers=1)[0] <= SUPPORT['centroid_distance']
    faces = candidate[keep]
    used, inverse = np.unique(faces, return_inverse=True)
    faces = inverse.reshape(-1, 3).astype(np.int32)
    retained_ids = mesh_ids[used]
    mesh_xyz = mesh_positions[used]
    np.savez(output / 'meshes' / f'{name}.npz', ids=retained_ids, faces=faces)
    all_edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    _, multiplicities = np.unique(all_edges, axis=0, return_counts=True)
    if len(faces):
        intensity = records['intensity'][valid][used]
        author_geometry(
            output / 'meshes' / f'{name}.usdc', mesh_xyz, retained_ids, intensity, faces
        )
    mesh_core_ids = retained_ids[np.isin(retained_ids, own_ids)]
    horizontal = int((np.abs(face_normal[keep, 2]) >= 0.7).sum())
    report = {
        'name': name,
        'key': list(key),
        'records': len(own_ids),
        'normal_valid': int(own_valid.sum()),
        'normal_invalid': int((~own_valid).sum()),
        'halo_records': len(records),
        'candidate_owned': len(candidate),
        'triangles': len(faces),
        'rejected_support': int((~keep).sum()),
        'mesh_vertices': len(used),
        'core_meshed_records': len(mesh_core_ids),
        'unmeshed_records': len(own_ids) - len(mesh_core_ids),
        'horizontal_triangles': horizontal,
        'other_triangles': len(faces) - horizontal,
        'open_boundary_edges': int((multiplicities == 1).sum()),
        'nonmanifold_edges': int((multiplicities > 2).sum()),
        'min': own_xyz.min(axis=0).astype(float).tolist(),
        'max': own_xyz.max(axis=0).astype(float).tolist(),
    }
    if len(faces):
        report['mesh_min'] = mesh_xyz.min(axis=0).astype(float).tolist()
        report['mesh_max'] = mesh_xyz.max(axis=0).astype(float).tolist()
    dump(output / 'reports' / f'{name}.json', report)
    return report


def make_layers(output: Path, reports: list[dict[str, Any]]) -> None:
    from pxr import UsdGeom

    entries, point_entries = [], []
    for filename, folder, only_mesh in [
        ('streamed.usda', 'meshes', True),
        ('points.usda', 'points', False),
    ]:
        stage = stage_base(output / filename, '/World')
        UsdGeom.Xform.Define(stage, '/World/Regions')
        for report in reports:
            if only_mesh and not report['triangles']:
                continue
            group = 'Regions' if only_mesh else 'PointRegions'
            path = f'/World/{group}/{report["name"]}'
            prim = UsdGeom.Xform.Define(stage, path).GetPrim()
            prim.GetPayloads().AddPayload(f'./{folder}/{report["name"]}.usdc', '/Tile')
            entry = {
                'path': path,
                'min': report.get('mesh_min', report['min']) if only_mesh else report['min'],
                'max': report.get('mesh_max', report['max']) if only_mesh else report['max'],
                'payload': f'{folder}/{report["name"]}.usdc',
            }
            (entries if only_mesh else point_entries).append(entry)
        stage.GetRootLayer().Save()
    dump(output / 'index.json', entries)
    dump(output / 'point-index.json', point_entries)
    dump(output / 'world-index.json', entries + point_entries)
    world = stage_base(output / 'world.usda', '/World')
    world.GetRootLayer().subLayerPaths = ['./streamed.usda', './points.usda']
    world.GetRootLayer().Save()
    # Point archive is deliberately independent of collision/render world.
    archive = stage_base(output / 'archival-points.usda', '/World')
    archive.GetRootLayer().subLayerPaths = ['./points.usda']
    archive.GetRootLayer().Save()


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    for folder in ('source', 'records', 'meshes', 'points', 'reports'):
        (output / folder).mkdir(parents=True, exist_ok=True)
    provenance = {}
    for year, source in (('2019', args.source), ('2018', args.reference_2018)):
        if source is None:
            continue
        archive = output / 'source' / source.name
        if source.resolve() != archive.resolve() and not archive.exists():
            shutil.copyfile(source, archive)
        _, metadata = open_records(archive)
        metadata.update(
            {
                'sha256': sha256(archive),
                'path': f'source/{source.name}',
                'official_url': SOURCE_URLS[year],
                'license': LICENSE,
                'role': 'primary' if year == '2019' else 'unfused reference',
            }
        )
        if metadata['sha256'] != sha256(source):
            raise AssertionError('Archive hash disagrees with source')
        if source.name in PUBLIC_HASHES and metadata['sha256'] != PUBLIC_HASHES[source.name]:
            raise AssertionError('Archive hash disagrees with acquired official source')
        provenance[year] = metadata
    records, _ = open_records(output / provenance['2019']['path'])
    positions = xyz(records)
    if not np.isfinite(positions).all():
        raise ValueError('Nonfinite XYZ requires a separate archival coverage policy')
    keys = np.floor(positions[:, :2].astype(np.float64) / args.tile_size).astype(np.int32)
    unique, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind='stable')
    occupied = {tuple(map(int, key)) for key in unique}
    start = 0
    for key, count in zip(unique, counts):
        ids = order[start : start + count].astype(np.int64)
        np.savez(
            output / 'records' / f'{tile_name(tuple(key))}.npz', records=records[ids], ids=ids
        )
        start += count
    del positions, keys, inverse, order
    manifest = {
        'schema': 1,
        'license': LICENSE,
        'attribution': ATTRIBUTION,
        'source_catalog': CATALOG,
        'sources': provenance,
        'tile_size': args.tile_size,
        'halo': args.halo,
        'support': SUPPORT,
        'frame': 'raw 2019 XYZ',
        'units': 'meters',
        'up_axis': 'Z',
        'color': 'source intensity grayscale, not RGB',
        'records': len(records),
        'occupied_tiles': len(occupied),
        'method': 'CPU Open3D ball pivoting, normalized supplied plane normals',
        'working_normal_orientation': 'largest absolute component positive; XYZ tie order',
        'rendering': 'all source points plus supported triangle surfaces; no point omission',
        'ownership': 'half-open XY face-centroid core; source-order normal-preserving input',
        'unsupported': 'open gaps remain; no invented ground, extrusion or filling',
    }
    dump(output / 'manifest.json', manifest)
    (output / 'NOTICE').write_text(
        'Unofficial derived Tsukuba Challenge map; provider fuRo.\n'
        'Data and derived geometry: CC BY-NC-SA 4.0.\n'
        'https://creativecommons.org/licenses/by-nc-sa/4.0/\n'
        f'Source catalog: {CATALOG}\n'
        f'Attribution and requested citation: {ATTRIBUTION}\n'
        'Changes: regional partitioning, intensity grayscale display, supported\n'
        'ball-pivoted triangles with exact original vertices. Working normal signs\n'
        'are oriented for pivoting; original fields remain unchanged.\n'
        '2019 is primary; 2018 is preserved separately, not silently fused.\n'
        'No geographic survey accuracy, RGB color, sealed surfaces or gap filling\n'
        'is asserted. See manifest.json for hashes, sources and reconstruction.\n'
        'Converter/viewer software is separately MIT licensed by usim.\n',
        encoding='utf-8',
    )
    print(f'ARCHIVED records={len(records)} tiles={len(occupied)}', flush=True)
    jobs = [(str(output), key, args.tile_size, args.halo, occupied) for key in sorted(occupied)]
    reports = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for report in executor.map(region_job, jobs, chunksize=1):
            reports.append(report)
            if len(reports) % 25 == 0:
                print(
                    f'PROGRESS tiles={len(reports)}/{len(jobs)} '
                    f'triangles={sum(r["triangles"] for r in reports)}',
                    flush=True,
                )
    make_layers(output, reports)
    summary: dict[str, Any] = {
        field: sum(report[field] for report in reports)
        for field in (
            'records',
            'normal_valid',
            'normal_invalid',
            'triangles',
            'mesh_vertices',
            'core_meshed_records',
            'unmeshed_records',
            'horizontal_triangles',
            'other_triangles',
            'open_boundary_edges',
            'nonmanifold_edges',
            'rejected_support',
        )
    }
    summary.update(
        {
            'occupied_tiles': len(reports),
            'meshed_tiles': sum(bool(r['triangles']) for r in reports),
            'unmeshed_tiles': [r['name'] for r in reports if not r['triangles']],
        }
    )
    dump(output / 'summary.json', summary)
    print('GENERATION_PASSED ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--reference-2018', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--tile-size', type=float, default=30.0)
    parser.add_argument('--halo', type=float, default=1.5)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not 20 <= args.tile_size <= 40 or not 0.7 <= args.halo < args.tile_size:
        parser.error('Require 20-40 meter cores and halo >= twice largest pivot radius')
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    run(args)
