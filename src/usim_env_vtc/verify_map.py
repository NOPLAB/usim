"""MIT tooling: independently reopen every USD tile and verify complete source fidelity.

Does not import reconstruction code or trust its acceptance masks/vertex buffers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

if not __package__:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'usim_env_vtc'
from .pcd import FIELDS, PUBLIC_HASHES, normals, open_records, sha256, xyz
from pxr import Usd, UsdGeom, UsdPhysics
from scipy.spatial import cKDTree


def require(condition: bool | np.bool_, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify(output: Path) -> dict[str, Any]:
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    source, metadata = open_records(output / manifest['sources']['2019']['path'])
    for year, item in manifest['sources'].items():
        path = output / item['path']
        archive, header = open_records(path)
        require(
            header['points'] == item['points'] and header['bytes'] == item['bytes'],
            f'{year} archival count/bytes mismatch',
        )
        require(sha256(path) == item['sha256'], f'{year} archival SHA256 mismatch')
        if path.name in PUBLIC_HASHES:
            require(
                item['sha256'] == PUBLIC_HASHES[path.name],
                f'{year} checksum differs from acquired official public source',
            )
        require(header['header'] == item['header'], f'{year} original header mismatch')
        require(
            all(np.isfinite(archive[field]).all() for field in FIELDS),
            f'{year} field contains nonfinite data',
        )
    require(metadata['points'] == manifest['records'], 'Primary count mismatch')
    seen = np.zeros(len(source), dtype=bool)
    reports = [
        json.loads(path.read_text(encoding='utf-8'))
        for path in sorted((output / 'reports').glob('*.json'))
    ]
    occupied = {tuple(report['key']): report['name'] for report in reports}
    require(len(occupied) == manifest['occupied_tiles'], 'Occupied tile coverage mismatch')
    source_keys = np.floor(xyz(source)[:, :2].astype(np.float64) / manifest['tile_size']).astype(
        np.int32
    )
    expected_keys = {tuple(key) for key in np.unique(source_keys, axis=0)}
    require(set(occupied) == expected_keys, 'Dataset occupied XY cores missing/extra')
    entries = json.loads((output / 'index.json').read_text(encoding='utf-8'))
    point_entries = json.loads((output / 'point-index.json').read_text(encoding='utf-8'))
    require(len(point_entries) == len(reports), 'Point payload coverage mismatch')
    expected_mesh_paths = {f'/World/Regions/{r["name"]}' for r in reports if r['triangles']}
    require({e['path'] for e in entries} == expected_mesh_paths, 'Mesh payload index mismatch')
    index = {entry['path'].split('/')[-1]: entry for entry in entries}
    totals = {
        'records': 0,
        'triangles': 0,
        'mesh_vertices': 0,
        'meshed_tiles': 0,
        'normal_valid': 0,
        'normal_invalid': 0,
        'open_boundary_edges': 0,
        'nonmanifold_edges': 0,
        'horizontal_triangles': 0,
    }
    support = manifest['support']
    for number, report in enumerate(reports, 1):
        name, key = report['name'], np.array(report['key'])
        low, high = key * manifest['tile_size'], (key + 1) * manifest['tile_size']
        archive = np.load(output / 'records' / f'{name}.npz')
        ids, records = archive['ids'], archive['records']
        require(
            ids.dtype == np.int64 and len(ids) == report['records'],
            f'{name}: archival index count/type',
        )
        require((ids >= 0).all() and (ids < len(source)).all(), f'{name}: invalid record ID')
        require(
            len(np.unique(ids)) == len(ids) and not seen[ids].any(),
            f'{name}: duplicate source records',
        )
        require(
            np.array_equal(records.view(np.uint8), source[ids].view(np.uint8)),
            f'{name}: original XYZ/normal/intensity/curvature bits changed',
        )
        require((source_keys[ids] == key).all(), f'{name}: archival half-open ownership')
        seen[ids] = True
        expected_xyz = xyz(records)
        _, normal_valid = normals(records)
        require(int(normal_valid.sum()) == report['normal_valid'], f'{name}: normal report')
        points_stage = Usd.Stage.Open(str(output / 'points' / f'{name}.usdc'))
        points = UsdGeom.Points(points_stage.GetPrimAtPath('/Tile/Points'))
        require(
            np.array_equal(
                np.asarray(points.GetPointsAttr().Get()).view(np.uint32),
                expected_xyz.view(np.uint32),
            ),
            f'{name}: every USD archival point must equal exact source float32',
        )
        require(
            np.array_equal(np.asarray(points.GetIdsAttr().Get()), ids),
            f'{name}: USD point record IDs',
        )
        require(
            not points.GetPrim().HasAPI(UsdPhysics.CollisionAPI),
            f'{name}: archival points unexpectedly collide',
        )
        expected_color = np.repeat((records['intensity'] / 255)[:, None], 3, axis=1)
        require(
            np.array_equal(np.asarray(points.GetDisplayColorPrimvar().Get()), expected_color),
            f'{name}: intensity grayscale changed',
        )
        totals['records'] += len(ids)
        totals['normal_valid'] += int(normal_valid.sum())
        totals['normal_invalid'] += int((~normal_valid).sum())
        mesh_archive = np.load(output / 'meshes' / f'{name}.npz')
        vertex_ids, expected_faces = mesh_archive['ids'], mesh_archive['faces']
        require(len(expected_faces) == report['triangles'], f'{name}: face count report')
        if not report['triangles']:
            require(
                not (output / 'meshes' / f'{name}.usdc').exists(),
                f'{name}: unexplained stale empty mesh',
            )
            continue
        mesh_stage = Usd.Stage.Open(str(output / 'meshes' / f'{name}.usdc'))
        mesh = UsdGeom.Mesh(mesh_stage.GetPrimAtPath('/Tile/Surface'))
        vertices = np.asarray(mesh.GetPointsAttr().Get())
        usd_ids = np.asarray(mesh.GetPrim().GetAttribute('source:recordIds').Get())
        require(np.array_equal(usd_ids, vertex_ids), f'{name}: mesh record IDs changed')
        require(
            np.array_equal(vertices.view(np.uint32), xyz(source[vertex_ids]).view(np.uint32)),
            f'{name}: EVERY authored mesh vertex/seam must equal original source float32',
        )
        require(
            (np.asarray(mesh.GetFaceVertexCountsAttr().Get()) == 3).all(),
            f'{name}: nontriangle faces',
        )
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        require(np.array_equal(faces, expected_faces), f'{name}: archived/USD indices differ')
        require((faces >= 0).all() and (faces < len(vertices)).all(), f'{name}: invalid faces')
        sorted_ids = np.sort(vertex_ids[faces], axis=1)
        require(len(np.unique(sorted_ids, axis=0)) == len(faces), f'{name}: duplicate faces')
        coordinate_keys = np.ascontiguousarray(vertices[faces]).view('V12').reshape(-1, 3)
        sorted_coordinates = np.ascontiguousarray(np.sort(coordinate_keys, axis=1))
        require(
            len(np.unique(sorted_coordinates.view('V36'))) == len(faces),
            f'{name}: duplicate geometry with different source IDs',
        )
        triangle_xyz = vertices[faces].astype(np.float64)
        centroid = triangle_xyz.mean(axis=1)
        require(
            ((centroid[:, :2] >= low) & (centroid[:, :2] < high)).all(),
            f'{name}: face-centroid half-open ownership; also forbids cross-tile duplicates',
        )
        require(
            (
                (vertices[:, :2] >= low - manifest['halo'])
                & (vertices[:, :2] < high + manifest['halo'])
            ).all(),
            f'{name}: halo escape',
        )
        edges = triangle_xyz[:, [1, 2, 0]] - triangle_xyz
        lengths = np.linalg.norm(edges, axis=2)
        cross = np.cross(edges[:, 0], -edges[:, 2])
        double_area = np.linalg.norm(cross, axis=1)
        require((double_area > support['min_double_area']).all(), f'{name}: degenerate face')
        require((lengths <= support['max_edge'] + 1e-12).all(), f'{name}: unsupported long edge')
        vertex_normals, valid = normals(source[vertex_ids])
        require(valid.all(), f'{name}: unsupplied mesh normal')
        if manifest.get('working_normal_orientation'):
            require(
                manifest['working_normal_orientation']
                == 'largest absolute component positive; XYZ tie order',
                'Unknown working normal orientation policy',
            )
            # Independent implementation; never change the original normal fields.
            for axis in range(3):
                selected = np.argmax(np.abs(vertex_normals), axis=1) == axis
                vertex_normals[selected & (vertex_normals[:, axis] < 0)] *= -1
        face_normal = cross / double_area[:, None]
        tri_normals = vertex_normals[faces]
        require(
            (
                np.einsum('fi,fvi->fv', face_normal, tri_normals)
                >= support['min_face_normal_dot'] - 1e-12
            ).all(),
            f'{name}: winding/normal support mismatch',
        )
        require(
            (
                np.einsum('fvi,fvi->fv', tri_normals, tri_normals[:, [1, 2, 0]])
                >= support['min_pair_normal_dot'] - 1e-12
            ).all(),
            f'{name}: incompatible source normals',
        )
        direction = edges / lengths[:, :, None]
        require(
            (
                np.abs(np.einsum('fvi,fvi->fv', direction, tri_normals))
                <= support['max_edge_normal_dot'] + 1e-12
            ).all(),
            f'{name}: edge leaves source tangent plane',
        )
        require(
            (
                np.abs(np.einsum('fvi,fvi->fv', direction, tri_normals[:, [1, 2, 0]]))
                <= support['max_edge_normal_dot'] + 1e-12
            ).all(),
            f'{name}: edge leaves destination tangent plane',
        )
        neighbors = []
        for x in range(key[0] - 1, key[0] + 2):
            for y in range(key[1] - 1, key[1] + 2):
                if (x, y) not in occupied:
                    continue
                neighbor = np.load(output / 'records' / f'{occupied[(x, y)]}.npz')['records']
                neighbor_xyz = xyz(neighbor)
                _, valid_neighbor = normals(neighbor)
                included = (
                    (neighbor_xyz[:, :2] >= low - manifest['halo'])
                    & (neighbor_xyz[:, :2] < high + manifest['halo'])
                ).all(axis=1)
                neighbors.append(neighbor_xyz[included & valid_neighbor])
        tree = cKDTree(np.concatenate(neighbors))
        midpoint = (triangle_xyz + triangle_xyz[:, [1, 2, 0]]) / 2
        require(
            (
                tree.query(midpoint.reshape(-1, 3), workers=1)[0]
                <= support['midpoint_distance'] + 1e-12
            ).all(),
            f'{name}: unsupported midpoint bridge',
        )
        require(
            (tree.query(centroid, workers=1)[0] <= support['centroid_distance'] + 1e-12).all(),
            f'{name}: unsupported face interior',
        )
        require(
            mesh.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
            and mesh.GetPrim().HasAPI(UsdPhysics.MeshCollisionAPI)
            and UsdPhysics.CollisionAPI(mesh).GetCollisionEnabledAttr().Get()
            and UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Get() == 'none',
            f'{name}: full-triangle collision API missing',
        )
        require(mesh.GetSubdivisionSchemeAttr().Get() == 'none', f'{name}: subdivision')
        require(
            np.array_equal(vertices.min(axis=0), index[name]['min'])
            and np.array_equal(vertices.max(axis=0), index[name]['max']),
            f'{name}: streamed payload bounds disagree',
        )
        mesh_edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
        _, multiplicities = np.unique(mesh_edges, axis=0, return_counts=True)
        require(
            int((multiplicities == 1).sum()) == report['open_boundary_edges'],
            f'{name}: open boundary report',
        )
        totals['triangles'] += len(faces)
        totals['mesh_vertices'] += len(vertices)
        totals['meshed_tiles'] += 1
        totals['open_boundary_edges'] += int((multiplicities == 1).sum())
        totals['nonmanifold_edges'] += int((multiplicities > 2).sum())
        totals['horizontal_triangles'] += int((np.abs(face_normal[:, 2]) >= 0.7).sum())
        if number % 100 == 0:
            print(f'VERIFY_PROGRESS tiles={number}/{len(reports)}', flush=True)
    require(seen.all(), 'Some source records absent from independently streamed archives')
    if (output / 'world-index.json').exists():
        world_entries = json.loads((output / 'world-index.json').read_text(encoding='utf-8'))
        require(world_entries == entries + point_entries, 'Complete rendering index mismatch')
        world = Usd.Stage.Open(str(output / 'world.usda'), load=Usd.Stage.LoadNone)
        require(
            set(world.GetRootLayer().subLayerPaths) == {'./streamed.usda', './points.usda'},
            'Default world omits full-resolution points or surfaces',
        )
        require(
            len({entry['path'] for entry in world_entries}) == len(world_entries),
            'Point and mesh payload paths collide',
        )
    totals['occupied_tiles'] = len(reports)
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    require(
        all(summary[field] == value for field, value in totals.items()),
        'Global quantitative reports mismatch',
    )
    for filename in ('world.usda', 'streamed.usda', 'points.usda', 'archival-points.usda'):
        stage = Usd.Stage.Open(str(output / filename), load=Usd.Stage.LoadNone)
        if stage is None:
            raise AssertionError(f'{filename}: USD stage cannot be opened')
        require(
            UsdGeom.GetStageUpAxis(stage) == 'Z' and UsdGeom.GetStageMetersPerUnit(stage) == 1,
            f'{filename}: frame metadata',
        )
        require(
            stage.GetRootLayer().customLayerData.get('license') == 'CC-BY-NC-SA-4.0',
            f'{filename}: license metadata',
        )
        if filename in ('streamed.usda', 'points.usda'):
            expected_entries = entries if filename == 'streamed.usda' else point_entries
            for entry in expected_entries:
                prim = stage.GetPrimAtPath(entry['path'])
                require(prim and prim.HasAuthoredPayloads(), f'{filename}: payload missing')
    result = {
        'status': 'passed',
        'source_sha256': manifest['sources']['2019']['sha256'],
        **totals,
        'all_vertices_exact_source_bits': True,
        'all_occupied_tiles_archived': True,
        'all_records_exact_source_bits': True,
        'cross_tile_duplicate_faces': 0,
        'seam_coordinate_mismatches': 0,
    }
    (output / 'verification.json').write_text(
        json.dumps(result, indent=2) + '\n', encoding='utf-8'
    )
    print('VERIFICATION_PASSED ' + json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    verify(parser.parse_args().output.resolve())
