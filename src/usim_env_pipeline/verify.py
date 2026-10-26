"""Independent source-bit, regional collision and authored-reference verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .models import PipelineInputError
from .pcd import open_records, sha256, xyz
from .storage import tile_name, write_json


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise PipelineInputError(reason)


def verify(output: Path) -> dict[str, int | str | bool]:
    from pxr import Usd, UsdGeom, UsdPhysics

    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    require(manifest['status'] == 'complete', 'Map generation is incomplete')
    source = manifest['sources'][manifest['primary_source']]
    original = output / source['path']
    require(sha256(original) == source['sha256'], 'Archived source hash differs')
    records, _ = open_records(original)
    seen = np.zeros(len(records), dtype=bool)
    totals = {'records': 0, 'triangles': 0, 'mesh_vertices': 0, 'occupied_tiles': 0}
    entries = {
        item['path'].rsplit('/', 1)[-1]: item
        for item in json.loads((output / 'index.json').read_text(encoding='utf-8'))
    }
    for filename in sorted(output.joinpath('reports').glob('*.json')):
        report = json.loads(filename.read_text(encoding='utf-8'))
        name = filename.stem
        with np.load(output / 'records' / f'{name}.npz') as archive:
            ids, actual = archive['ids'], archive['records']
        require(
            bool((ids >= 0).all() and (ids < len(records)).all() and (np.diff(ids) > 0).all()),
            'Regional source IDs are invalid',
        )
        require(not bool(seen[ids].any()), 'Source row archived more than once')
        require(actual.dtype == records.dtype, 'Original field dtype/order differs')
        require(actual.tobytes() == records[ids].tobytes(), 'Original record bits differ')
        seen[ids] = True
        totals['records'] += len(ids)
        totals['occupied_tiles'] += 1
        with np.load(output / 'meshes' / f'{name}.npz') as mesh:
            vertex_ids, faces = mesh['ids'], mesh['faces']
        totals['triangles'] += len(faces)
        totals['mesh_vertices'] += len(vertex_ids)
        require(len(faces) == report['triangles'], 'Reported face count differs')
        require(
            bool((vertex_ids >= 0).all() and (vertex_ids < len(records)).all()),
            'Mesh source vertex IDs are invalid',
        )
        if not len(faces):
            require(name not in entries, 'Empty surface has a mesh payload entry')
            continue
        expected_points = xyz(records[vertex_ids])
        require(
            bool((faces >= 0).all() and (faces < len(vertex_ids)).all()),
            'Face indices are invalid',
        )
        centers = expected_points[faces].mean(axis=1, dtype=np.float64)[:, :2]
        low = np.asarray(report['key']) * manifest['tile_size']
        require(
            bool(((centers >= low) & (centers < low + manifest['tile_size'])).all()),
            'Cross-core face-centroid ownership differs',
        )
        for folder in ('meshes', 'authored/surfaces'):
            path = output / folder / f'{name}.usdc'
            if folder.startswith('authored') and not (output / 'authored-scene.json').exists():
                continue
            stage = Usd.Stage.Open(str(path))
            geometry = UsdGeom.Mesh.Get(stage, '/Tile/Surface')
            require(
                bool(np.array_equal(geometry.GetPointsAttr().Get(), expected_points)),
                'USD vertex coordinates differ from source',
            )
            require(
                bool(np.array_equal(geometry.GetFaceVertexIndicesAttr().Get(), faces.ravel())),
                'USD face indices differ from archive',
            )
            require(
                bool(
                    np.array_equal(
                        geometry.GetPrim().GetAttribute('source:recordIds').Get(), vertex_ids
                    )
                ),
                'USD source vertex IDs differ',
            )
            require(
                bool(UsdPhysics.CollisionAPI(geometry.GetPrim()).GetCollisionEnabledAttr().Get())
                and UsdPhysics.MeshCollisionAPI(geometry.GetPrim()).GetApproximationAttr().Get()
                == 'none',
                'Full triangle collision is missing',
            )
        entry = entries[name]
        require(
            bool(
                (expected_points.min(0) >= entry['min']).all()
                and (expected_points.max(0) <= entry['max']).all()
            ),
            'Streaming bounds do not contain measured geometry',
        )
    require(bool(seen.all()), 'Not every source record was archived')
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    for key, value in totals.items():
        require(value == summary[key], f'Summary {key} differs')
    ground_triangles = 0
    if (output / 'authored-scene.json').exists():
        authored = json.loads((output / 'authored-scene.json').read_text(encoding='utf-8'))
        index = json.loads((output / 'authored-index.json').read_text(encoding='utf-8'))
        for entry in index:
            if not entry['path'].startswith('/World/Ground/'):
                continue
            stage = Usd.Stage.Open(str(output / entry['payload']))
            geometry = UsdGeom.Mesh.Get(stage, '/Tile/Surface')
            ids = np.asarray(geometry.GetPrim().GetAttribute('source:recordIds').Get())
            points = np.asarray(geometry.GetPointsAttr().Get())
            faces = np.asarray(geometry.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
            require(
                bool((ids >= 0).all() and (ids < len(records)).all()),
                'Ground source vertex IDs are invalid',
            )
            require(np.array_equal(points, xyz(records[ids])), 'Ground source coordinates differ')
            require(
                bool((faces >= 0).all() and (faces < len(ids)).all()),
                'Ground face indices are invalid',
            )
            triangles = points[faces].astype(np.float64)
            edges = triangles[:, [1, 2, 0]] - triangles
            cross = np.cross(edges[:, 0], -edges[:, 2])
            require(bool((cross[:, 2] > 0).all()), 'Ground winding is not upward')
            require(
                bool(
                    (
                        np.linalg.norm(edges, axis=2).max(axis=1)
                        <= manifest['support']['max_edge']
                    ).all()
                ),
                'Ground triangle bridges unsupported spacing',
            )
            keys = np.floor(triangles.mean(axis=1)[:, :2] / manifest['tile_size']).astype(int)
            names = {tile_name((int(key[0]), int(key[1]))) for key in np.unique(keys, axis=0)}
            require(names == {entry['path'].rsplit('/', 1)[-1]}, 'Ground ownership differs')
            require(
                bool(UsdPhysics.CollisionAPI(geometry.GetPrim()).GetCollisionEnabledAttr().Get())
                and UsdPhysics.MeshCollisionAPI(geometry.GetPrim()).GetApproximationAttr().Get()
                == 'none',
                'Ground full triangle collision is missing',
            )
            require(
                bool(
                    (points.min(0) >= entry['min']).all() and (points.max(0) <= entry['max']).all()
                ),
                'Ground streaming bounds differ',
            )
            ground_triangles += len(faces)
        require(
            ground_triangles == authored.get('ground_triangles', 0), 'Ground face count differs'
        )
    result = {
        'status': 'passed',
        **totals,
        'all_source_records_exact': True,
        'all_source_vertices_exact': True,
        'all_triangle_collisions_preserved': True,
        'all_surface_bounds_conservative': True,
        'ground_triangles': ground_triangles,
        'all_ground_vertices_exact': True,
        'all_ground_collisions_preserved': True,
    }
    write_json(output / 'verification.json', result)
    print('MAP_VERIFICATION_PASSED ' + json.dumps(result), flush=True)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', type=Path, required=True)
    verify(parser.parse_args(argv).map)


if __name__ == '__main__':
    main()
