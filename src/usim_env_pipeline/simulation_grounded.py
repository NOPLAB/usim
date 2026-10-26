"""Publish a non-destructive fitted view containing only floor-connected surfaces."""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .models import PipelineInputError
from .pcd import sha256
from .simulation_ground_connectivity import (
    ContactGraph,
    ground_connections,
    mesh_components,
    triangle_chunks,
    weld_components,
)
from .storage import write_json
from .simulation_grounded_output import CachedMesh


def remove_floating(output: Path, *, angular: bool = False, angular_name: str = 'angular') -> None:
    """Keep whole mesh components reached through shared points or surface intersection."""
    from pxr import Usd, UsdGeom, UsdPhysics

    from .simulation_fit_verify import verify_ground
    from .simulation_grounded_output import publish
    from .usd import LayerMetadata

    output = output.resolve()
    folder = output / (f'{angular_name}-grounded-models' if angular else 'grounded-models')
    work = folder / 'work'
    work.mkdir(parents=True, exist_ok=True)
    fitted = output / (f'simulation-{angular_name}.usda' if angular else 'simulation-fitted.usda')
    index_file = output / (
        f'{angular_name}-models/full-index.json' if angular else 'geometry-fit/index.json'
    )
    rows = json.loads(index_file.read_text(encoding='utf-8'))
    source = Usd.Stage.Open(str(fitted), Usd.Stage.LoadNone)
    source.Load('/World/SimulationGround')
    verify_ground(output, source)
    floor = next(
        UsdGeom.Mesh(prim)
        for prim in source.Traverse()
        if prim.IsA(UsdGeom.Mesh) and prim.GetPath().HasPrefix('/World/SimulationGround')
    )
    tracked = {
        fitted,
        output / 'simulation.usda',
        index_file,
        output / 'simulation-models/terrain.npz',
        output / 'simulation-models/source-hashes.json',
        output / 'simulation-models/index.json',
    }
    tracked.update(
        output / relative
        for relative in json.loads(
            (output / 'simulation-models/source-hashes.json').read_text(encoding='utf-8')
        )
    )
    for row in rows:
        tracked.update((output / row['payload'], (output / row['payload']).with_suffix('.npz')))
    tracked.update(Path(layer.realPath) for layer in source.GetUsedLayers() if layer.realPath)
    fingerprints = {str(path): sha256(path) for path in tracked}
    layer_data = source.GetRootLayer().customLayerData
    metadata = LayerMetadata(
        layer_data['license'],
        layer_data['attribution'],
        layer_data['sourceCatalog'],
        layer_data['frame'],
        layer_data['color'],
    )
    sources, chunk_tables, assets = [], [], []
    seam_points, seam_owners = [], []
    count = 0
    simulation_rows = json.loads(
        (output / 'simulation-models/index.json').read_text(encoding='utf-8')
    )
    entries = [next(row for row in simulation_rows if row['kind'] == 'ground'), *rows]
    for number, row in enumerate(entries):
        if number and not source.GetPrimAtPath(row['path']).IsActive():
            continue
        source.Load(row['path'])
        mesh = floor if number == 0 else UsdGeom.Mesh.Get(source, row['path'] + '/Surface')
        archive = (
            output / 'simulation-models/terrain.npz'
            if number == 0
            else (output / row['payload']).with_suffix('.npz')
        )
        with np.load(archive) as data:
            positions, faces = data['positions'], data['faces']
        prim = mesh.GetPrim()
        if not (
            np.array_equal(positions, mesh.GetPointsAttr().Get())
            and np.array_equal(
                faces, np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
            )
            and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            and UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == 'none'
        ):
            raise PipelineInputError(
                f'Published fitted geometry differs from its cache: {archive}'
            )
        matrix = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(prim), np.float64)
        world_points = positions.astype(np.float64) @ matrix[:3, :3] + matrix[3, :3]
        parts = mesh_components(world_points, faces)
        if number == 0 and parts.count != 1:
            raise PipelineInputError('Ground must remain one connected finite surface')
        labels = parts.face_parts.astype(np.int64) + count
        label_file, order_file = work / f'{number:04d}-parts.npy', work / f'{number:04d}-order.npy'
        np.save(label_file, labels)
        if number:
            np.savez_compressed(
                label_file.with_suffix('.appearance.npz'),
                colors=np.asarray(mesh.GetDisplayColorPrimvar().ComputeFlattened()),
                normals=np.asarray(mesh.GetNormalsAttr().Get()),
                colors_interpolation=str(mesh.GetDisplayColorPrimvar().GetInterpolation()),
                normals_interpolation=str(mesh.GetNormalsInterpolation()),
            )
        chunks = triangle_chunks(world_points, faces, labels)
        np.save(order_file, chunks.order)
        assets.append(np.full(len(chunks.owners), len(sources), np.int32))
        sources.append(
            CachedMesh(
                row['path'], row['payload'], archive, matrix, label_file, order_file, len(faces)
            )
        )
        chunk_tables.append(chunks)
        seam_points.append(parts.positions)
        seam_owners.append(parts.vertex_parts.astype(np.int64) + count)
        count += parts.count
        source.Unload(row['path'])
        print(
            f'GROUND_CONNECTION_TOPOLOGY source={number}/{len(entries) - 1} '
            f'components={count} faces={len(faces)}',
            flush=True,
        )
    coordinates, owners = np.concatenate(seam_points), np.concatenate(seam_owners)
    del seam_points, seam_owners
    component_count, merged = weld_components(coordinates, owners, count)
    del coordinates, owners
    graph = ContactGraph(
        merged[np.concatenate([table.owners for table in chunk_tables])].astype(np.int64),
        np.concatenate([table.lower for table in chunk_tables]),
        np.concatenate([table.upper for table in chunk_tables]),
    )
    asset_ids = np.concatenate(assets)
    starts = np.concatenate([table.starts for table in chunk_tables])
    stops = np.concatenate([table.stops for table in chunk_tables])
    del chunk_tables, assets
    write_json(
        folder / 'sources.json',
        [
            {
                'path': item.path,
                'archive': item.archive.relative_to(output).as_posix(),
                'matrix': item.matrix.tolist(),
                'labels': item.labels.relative_to(output).as_posix(),
                'order': item.order.relative_to(output).as_posix(),
            }
            for item in sources
        ],
    )

    @lru_cache(maxsize=2)
    def load_mesh(index: int) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
        item = sources[index]
        with np.load(item.archive) as data:
            points, triangles = data['positions'], data['faces']
        points = points.astype(np.float64) @ item.matrix[:3, :3] + item.matrix[3, :3]
        return points, triangles

    @lru_cache(maxsize=2)
    def load_order(index: int) -> NDArray[np.int64]:
        return np.load(sources[index].order, mmap_mode='r')

    def load_chunk(index: int) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
        asset = int(asset_ids[index])
        points, triangles = load_mesh(asset)
        order = load_order(asset)
        return points, triangles[order[starts[index] : stops[index]]]

    connections = ground_connections(graph, load_chunk, component_count, int(merged[0]))
    load_mesh.cache_clear()
    load_order.cache_clear()
    np.savez_compressed(
        folder / 'connections.npz',
        keep=connections.keep,
        merged=merged,
        witnesses=connections.witnesses,
        chunk_owners=graph.owners,
        chunk_assets=asset_ids,
        chunk_starts=starts,
        chunk_stops=stops,
    )
    published = publish(
        output,
        sources,
        connections.keep,
        merged,
        metadata,
        angular=angular,
        angular_name=angular_name,
    )
    if any(sha256(Path(path)) != value for path, value in fingerprints.items()):
        raise PipelineInputError('An original source changed while filtering floating geometry')
    report = {
        'source_files_unchanged': True,
        'source_layer': fitted.relative_to(output).as_posix(),
        'source_hashes': fingerprints,
        'components': component_count,
        'grounded_components': int(connections.keep.sum()),
        'removed_components': int((~connections.keep).sum()),
        'original_faces': sum(item['source_faces'] for item in published),
        'retained_faces': sum(item['kept_faces'] for item in published),
        'removed_faces': sum(item['source_faces'] - item['kept_faces'] for item in published),
        'checked_surface_pairs': connections.checked_pairs,
    }
    write_json(folder / 'report.json', report)
    from .simulation_grounded_verify import verify

    verify(output, angular=angular, angular_name=angular_name)
    summary = {name: value for name, value in report.items() if name != 'source_hashes'}
    print('GROUNDED_FILTER_COMPLETE ' + json.dumps(summary, ensure_ascii=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', type=Path, required=True)
    parser.add_argument(
        '--angular', action='store_true', help='Filter the angular view without FittedPlanes'
    )
    parser.add_argument('--angular-name', default='angular')
    args = parser.parse_args()
    remove_floating(args.map, angular=args.angular, angular_name=args.angular_name)


if __name__ == '__main__':
    main()
