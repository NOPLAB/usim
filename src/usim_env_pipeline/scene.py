"""Original appearance and supported vegetation over an immutable measured map."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import TypedDict

import numpy as np

from .appearance import surface_colors
from .features import infer_posts, infer_trees, relative_heights
from .models import Bounds, Payload, PipelineInputError, Support
from .pcd import open_records, xyz
from .storage import tile_name, write_json
from .structures import author_posts
from .trees import author_tree
from .usd import LayerMetadata, stage_base


class SceneReport(TypedDict):
    world: str
    index: str
    source_records: int
    measured_faces: int
    measured_regions: int
    point_only_regions: int
    authored_trees: int
    tree_cells: int
    tree_prototypes: int
    authored_posts: int
    positions_unchanged: bool
    collision_faces_unchanged: bool
    ground_triangles: int
    ground_source_vertices: int
    ground_regions: int
    ground_projected_area: float


def decorate(
    output: Path, *, infer_vegetation: bool = True, measured_ground: bool = False
) -> SceneReport:
    """Add original colors/models; source geometry and raw archives remain untouched."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt
    from scipy.spatial import cKDTree

    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    source = manifest['sources'][manifest['primary_source']]
    records, _ = open_records(output / source['path'])
    positions = xyz(records).astype(np.float64)
    heights = relative_heights(positions)
    metadata = LayerMetadata(
        source['license'],
        source['attribution'],
        manifest['source_catalog'],
        manifest['frame'],
        'original procedural surface colors; synthetic windows, not measured facade labels',
    )
    folder = output / 'authored'
    (folder / 'surfaces').mkdir(parents=True, exist_ok=True)
    (folder / 'trees').mkdir(exist_ok=True)
    (folder / 'models').mkdir(exist_ok=True)
    entries: list[Bounds] = []
    world = stage_base(output / 'authored.usda', '/World', metadata)
    UsdGeom.Xform.Define(world, '/World/Regions')
    for entry in json.loads((output / 'index.json').read_text(encoding='utf-8')):
        name = entry['path'].rsplit('/', 1)[-1]
        with np.load(output / 'meshes' / f'{name}.npz') as mesh:
            ids, normals, faces = mesh['ids'], mesh['normals'], mesh['faces']
        cell = stage_base(folder / 'surfaces' / f'{name}.usdc', '/Tile', metadata)
        cell.GetDefaultPrim().GetReferences().AddReference(f'../../meshes/{name}.usdc', '/Tile')
        geometry = UsdGeom.Mesh.Get(cell, '/Tile/Surface')
        geometry.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(
            Vt.Vec3fArray.FromNumpy(
                surface_colors(positions[ids].astype(np.float32), normals, heights[ids])
            )
        )
        if not (
            np.array_equal(geometry.GetPointsAttr().Get(), positions[ids])
            and np.array_equal(geometry.GetFaceVertexIndicesAttr().Get(), faces.ravel())
            and UsdPhysics.CollisionAPI(geometry.GetPrim()).GetCollisionEnabledAttr().Get()
            and UsdPhysics.MeshCollisionAPI(geometry.GetPrim()).GetApproximationAttr().Get()
            == 'none'
        ):
            raise PipelineInputError('Original measured positions/faces/collision changed')
        cell.GetRootLayer().Save()
        path = entry['path']
        payload = f'authored/surfaces/{name}.usdc'
        UsdGeom.Xform.Define(world, path).GetPrim().GetPayloads().AddPayload(payload, '/Tile')
        entries.append(
            Payload(
                path=path,
                min=entry['min'],
                max=entry['max'],
                payload=payload,
            )
        )
    print(f'APPEARANCE_PASSED regions={len(entries)}', flush=True)
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    unmeshed = set(summary['unmeshed_tiles'])
    UsdGeom.Xform.Define(world, '/World/PointRegions')
    for entry in json.loads((output / 'point-index.json').read_text(encoding='utf-8')):
        # BPA uses only part of a tile: keep unsupported facade and ground returns too.
        UsdGeom.Xform.Define(world, entry['path']).GetPrim().GetPayloads().AddPayload(
            entry['payload'], '/Tile'
        )
        entries.append(entry)
    trees = infer_trees(positions) if infer_vegetation else []
    tree_info = []
    cells: dict[tuple[int, int], Usd.Stage] = {}
    for seed in range(4 if trees else 0):
        prototype = stage_base(folder / 'models' / f'tree_{seed}.usdc', '/Model', metadata)
        author_tree(prototype, '/Model', 'broadleaf', seed=seed)
        prototype.GetRootLayer().Save()
    if trees:
        nearest = cKDTree(positions[:, :2]).query(
            np.array([[tree.x, tree.y] for tree in trees]), workers=1
        )[1]
        ground = positions[nearest, 2] - heights[nearest]
        UsdGeom.Xform.Define(world, '/World/Vegetation')
        for number, (tree, z) in enumerate(zip(trees, ground)):
            key = (int(np.floor(tree.x / 64)), int(np.floor(tree.y / 64)))
            if key not in cells:
                cells[key] = stage_base(
                    folder / 'trees' / f'{tile_name(key)}.usdc', '/Cell', metadata
                )
            prim = UsdGeom.Xform.Define(cells[key], f'/Cell/Tree_{number:04}')
            prim.GetPrim().GetReferences().AddReference(
                f'../models/tree_{number % 4}.usdc', '/Model'
            )
            prim.AddTranslateOp().Set(Gf.Vec3d(tree.x, tree.y, float(z)))
            prim.AddScaleOp().Set(Gf.Vec3f(tree.radius, tree.radius, tree.height))
            prim.GetPrim().SetInstanceable(True)
            prim.GetPrim().SetCustomDataByKey('support:canopySamples', tree.supported_points)
            tree_info.append({**asdict(tree), 'ground': float(z)})
        for key, cell in sorted(cells.items()):
            cell.GetRootLayer().Save()
            bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            extent = bounds.ComputeWorldBound(cell.GetDefaultPrim()).ComputeAlignedRange()
            path = f'/World/Vegetation/{tile_name(key)}'
            payload = f'authored/trees/{tile_name(key)}.usdc'
            UsdGeom.Xform.Define(world, path).GetPrim().GetPayloads().AddPayload(payload, '/Cell')
            entries.append(
                Payload(
                    path=path,
                    min=list(extent.GetMin()),
                    max=list(extent.GetMax()),
                    payload=payload,
                )
            )
    posts = infer_posts(positions, trees, heights)
    if posts:
        indices = cKDTree(positions[:, :2]).query(
            np.array([[post.x, post.y] for post in posts]), workers=1
        )[1]
        ground = positions[indices, 2] - heights[indices]
        folder.joinpath('structures').mkdir(exist_ok=True)
        entries.extend(
            author_posts(
                world,
                [(post, float(z)) for post, z in zip(posts, ground)],
                folder / 'structures',
            )
        )
    ground_triangles, ground_vertices, ground_regions, ground_area = 0, 0, 0, 0.0
    if measured_ground:
        from .ground import author_ground, reconstruct_ground

        support_values = manifest['support']
        support_values['radii'] = tuple(support_values['radii'])
        ground = reconstruct_ground(
            positions.astype(np.float32), heights, Support(**support_values)
        )
        ground_entries = author_ground(
            world, output, positions.astype(np.float32), ground, manifest['tile_size'], metadata
        )
        entries.extend(ground_entries)
        ground_triangles, ground_vertices = len(ground.faces), len(ground.ids)
        ground_regions = len(ground_entries)
        triangles = positions[ground.ids][ground.faces]
        ground_area = float(
            np.abs(
                np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])[
                    :, 2
                ]
            ).sum()
            / 2
        )
    world.GetRootLayer().Save()
    write_json(output / 'authored-index.json', entries)
    write_json(folder / 'trees.json', tree_info)
    report: SceneReport = {
        'world': 'authored.usda',
        'index': 'authored-index.json',
        'source_records': summary['records'],
        'measured_faces': summary['triangles'],
        'measured_regions': summary['meshed_tiles'],
        'authored_trees': len(trees),
        'point_only_regions': len(unmeshed),
        'tree_cells': len(cells),
        'tree_prototypes': 4 if trees else 0,
        'authored_posts': len(posts),
        'positions_unchanged': True,
        'collision_faces_unchanged': True,
        'ground_triangles': ground_triangles,
        'ground_source_vertices': ground_vertices,
        'ground_regions': ground_regions,
        'ground_projected_area': ground_area,
    }
    write_json(output / 'authored-scene.json', report)
    print('AUTHORED_SCENE_PASSED ' + json.dumps(report), flush=True)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', type=Path, required=True)
    args = parser.parse_args(argv)
    decorate(args.map)


if __name__ == '__main__':
    main()
