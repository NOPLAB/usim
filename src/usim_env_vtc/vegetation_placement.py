"""Ground inferred VTC features and author bounded scene payloads."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np

from usim_env_pipeline.features import Tree, infer_posts, infer_trees, relative_heights
from .vegetation_input import VegetationConfig, exclude_buildings, read_points


def generate(args: VegetationConfig) -> None:
    """Retain source scene geometry while adding terrain-grounded feature instances."""
    from PIL import Image
    from pxr import Gf, Usd, UsdGeom

    points, source_sha256 = read_points(args)
    base = Usd.Stage.Open(str(args.world.resolve()), Usd.Stage.LoadNone)
    points = exclude_buildings(base, points, args.bounds)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    trees = infer_trees(points)
    print(f'VEGETATION_CANDIDATES {len(trees)} cropPoints={len(points)}', flush=True)
    posts = []
    if args.with_structures:
        posts = infer_posts(points, trees, relative_heights(points))
        print(f'STRUCTURE_CANDIDATES {len(posts)}', flush=True)
    entries = json.loads(args.index.read_text(encoding='utf-8'))
    tiles = [entry for entry in entries if entry['path'].startswith('/World/Terrain/')]
    grounded = []
    heights_cache = {}
    masks_cache = {}
    for tree in trees:
        tile = next(
            (
                entry
                for entry in tiles
                if all(
                    entry['min'][axis] <= coordinate <= entry['max'][axis]
                    for axis, coordinate in enumerate((tree.x, tree.y))
                )
            ),
            None,
        )
        if tile is None:
            continue
        name = tile['path'].rsplit('/', 1)[1]
        if name not in heights_cache:
            heights_cache[name] = np.load(
                args.terrain / 'source-textures' / name / 'height_uint16.npy'
            )[::-1]
            with (
                Image.open(args.terrain / 'shader-masks' / f'{name}-0.png') as first,
                Image.open(args.terrain / 'shader-masks' / f'{name}-1.png') as second,
            ):
                masks, extra = np.asarray(first), np.asarray(second)
                masks_cache[name] = (
                    masks[..., 1].astype(np.uint16) + masks[..., 2] + extra[..., 1]
                )[::-1]
        col = int(np.clip(round((tree.x - tile['min'][0]) * 10), 0, 1008))
        row = int(np.clip(round((tree.y - tile['min'][1]) * 10), 0, 1008))
        z = (int(heights_cache[name][row, col]) - 32768) / 1280
        if z > -10 and masks_cache[name][row, col] < 128:
            grounded.append((tree, z))
    args.out.mkdir(parents=True, exist_ok=True)
    prototype = Usd.Stage.Open(str(args.prototype.resolve()))
    root = prototype.GetDefaultPrim()
    prototype_bounds = cache.ComputeWorldBound(root).ComputeAlignedRange()
    size = prototype_bounds.GetSize()
    cells: dict[tuple[int, int], list[tuple[Tree, float]]] = {}
    for tree, z in grounded:
        key = (int(np.floor(tree.x / 30)), int(np.floor(tree.y / 30)))
        cells.setdefault(key, []).append((tree, z))
    world = Usd.Stage.CreateNew(str((args.out / 'world.usda').resolve()), Usd.Stage.LoadNone)
    world.GetRootLayer().subLayerPaths = [
        Path(os.path.relpath(args.world.resolve(), args.out.resolve())).as_posix(),
    ]
    world.SetDefaultPrim(world.GetPrimAtPath('/World'))
    UsdGeom.SetStageUpAxis(world, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(world, 1)
    for key, members in cells.items():
        name = f't_{key[0]}_{key[1]}'.replace('-', 'm')
        filename = args.out / (name + '.usdc')
        cell = Usd.Stage.CreateNew(str(filename.resolve()))
        cell.SetDefaultPrim(UsdGeom.Xform.Define(cell, '/Cell').GetPrim())
        UsdGeom.SetStageUpAxis(cell, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(cell, 1)
        for index, (tree, z) in enumerate(members):
            xform = UsdGeom.Xform.Define(cell, f'/Cell/Tree_{index}')
            prim = xform.GetPrim()
            prim.GetReferences().AddReference(
                Path(os.path.relpath(args.prototype.resolve(), args.out.resolve())).as_posix(),
                str(root.GetPath()),
            )
            prim.SetInstanceable(True)
            scale_z = tree.height / size[2]
            xform.AddTranslateOp().Set(
                (
                    tree.x,
                    tree.y,
                    z - prototype_bounds.GetMin()[2] * scale_z,
                )
            )
            xform.AddRotateZOp().Set((index * 137.507764) % 360)
            xform.AddScaleOp().Set(
                Gf.Vec3f(
                    2 * tree.radius / max(size[0], size[1]),
                    2 * tree.radius / max(size[0], size[1]),
                    scale_z,
                )
            )
        cell.GetRootLayer().Save()
        path = '/World/Vegetation/' + name
        prim = world.DefinePrim(path, 'Xform')
        prim.GetPayloads().AddPayload(name + '.usdc', '/Cell')
        cache.Clear()
        bounds = cache.ComputeWorldBound(cell.GetDefaultPrim()).ComputeAlignedRange()
        entries.append(
            {
                'path': path,
                'asset': name + '.usdc',
                'prim': '/Cell',
                'min': list(bounds.GetMin()),
                'max': list(bounds.GetMax()),
            }
        )
    grounded_posts = []
    for post in posts:
        tile = next(
            (
                entry
                for entry in tiles
                if all(
                    entry['min'][axis] <= coordinate <= entry['max'][axis]
                    for axis, coordinate in enumerate((post.x, post.y))
                )
            ),
            None,
        )
        if tile is None:
            continue
        base.Load(tile['path'])
        mesh_cache = UsdGeom.XformCache()
        matrix = mesh_cache.GetLocalToWorldTransform(base.GetPrimAtPath(tile['path']))
        local = matrix.GetInverse().Transform(Gf.Vec3d(post.x, post.y, 0))
        ix = int(np.clip(np.floor(local[0] / 12.6), 0, 7))
        iy = int(np.clip(np.floor(-local[1] / 12.6), 0, 7))
        mesh = UsdGeom.Mesh(
            base.GetPrimAtPath(
                tile['path'] + f'/LandscapeComponent_{ix + 8 * iy}',
            )
        )
        col = int(np.clip(round(local[0] * 10) - ix * 126, 0, 126))
        row = int(np.clip(round(-local[1] * 10) - iy * 126, 0, 126))
        point = mesh.GetPointsAttr().Get()[row * 127 + col]
        z = float(matrix.Transform(Gf.Vec3d(*point))[2])
        if z > -10:
            grounded_posts.append((post, z))
    if grounded_posts:
        from usim_env_pipeline.structures import author_posts

        entries.extend(author_posts(world, grounded_posts, args.out))
    world.GetRootLayer().customLayerData = {
        'originalAssetsModified': False,
        'pointRecordsModified': False,
        'vegetationDerivedFrom': 'registered 2019 LiDAR crown support',
        'distribution': 'LOCAL inspection only; references mixed-license VTC tree artwork',
        'licenseAudit': 'docs/vtc/tsukuba-full-map/LICENSES.md',
    }
    world.GetRootLayer().Save()
    (args.out / 'world.index.json').write_text(json.dumps(entries, indent=2))
    report = {
        'source': str(args.source.resolve()),
        'sourceSha256': source_sha256,
        'bounds': args.bounds,
        'trees': [{**asdict(tree), 'groundZ': z} for tree, z in grounded],
        'treeCount': len(grounded),
        'payloadCount': len(cells),
        'prototype': str(args.prototype.resolve()),
        'prototypePrim': str(root.GetPath()),
        'unregistered2018Used': False,
        'originalAssetsModified': False,
        'placementAccuracyEstablished': False,
        'posts': [{**asdict(post), 'groundZ': z} for post, z in grounded_posts],
        'postCount': len(grounded_posts),
        'postIdentityEstablished': False,
    }
    (args.out / 'vegetation.json').write_text(json.dumps(report, indent=2))
    print(f'VEGETATION_COMPLETE trees={len(grounded)} cells={len(cells)}', flush=True)
