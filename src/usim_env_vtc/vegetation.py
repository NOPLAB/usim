"""Derive local park tree placements from registered 2019 LiDAR canopy support."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

if not __package__:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'usim_env_vtc'
from .pcd import PUBLIC_HASHES, open_records, sha256, xyz


@dataclass(frozen=True, slots=True)
class Tree:
    x: float
    y: float
    height: float
    radius: float
    supported_points: int


CELL_METERS: Final = 1.0


def relative_heights(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Estimate local measured-ground offsets without using simplified VTC heights."""
    from scipy.ndimage import distance_transform_edt, median_filter, minimum_filter
    from scipy.stats import binned_statistic_2d

    origin = np.floor(points[:, :2].min(0)) - 1
    cells = np.floor((points[:, :2] - origin) / CELL_METERS).astype(np.int64)
    shape = tuple((cells.max(0) + 2).tolist())
    # Local PCD ground, not VTC's simplified terrain, defines measured tree height.
    ground, _, _, _ = binned_statistic_2d(
        cells[:, 0],
        cells[:, 1],
        points[:, 2],
        statistic=lambda values: float(np.quantile(values, 0.1)),
        bins=[np.arange(shape[0] + 1), np.arange(shape[1] + 1)],
    )
    missing = ~np.isfinite(ground)
    nearest = distance_transform_edt(missing, return_distances=False, return_indices=True)
    ground[missing] = ground[tuple(nearest[:, missing])]
    ground = median_filter(minimum_filter(ground, size=5), size=3)
    return points[:, 2] - ground[cells[:, 0], cells[:, 1]]


def infer_trees(points: NDArray[np.float64]) -> list[Tree]:
    """Find volumetric crowns, rejecting flat roofs and narrow planar facades."""
    from scipy.ndimage import gaussian_filter, label, maximum_filter
    from scipy.spatial import cKDTree

    origin = np.floor(points[:, :2].min(0)) - 1
    cells = np.floor((points[:, :2] - origin) / CELL_METERS).astype(np.int64)
    shape = tuple((cells.max(0) + 2).tolist())
    relative = relative_heights(points)
    keep = (relative > 3) & (relative < 25)
    elevated, bins, heights = points[keep], cells[keep], relative[keep]
    upper = np.zeros(shape)
    lower = np.full(shape, np.inf)
    np.maximum.at(upper, (bins[:, 0], bins[:, 1]), heights)
    np.minimum.at(lower, (bins[:, 0], bins[:, 1]), heights)
    # A roof's constant height cannot supply a three-dimensional crown.
    crown = (upper - lower) > 1.2
    smooth = gaussian_filter(np.where(crown, upper, 0), sigma=1)
    maxima = (smooth == maximum_filter(smooth, size=7)) & (smooth > 4)
    groups, count = label(maxima)
    seeds = []
    for group in range(1, count + 1):
        positions = np.argwhere(groups == group)
        chosen = positions[np.argmax(smooth[tuple(positions.T)])]
        seeds.append(origin + chosen + 0.5)
    if not seeds:
        return []
    distance, ownership = cKDTree(np.asarray(seeds)).query(elevated[:, :2], workers=4)
    trees = []
    for index, center in enumerate(seeds):
        members = (ownership == index) & (distance < 7)
        cloud, h = elevated[members], heights[members]
        if len(cloud) < 100:
            continue
        covariance = np.cov(cloud[:: max(1, len(cloud) // 2000)].T)
        eigen = np.linalg.eigvalsh(covariance)
        radius = float(np.quantile(np.linalg.norm(cloud[:, :2] - center, axis=1), 0.9))
        if eigen[0] / eigen.sum() < 0.025 or radius < 1.3:
            continue
        trees.append(
            Tree(
                float(center[0]),
                float(center[1]),
                float(np.quantile(h, 0.98)),
                radius,
                len(cloud),
            )
        )
    return trees


def main() -> None:
    """Create streamed tree payloads above the unchanged, road-corrected world."""
    from PIL import Image
    from pxr import Gf, Usd, UsdGeom

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--world', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--terrain', type=Path, required=True)
    parser.add_argument('--prototype', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--with-structures', action='store_true')
    parser.add_argument('--bounds', nargs=4, type=float, default=[-340, 30, -20, 290])
    args = parser.parse_args()
    registration = json.loads(args.registration.read_text(encoding='utf-8'))
    if sha256(args.source) != PUBLIC_HASHES['map_tc19_o085_f-04_t05.pcd']:
        raise ValueError('Tree extraction requires the original verified 2019 PCD')
    matrix = np.asarray(registration['source19_to_vtc'], dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('Require validated 2019-to-VTC homogeneous registration')
    records, _ = open_records(args.source)
    chunks = []
    low_x, low_y, high_x, high_y = args.bounds
    for start in range(0, len(records), 250000):
        source = xyz(records[start : start + 250000]).astype(np.float64)
        points = source @ matrix[:3, :3].T + matrix[:3, 3]
        selected = (
            (points[:, 0] > low_x)
            & (points[:, 0] < high_x)
            & (points[:, 1] > low_y)
            & (points[:, 1] < high_y)
        )
        chunks.append(points[selected])
    points = np.concatenate(chunks)
    base = Usd.Stage.Open(str(args.world.resolve()), Usd.Stage.LoadNone)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    # Existing building geometry must not become inferred vegetation.
    for prim in base.GetPrimAtPath('/World/Structures/VTC').GetChildren():
        if any(name in prim.GetName() for name in ('Tree', 'Pine', 'Bush')):
            continue
        bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        lo, hi = np.asarray(bounds.GetMin()), np.asarray(bounds.GetMax())
        if (
            np.isfinite(lo).all()
            and (hi - lo > 2).all()
            and lo[0] < high_x
            and hi[0] > low_x
            and lo[1] < high_y
            and hi[1] > low_y
        ):
            inside = ((points[:, :2] >= lo[:2] - 0.5) & (points[:, :2] <= hi[:2] + 0.5)).all(1)
            points = points[~inside]
    trees = infer_trees(points)
    print(f'VEGETATION_CANDIDATES {len(trees)} cropPoints={len(points)}', flush=True)
    posts = []
    if args.with_structures:
        from .structures import infer_posts

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
        from .structures import author_posts

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
        'sourceSha256': registration['source_sha256'],
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


if __name__ == '__main__':
    main()
