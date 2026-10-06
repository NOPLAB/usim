"""Create a separate, full-resolution road-relief correction for local VTC inspection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pxr import Usd


def author_road_mesh(stage: Usd.Stage, path: str, points: NDArray[np.float32]) -> None:
    """Author local mesh opinions, stronger than geometry introduced by parent references."""
    from pxr import UsdGeom, Vt

    mesh = UsdGeom.Mesh(stage.OverridePrim(path))
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.stack((points.min(0), points.max(0)))))


def repair_heights(heights: NDArray[np.float32], paved: NDArray[np.bool_]) -> NDArray[np.float32]:
    """Soften abrupt paved-interior relief without changing unpaved terrain or gentle grades."""
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    interior = distance_transform_edt(paved) >= 3
    abrupt = np.zeros(heights.shape, dtype=bool)
    for axis in (0, 1):
        left = (slice(None, -1), slice(None)) if axis == 0 else (slice(None), slice(None, -1))
        right = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
        edges = (np.abs(heights[right] - heights[left]) > 0.08) & interior[left] & interior[right]
        abrupt[left] |= edges
        abrupt[right] |= edges
    if not abrupt.any():
        return heights.copy()
    # The source lattice is 10 cm. A 1.5 m kernel replaces cliff-like jumps with
    # gradual source-derived transitions; this is a derived surface, not a new survey.
    weights = gaussian_filter(paved.astype(np.float32), sigma=15, mode='nearest')
    smooth = gaussian_filter(np.where(paved, heights, 0), sigma=15, mode='nearest')
    smooth /= np.maximum(weights, 1e-6)
    distance = distance_transform_edt(~abrupt)
    blend = np.clip((45 - distance) / 30, 0, 1)
    blend *= np.clip(distance_transform_edt(paved) / 3, 0, 1)
    result = heights.copy()
    changed = paved & (blend > 0)
    result[changed] += ((smooth - heights) * blend)[changed]
    return result


def main() -> None:
    """Retain source assets and topology, authoring only corrected points in new payload layers."""
    from PIL import Image
    from pxr import Usd, UsdGeom

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--terrain', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--world', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    tiles = manifest['tiles']
    height = np.empty((8065, 8065), dtype=np.float32)
    paved = np.zeros(height.shape, dtype=bool)
    for tile in tiles:
        name = tile['name']
        i, j = map(int, name.removeprefix('TC_x').split('_y'))
        region = (slice(j * 1008, j * 1008 + 1009), slice(i * 1008, i * 1008 + 1009))
        source = np.load(args.terrain / 'source-textures' / name / 'height_uint16.npy')
        height[region] = (source[::-1].astype(np.float32) - 32768) / 1280
        with (
            Image.open(args.terrain / 'shader-masks' / f'{name}-0.png') as first,
            Image.open(args.terrain / 'shader-masks' / f'{name}-1.png') as second,
        ):
            masks, extra = np.asarray(first), np.asarray(second)
            road = masks[..., 1].astype(np.uint16) + masks[..., 2] + extra[..., 1]
            paved[region] = road[::-1] >= 128
    corrected = repair_heights(height, paved)
    if not np.array_equal(corrected[~paved], height[~paved]):
        raise AssertionError('Road correction changed unpaved terrain')
    args.out.mkdir(parents=True, exist_ok=True)
    correction = Usd.Stage.CreateNew(str((args.out / 'road-correction.usdc').resolve()))
    world = Usd.Stage.CreateNew(str((args.out / 'world.usda').resolve()), load=Usd.Stage.LoadNone)
    world.GetRootLayer().subLayerPaths = [
        'road-correction.usdc',
        Path(os.path.relpath(args.world.resolve(), args.out.resolve())).as_posix(),
    ]
    world.SetDefaultPrim(world.GetPrimAtPath('/World'))
    UsdGeom.SetStageUpAxis(world, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(world, 1)
    changed_vertices = 0
    changed_components = 0
    entries = json.loads(args.index.read_text(encoding='utf-8'))
    index = {entry['path']: entry for entry in entries}
    for tile in tiles:
        name = tile['name']
        i, j = map(int, name.removeprefix('TC_x').split('_y'))
        original_path = (args.terrain / 'tiles' / f'{name}.usdc').resolve()
        source_stage = Usd.Stage.Open(str(original_path))
        for component in tile['components']:
            sx, sy = component['section']
            rows = j * 1008 + 1008 - sy - np.arange(127)
            cols = i * 1008 + sx + np.arange(127)
            z = corrected[np.ix_(rows, cols)].ravel()
            path = '/Tile/' + component['name']
            original_mesh = UsdGeom.Mesh(source_stage.GetPrimAtPath(path))
            points = np.array(original_mesh.GetPointsAttr().Get())
            changed = z != points[:, 2]
            if changed.any():
                points[:, 2] = z
                author_road_mesh(
                    correction, '/World/Terrain/' + name + '/' + component['name'], points
                )
                changed_vertices += int(changed.sum())
                changed_components += 1
        prim = world.OverridePrim('/World/Terrain/' + name)
        region = corrected[j * 1008 : j * 1008 + 1009, i * 1008 : i * 1008 + 1009]
        entry = index[str(prim.GetPath())]
        entry['min'][2] = min(entry['min'][2], float(region.min()))
        entry['max'][2] = max(entry['max'][2], float(region.max()))
        print(f'ROAD_TILE_WRITTEN {name}', flush=True)
    correction.GetRootLayer().Save()
    world.GetRootLayer().customLayerData = {
        'originalAssetsModified': False,
        'roadReliefCorrection': True,
        'geometryDecimation': False,
        'geographicSurveyAccuracyEstablished': False,
    }
    world.GetRootLayer().Save()
    # A shared border may borrow relief from the adjacent tile. Expand Z bounds
    # where needed so the existing frustum/residency contract remains conservative.
    (args.out / 'world.index.json').write_text(json.dumps(entries, indent=2), encoding='utf-8')
    delta = np.abs(corrected - height)
    report = {
        'originalAssetsModified': False,
        'geometryDecimation': False,
        'unmodifiedUnpavedSamples': True,
        'changedUniqueSamples': int(np.count_nonzero(delta)),
        'changedComponentVertices': changed_vertices,
        'changedComponents': changed_components,
        'maximumCorrectionMeters': float(delta.max()),
        'kernelSigmaMeters': 1.5,
        'sourceWorld': args.world.resolve().as_posix(),
        'correctedWorld': (args.out / 'world.usda').resolve().as_posix(),
    }
    (args.out / 'road-repair.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('ROAD_REPAIR_COMPLETE ' + json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
