"""Fit straight planar outlines and smooth continuous residual envelopes in a separate view."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .pcd import sha256
from .simulation_curve_surface import CurveSurface, curve_surface
from .simulation_fit_output import publish_surface
from .simulation_metrics import signed_volume
from .storage import tile_name, write_json
from .usd import LayerMetadata, stage_base


def fit(output: Path) -> None:
    """Preserve the baseline and its source authorities, authoring only a new fitted layer."""
    from pxr import Usd, UsdGeom

    from .simulation_polygon_slab import polygon_slab

    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    config = json.loads((output / 'simulation-models/config.json').read_text())
    cell = float(config['obstacles']['cell'])
    source = manifest['sources'][manifest['primary_source']]
    metadata = LayerMetadata(
        source['license'],
        source['attribution'],
        manifest['source_catalog'],
        manifest['frame'],
        'straight polygon outlines and continuous implicit curved envelopes',
    )
    folder = output / 'geometry-fit'
    folder.mkdir(parents=True, exist_ok=True)
    baseline_hash = sha256(output / 'simulation.usda')
    world = stage_base(output / 'simulation-fitted.usda', '/World', metadata)
    world.SetLoadRules(Usd.StageLoadRules.LoadNone())
    world.GetRootLayer().subLayerPaths = ['./simulation.usda']
    for name in ('SimulationObstacles', 'SimulationPlanes', 'SimulationUnresolved'):
        world.OverridePrim('/World/' + name).SetActive(False)
    rows = []
    plane_faces = 0
    for row in json.loads((output / 'simulation-models/planes/index.json').read_text()):
        name = row['name']
        authority = output / 'simulation-models/planes' / f'{name}.npz'
        with np.load(authority) as data:
            slab = polygon_slab(data['keys'], cell, 0.16)
            normal, offset, u, v = data['normal'], float(data['offset']), data['u'], data['v']
        positions = (
            slab.positions[:, :1] * u
            + slab.positions[:, 1:2] * v
            + (slab.positions[:, 2:3] - offset) * normal
        ).astype(np.float32)
        surface = CurveSurface(positions, slab.faces)
        path = folder / 'planes' / f'{name}.usdc'
        publish_surface(path, surface, metadata)
        np.savez_compressed(
            path.with_suffix('.npz'),
            positions=positions,
            faces=slab.faces,
            area_m2=slab.area_m2,
            outline_error_m=slab.outline_error_m,
        )
        prim_path = f'/World/FittedPlanes/{name}'
        UsdGeom.Xform.Define(world, prim_path).GetPrim().GetPayloads().AddPayload(
            './' + path.relative_to(output).as_posix(), '/Tile'
        )
        rows.append(
            {
                'path': prim_path,
                'payload': path.relative_to(output).as_posix(),
                'kind': 'plane',
                'authority': authority.relative_to(output).as_posix(),
                'authority_sha256': sha256(authority),
            }
        )
        plane_faces += len(slab.faces)
    print(f'GEOMETRY_FIT_PLANES models={len(rows)} faces={plane_faces}', flush=True)
    with np.load(output / 'simulation-models/obstacles.npz') as obstacles:
        keys = obstacles['keys']
    width = int(round(15 / cell))
    core_tiles = np.unique(keys[:, :2] // width, axis=0)
    tiles = np.unique(
        (core_tiles[:, None] + np.array([[x, y] for x in (-1, 0, 1) for y in (-1, 0, 1)])).reshape(
            -1, 2
        ),
        axis=0,
    )
    curved_faces = 0
    volume = 0.0
    for number, tile in enumerate(tiles):
        key = (int(tile[0]), int(tile[1]))
        surface = curve_surface(keys, key, cell)
        if not len(surface.faces):
            continue
        name = tile_name(key)
        path = folder / 'curves' / f'{name}.usdc'
        publish_surface(path, surface, metadata)
        np.savez_compressed(
            path.with_suffix('.npz'), positions=surface.positions, faces=surface.faces
        )
        prim_path = f'/World/FittedCurves/{name}'
        UsdGeom.Xform.Define(world, prim_path).GetPrim().GetPayloads().AddPayload(
            './' + path.relative_to(output).as_posix(), '/Tile'
        )
        rows.append(
            {
                'path': prim_path,
                'payload': path.relative_to(output).as_posix(),
                'kind': 'curve',
                'tile': list(key),
            }
        )
        curved_faces += len(surface.faces)
        volume += signed_volume(surface.positions, surface.faces)
        if number % 100 == 0:
            print(f'GEOMETRY_FIT_CURVES_PROGRESS tiles={number}/{len(tiles)}', flush=True)
    world.GetRootLayer().Save()
    write_json(folder / 'index.json', rows)
    write_json(
        folder / 'report.json',
        {
            'baseline_sha256': baseline_hash,
            'cell_m': cell,
            'curve_scale_m': 0.1,
            'envelope_radius_m': 0.2,
            'distance_truncation_m': 0.4,
            'curve_support_bound_m': 0.3,
            'plane_faces': plane_faces,
            'curve_faces': curved_faces,
            'curved_volume_m3': volume,
            'raw_residual_active': False,
        },
    )
    print(
        f'GEOMETRY_FIT_COMPLETE plane_faces={plane_faces} curve_faces={curved_faces}', flush=True
    )
    from .simulation_facades import fit_facades

    fit_facades(output)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', type=Path, required=True)
    fit(parser.parse_args().map)


if __name__ == '__main__':
    main()
