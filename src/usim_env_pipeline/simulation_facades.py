"""Replace inferred rectangular building walls with measured footprint slabs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .models import PipelineInputError
from .pcd import open_records, sha256, xyz
from .simulation_curve_surface import CurveSurface
from .simulation_fit_output import publish_surface
from .simulation_polygon_slab import polygon_slab
from .storage import write_json
from .usd import LayerMetadata


def fit_facades(output: Path) -> None:
    """Keep only occupied facade cells; never fill their bounding rectangles."""
    from pxr import Usd, UsdGeom

    manifest = json.loads((output / 'manifest.json').read_text())
    source = manifest['sources'][manifest['primary_source']]
    facades = json.loads((output / 'building-models/facades.json').read_text())
    points = np.empty((0, 3), np.float32)
    if facades:
        if sha256(output / source['path']) != source['sha256']:
            raise PipelineInputError('Original facade source changed')
        records, _ = open_records(output / source['path'])
        points = xyz(records)
    folder = output / 'geometry-fit'
    report = json.loads((folder / 'report.json').read_text())
    cell = float(report['cell_m'])
    metadata = LayerMetadata(
        source['license'],
        source['attribution'],
        manifest['source_catalog'],
        manifest['frame'],
        'measured facade footprints; no inferred rectangular walls',
    )
    stage = Usd.Stage.Open(str(output / 'simulation-fitted.usda'), Usd.Stage.LoadNone)
    stage.OverridePrim('/World/BuildingModels').SetActive(False)
    stage.RemovePrim('/World/FittedFacades')
    rows = [
        row
        for row in json.loads((folder / 'index.json').read_text())
        if not row['path'].startswith('/World/FittedFacades/')
    ]
    count = 0
    for facade in facades:
        name = facade['name']
        original = output / 'building-models/facades' / f'{name}.npz'
        with np.load(original) as archive:
            normal, offset, ids = (
                archive['normal'],
                float(archive['offset']),
                archive['support_ids'],
            )
        u, v = np.array([-normal[1], normal[0], 0.0]), np.array([0.0, 0.0, 1.0])
        support = points[ids].astype(np.float64)
        keys = np.unique(
            np.floor(np.column_stack((support @ u, support @ v)) / cell).astype(np.int64),
            axis=0,
        )
        slab = polygon_slab(keys, cell, 0.16)
        positions = (
            slab.positions[:, :1] * u
            + slab.positions[:, 1:2] * v
            + (slab.positions[:, 2:3] - offset) * normal
        ).astype(np.float32)
        path = folder / 'facades' / f'{name}.usdc'
        publish_surface(path, CurveSurface(positions, slab.faces), metadata)
        authority = path.with_suffix('.npz')
        np.savez_compressed(
            authority,
            positions=positions,
            faces=slab.faces,
            keys=keys,
            u=u,
            v=v,
            normal=normal,
            offset=offset,
            area_m2=slab.area_m2,
            outline_error_m=slab.outline_error_m,
        )
        prim_path = f'/World/FittedFacades/{name}'
        UsdGeom.Xform.Define(stage, prim_path).GetPrim().GetPayloads().AddPayload(
            './' + path.relative_to(output).as_posix(), '/Tile'
        )
        rows.append(
            {
                'path': prim_path,
                'payload': path.relative_to(output).as_posix(),
                'kind': 'plane',
                'authority': authority.relative_to(output).as_posix(),
                'authority_sha256': sha256(authority),
                'source_facade': original.relative_to(output).as_posix(),
                'source_facade_sha256': sha256(original),
            }
        )
        count += len(slab.faces)
        print(f'MEASURED_FACADE_PROGRESS {name}', flush=True)
    stage.GetRootLayer().Save()
    write_json(folder / 'index.json', rows)
    report['inferred_building_walls_active'] = False
    report['measured_facades'] = len(facades)
    report['measured_facade_faces'] = count
    write_json(folder / 'report.json', report)
    print(f'MEASURED_FACADES_FITTED models={len(facades)} faces={count}', flush=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', type=Path, required=True)
    fit_facades(parser.parse_args().map)


if __name__ == '__main__':
    main()
