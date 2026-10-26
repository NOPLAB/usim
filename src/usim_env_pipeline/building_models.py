"""Replace measured building facades with closed cuboids; keep ground independent."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .building_fit import facade_components, fit_cuboids, merge_facades
from .building_types import BuildingConfig, BuildingReport, Facade
from .models import PipelineInputError
from .pcd import open_records, sha256, xyz
from .storage import write_json
from .usd import LayerMetadata


def reconstruct(output: Path, config: BuildingConfig = BuildingConfig()) -> BuildingReport:
    """Generate an auditable building-only layer from existing measured plane support."""
    from .building_output import publish

    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    source = manifest['sources'][manifest['primary_source']]
    fingerprint = sha256(output / source['path'])
    if fingerprint != source['sha256']:
        raise PipelineInputError('Original building source hash changed')
    planar = json.loads((output / 'planar-report.json').read_text(encoding='utf-8'))
    if planar['source_sha256'] != fingerprint:
        raise PipelineInputError('Building plane archives do not match the original source')
    records, _ = open_records(output / source['path'])
    points = xyz(records)
    facades: list[Facade] = []
    for number in range(planar['planes']):
        with np.load(output / 'optimized/planes' / f'plane_{number:04}.npz') as archive:
            facades.extend(
                facade_components(number, archive['equation'], archive['ids'], points, config)
            )
    facades = merge_facades(facades, points, config)
    if not facades:
        raise PipelineInputError('No connected, supported building facades satisfy the thresholds')
    models = fit_cuboids(facades, points, config)
    folder = output / 'building-models'
    (folder / 'facades').mkdir(parents=True, exist_ok=True)
    write_json(folder / 'config.json', asdict(config))
    rows = []
    for number, facade in enumerate(facades):
        name = f'facade_{number:04}'
        np.savez_compressed(
            folder / 'facades' / f'{name}.npz',
            normal=facade.normal,
            offset=facade.offset,
            lower=facade.lower,
            upper=facade.upper,
            support_ids=facade.support_ids,
            source_planes=np.asarray(facade.source_planes),
        )
        rows.append(
            {
                'name': name,
                'source_planes': list(facade.source_planes),
                'normal': facade.normal.tolist(),
                'offset': facade.offset,
                'lower': facade.lower.tolist(),
                'upper': facade.upper.tolist(),
                'support_records': len(facade.support_ids),
                'observed_coverage': facade.coverage,
            }
        )
    write_json(folder / 'facades.json', rows)
    entries = json.loads((output / 'authored-index.json').read_text(encoding='utf-8'))
    ground = {
        entry['payload']: sha256(output / entry['payload'])
        for entry in entries
        if entry['path'].startswith('/World/Ground/')
    }
    write_json(folder / 'ground-source.json', ground)
    residual, replaced = publish(
        output,
        facades,
        models,
        config,
        LayerMetadata(
            source['license'],
            source['attribution'],
            manifest['source_catalog'],
            manifest['frame'],
            'measured facade support; inferred closed cuboid building approximations',
        ),
    )
    report: BuildingReport = {
        'source_sha256': fingerprint,
        'source_records': len(records),
        'input_planes': planar['planes'],
        'facades': len(facades),
        'building_cuboids': sum(model.kind == 'building' for model in models),
        'facade_cuboids': sum(model.kind == 'facade' for model in models),
        'supported_records': len(
            np.unique(np.concatenate([wall.support_ids for wall in facades]))
        ),
        'world': 'buildings.usda',
        'authority': 'building-models/model.brep',
        'ground_modified': False,
        'residual_triangles': residual,
        'replaced_triangles': replaced,
    }
    write_json(output / 'building-report.json', report)
    print('BUILDING_RECONSTRUCTION_PASSED ' + json.dumps(report), flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse
    from dataclasses import replace

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', type=Path, required=True)
    parser.add_argument('--gap', type=float, default=1.0)
    parser.add_argument('--wall-thickness', type=float, default=0.3)
    parser.add_argument('--minimum-height', type=float, default=2.5)
    parser.add_argument('--minimum-width', type=float, default=3.0)
    parser.add_argument('--minimum-coverage', type=float, default=0.25)
    args = parser.parse_args(argv)
    try:
        config = replace(
            BuildingConfig(),
            gap=args.gap,
            wall_thickness=args.wall_thickness,
            minimum_height=args.minimum_height,
            minimum_width=args.minimum_width,
            minimum_coverage=args.minimum_coverage,
        )
        reconstruct(args.map, config)
    except PipelineInputError as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
