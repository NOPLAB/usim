"""Reapply float32-safe mesh publication to owned fitted curves without refitting sources."""

from __future__ import annotations

import json
from pathlib import Path

from typing_extensions import assert_never

import numpy as np
from scipy.spatial import cKDTree

from .simulation_curve_surface import curve_surface, supported_surface
from .simulation_fit_output import publish_surface
from .simulation_fit_verify import SurfaceKind
from .simulation_metrics import signed_volume
from .storage import write_json
from .usd import LayerMetadata


def republish(output: Path, *, refit_curves: bool = False) -> None:
    """Update only the derived curve payloads; retain all planar fits and source layers."""
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    source = manifest['sources'][manifest['primary_source']]
    metadata = LayerMetadata(
        source['license'],
        source['attribution'],
        manifest['source_catalog'],
        manifest['frame'],
        'float32-safe continuous implicit curved envelopes',
    )
    folder = output / 'geometry-fit'
    rows = json.loads((folder / 'index.json').read_text())
    report = json.loads((folder / 'report.json').read_text())
    with np.load(output / 'simulation-models/obstacles.npz') as authority:
        keys = authority['keys']
    cell = float(report['cell_m'])
    support = cKDTree((keys + 0.5) * cell)
    count, removed, volume = 0, 0, 0.0
    for row in rows:
        match SurfaceKind(row['kind']):
            case SurfaceKind.PLANE:
                continue
            case SurfaceKind.CURVE:
                pass
            case unreachable:
                assert_never(unreachable)
        path = output / row['payload']
        tile = (int(row['tile'][0]), int(row['tile'][1]))
        if refit_curves:
            surface = curve_surface(keys, tile, cell)
        else:
            with np.load(path.with_suffix('.npz')) as previous:
                positions, faces = previous['positions'], previous['faces']
            surface = supported_surface(positions, faces, support)
            removed += len(faces) - len(surface.faces)
        publish_surface(path, surface, metadata)
        np.savez_compressed(
            path.with_suffix('.npz'), positions=surface.positions, faces=surface.faces
        )
        count += len(surface.faces)
        volume += signed_volume(surface.positions, surface.faces)
        print(f'GEOMETRY_FIT_REPUBLISH_TILE {row["payload"]}', flush=True)
    report['curve_faces'] = count
    report['curved_volume_m3'] = volume
    if not refit_curves:
        report['quantization_collapsed_faces_removed'] = (
            report.get('quantization_collapsed_faces_removed', 0) + removed
        )
    report['curve_support_bound_m'] = 0.3
    write_json(folder / 'report.json', report)
    print(f'GEOMETRY_FIT_REPUBLISHED faces={count} collapsed={removed}', flush=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', type=Path, required=True)
    parser.add_argument('--refit-curves', action='store_true')
    args = parser.parse_args()
    republish(args.map, refit_curves=args.refit_curves)


if __name__ == '__main__':
    main()
