"""Generate the complete Tsudanuma measured map through the shared CPU pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if not __package__:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'usim_env_tsudanuma'

from usim_env_pipeline import MapConfig, Source, generate
from usim_env_pipeline.models import PipelineInputError, Summary

from .sources import ASSETS, ATTRIBUTION, CATALOG_URL, LICENSE, SOURCE_URLS


def run(args: argparse.Namespace) -> Summary:
    """Keep every source row; estimate working normals because the map has none."""
    acquisition_path = args.source.with_name(args.source.name + '.acquisition.json')
    expected = None
    if acquisition_path.exists():
        acquisition = json.loads(acquisition_path.read_text(encoding='utf-8'))
        expected = acquisition['sha256']
    config = MapConfig(
        tile_size=args.tile_size,
        halo=args.halo,
        workers=args.workers,
        normal_policy='auto',
        frame='raw Tsudanuma 2020 XYZ',
        resume=getattr(args, 'resume', False),
    )
    summary = generate(
        Source(
            args.source,
            key='map',
            license=LICENSE,
            attribution=ATTRIBUTION,
            url=SOURCE_URLS['map'],
            catalog=CATALOG_URL,
            expected_sha256=expected,
        ),
        args.output,
        config,
    )
    manifest_path = args.output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['catalogue_points'] = ASSETS['map'].get('points')
    manifest['actual_points'] = summary['records']
    manifest['sensor_data'] = {
        'url': SOURCE_URLS['sensor'],
        'role': 'separate raw sensor archive, not fused or required for the published map',
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--tile-size', type=float, default=30)
    parser.add_argument('--halo', type=float, default=1.5)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    try:
        run(args)
    except PipelineInputError as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
