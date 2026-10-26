"""Generate a regional collision/render map from PCD or PLY in meter/Z-up coordinates."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from . import MapConfig, Source, generate
from .models import PipelineInputError, Support


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--tile-size', type=float, default=30.0)
    parser.add_argument('--halo', type=float, default=1.5)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--normal-policy', choices=('auto', 'supplied'), default='auto')
    parser.add_argument(
        '--spacing',
        type=float,
        default=0.1,
        help='Nominal point spacing in meters; scales geometric support limits',
    )
    parser.add_argument('--license', default='unspecified')
    parser.add_argument('--attribution', default='')
    parser.add_argument('--source-url', default='')
    parser.add_argument('--source-catalog', default='')
    parser.add_argument('--sha256')
    parser.add_argument('--frame', default='source XYZ')
    args = parser.parse_args(argv)
    scale = args.spacing / 0.1
    base = Support()
    try:
        support = replace(
            base,
            radii=tuple(r * scale for r in base.radii),
            max_edge=base.max_edge * scale,
            midpoint_distance=base.midpoint_distance * scale,
            centroid_distance=base.centroid_distance * scale,
            min_double_area=base.min_double_area * scale**2,
        )
        config = MapConfig(
            tile_size=args.tile_size,
            halo=args.halo,
            workers=args.workers,
            normal_policy=args.normal_policy,
            frame=args.frame,
            support=support,
        )
        generate(
            Source(
                args.source,
                license=args.license,
                attribution=args.attribution,
                url=args.source_url,
                catalog=args.source_catalog,
                expected_sha256=args.sha256,
            ),
            args.output,
            config,
        )
    except PipelineInputError as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
