"""VTC source/provenance adapter for the generic CPU point-cloud map pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

if not __package__:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'usim_env_vtc'

from usim_env_pipeline import MapConfig, Source, generate
from usim_env_pipeline.models import PipelineInputError, Summary

from .sources import ATTRIBUTION as ATTRIBUTION, CATALOG as CATALOG, LICENSE as LICENSE
from .sources import LICENSE_NAME, LICENSE_URL, PROVIDER, PUBLIC_HASHES
from .sources import SOURCE_URLS as SOURCE_URLS


def vtc_source(path: Path, year: str) -> Source:
    return Source(
        path,
        key=year,
        license=LICENSE,
        attribution=ATTRIBUTION,
        url=SOURCE_URLS[year],
        catalog=CATALOG,
        expected_sha256=PUBLIC_HASHES.get(path.name),
        strict_xyzinormal=True,
    )


def run(args: argparse.Namespace) -> Summary:
    """Retain exact original 2019 records and a separate, unfused 2018 reference."""
    if not 20 <= args.tile_size <= 40:
        raise PipelineInputError('VTC requires 20-40 meter cores')
    references = (
        (vtc_source(args.reference_2018, '2018'),) if args.reference_2018 is not None else ()
    )
    config = MapConfig(
        tile_size=args.tile_size,
        halo=args.halo,
        workers=args.workers,
        normal_policy='supplied',
        frame='raw 2019 XYZ',
        references=references,
        resume=getattr(args, 'resume', False),
    )
    result = generate(vtc_source(args.source, '2019'), args.output, config)
    # Dataset-specific conditions remain the adapter's responsibility.
    with (args.output / 'NOTICE').open('a', encoding='utf-8') as notice:
        notice.write(
            f'\nUnofficial derived Tsukuba Challenge map; provider {PROVIDER}.\n'
            f'Data and derived geometry: {LICENSE_NAME}.\n'
            f'{LICENSE_URL}\n'
            '2019 is primary; 2018 is preserved separately, not silently fused.\n'
            f'Attribution and requested citation: {ATTRIBUTION}\n'
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--reference-2018', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--tile-size', type=float, default=30.0)
    parser.add_argument('--halo', type=float, default=1.5)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    try:
        run(parser.parse_args(argv))
    except PipelineInputError as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
