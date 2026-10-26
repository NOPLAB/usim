"""Download published originals or generate dataset-specific environments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from usim_env_pipeline import MapConfig

from .api import ENVIRONMENTS, DownloadRequest, GenerateRequest, get_environment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list', help='List datasets and original asset keys/names')
    download = commands.add_parser('download', help='Acquire unchanged published originals')
    download.add_argument('dataset', choices=ENVIRONMENTS)
    download.add_argument('--output-dir', type=Path, required=True)
    download.add_argument('assets', nargs='*', metavar='ASSET')
    generate = commands.add_parser('generate', help='Generate a dataset-specific environment')
    generate.add_argument('dataset', choices=ENVIRONMENTS)
    generate.add_argument('--source', type=Path, required=True)
    generate.add_argument('--output', type=Path, required=True)
    generate.add_argument('--tile-size', type=float, default=30.0)
    generate.add_argument('--halo', type=float, default=1.5)
    generate.add_argument('--workers', type=int, default=4)
    generate.add_argument('--reference-2018', type=Path)
    generate.add_argument('--measured-only', action='store_true')
    generate.add_argument('--resume', action='store_true')
    arguments = list(sys.argv[1:] if argv is None else argv)
    # Intermixed parsing permits the documented trailing ASSET after --output-dir.
    args = (
        download.parse_intermixed_args(arguments[1:])
        if arguments and arguments[0] == 'download'
        else parser.parse_args(arguments)
    )
    if getattr(args, 'command', None) == 'list':
        for name in ENVIRONMENTS:
            environment = get_environment(name)
            print(
                json.dumps({'dataset': name, 'assets': dict(environment.assets)}, sort_keys=True)
            )
        return 0
    try:
        environment = get_environment(args.dataset)
        if arguments[0] == 'download':
            for record in environment.download(
                DownloadRequest(args.output_dir, tuple(args.assets))
            ):
                print(json.dumps(record, sort_keys=True))
        else:
            request = GenerateRequest(
                args.source,
                args.output,
                MapConfig(
                    tile_size=args.tile_size,
                    halo=args.halo,
                    workers=args.workers,
                    resume=args.resume,
                ),
                args.reference_2018,
                not args.measured_only,
            )
            print(json.dumps(environment.generate(request), sort_keys=True))
    except ValueError as error:
        parser.error(str(error))
    return 0
