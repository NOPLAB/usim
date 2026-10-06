"""Keep one native GUI process for the user's complete map inspection."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> None:
    """Launch once; region residency changes belong to the existing native window."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--world', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--region', type=int, nargs=2, default=(4, 4))
    parser.add_argument('--grid', type=int, default=8)
    parser.add_argument('--oblique', action='store_true')
    parser.add_argument(
        '--probe-region',
        type=int,
        nargs=2,
        help='Move the real camera across residency in the same process, then exit',
    )
    args = parser.parse_args()
    command = [
        sys.executable,
        '-u',
        str(Path(__file__).with_name('native_view.py')),
        '--world',
        str(args.world),
        '--index',
        str(args.index),
        '--out',
        str(args.out),
        '--region',
        *map(str, args.region),
        '--grid',
        str(args.grid),
    ]
    if not args.probe_region:
        command.append('--interactive')
    if args.oblique:
        command.append('--oblique')
    if args.probe_region:
        command += ['--pan-to', *map(str, args.probe_region)]
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
