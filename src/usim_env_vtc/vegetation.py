"""VTC vegetation CLI; shared inference and VTC placement stay separate."""

from __future__ import annotations

import argparse
from pathlib import Path

if not __package__:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'usim_env_vtc'

from .vegetation_input import VegetationConfig
from .vegetation_placement import generate


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--world', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--terrain', type=Path, required=True)
    parser.add_argument('--prototype', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--with-structures', action='store_true')
    parser.add_argument('--bounds', nargs=4, type=float, default=[-340, 30, -20, 290])
    args = parser.parse_args(argv)
    generate(
        VegetationConfig(
            source=args.source,
            registration=args.registration,
            world=args.world,
            index=args.index,
            terrain=args.terrain,
            prototype=args.prototype,
            out=args.out,
            with_structures=args.with_structures,
            bounds=(args.bounds[0], args.bounds[1], args.bounds[2], args.bounds[3]),
        )
    )


if __name__ == '__main__':
    main()
