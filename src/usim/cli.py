"""Robot simulation and robot authoring commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Iterable

from usim.robot import MobileRobot, render_robot

Registrar = Callable[['argparse._SubParsersAction[argparse.ArgumentParser]'], None]


def _backends(args: argparse.Namespace) -> dict[str, list[str]]:
    from importlib.metadata import entry_points

    from usim.factory import capabilities

    names = {entry.name for entry in entry_points(group='usim.runners')}
    names.update(entry.name for entry in entry_points(group='usim.simulators'))
    return {name: sorted(capabilities(name)) for name in sorted(names)}


def _create_robot(args: argparse.Namespace) -> dict[str, str | float | list[float]]:
    """Validate CLI geometry before writing the reusable robot description."""
    length, width, height = args.body_size
    x, y, z = args.camera_offset
    configuration = MobileRobot(
        name=args.name,
        wheel_radius=args.wheel_radius,
        wheel_separation=args.wheel_separation,
        wheel_width=args.wheel_width,
        body_size=(length, width, height),
        mass=args.mass,
        camera_offset=(x, y, z),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_robot(configuration), encoding='utf-8')
    return {
        'robot': configuration.name,
        'urdf': str(args.out.resolve()),
        'wheel_radius_m': configuration.wheel_radius,
        'wheel_separation_m': configuration.wheel_separation,
        'camera_offset_m': list(configuration.camera_offset),
    }


def _register_robot(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Expose a simulator-independent robot authoring command."""
    parser = commands.add_parser('create-robot', help='create a primitive differential-drive URDF')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--name', default='mobile_robot')
    parser.add_argument('--wheel-radius', type=float, default=0.08)
    parser.add_argument('--wheel-separation', type=float, default=0.32)
    parser.add_argument('--wheel-width', type=float, default=0.04)
    parser.add_argument(
        '--body-size',
        type=float,
        nargs=3,
        default=(0.45, 0.24, 0.10),
        metavar=('LENGTH', 'WIDTH', 'HEIGHT'),
    )
    parser.add_argument('--mass', type=float, default=6.0)
    parser.add_argument(
        '--camera-offset', type=float, nargs=3, default=(0.12, 0.0, 0.28), metavar=('X', 'Y', 'Z')
    )
    parser.set_defaults(handler=_create_robot)


def build_parser(registrars: Iterable[Registrar] | None = None) -> argparse.ArgumentParser:
    """Register the built-in ports, or explicitly supplied command extensions."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    _register_robot(commands)
    backends = commands.add_parser('backends', help='list installed engines and execution models')
    backends.set_defaults(handler=_backends)
    if registrars is None:
        from usim.runtime_cli import register
        from usim.episode_cli import register as register_episode

        registrars = (register, register_episode)
    for register in registrars:
        register(commands)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        result = args.handler(args)
    except (KeyError, ValueError, OSError) as error:
        parser.error(str(error))
    if result is not None:
        print(json.dumps(result))


if __name__ == '__main__':
    main()
