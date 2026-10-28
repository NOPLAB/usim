"""Select the isolated SDK interpreter in the all-in-one container."""

from __future__ import annotations

import os
import sys

from usim.cli import build_parser


def main() -> None:
    """Reuse CLI parsing and replace this process with the selected SDK runtime."""
    parser = build_parser()
    args = parser.parse_args()
    engines: dict[str, str] = {
        'run': getattr(args, 'engine', ''),
        'simulate': getattr(args, 'backend', ''),
        'convert-world': 'isaacsim',
    }
    command: str = args.command
    engine = engines.get(command, '')
    interpreters = {
        '': sys.executable,
        'gazebo': sys.executable,
        'genesis': '/opt/usim/genesis/bin/python',
        'maniskill': '/opt/usim/maniskill/bin/python',
        'bullet': '/opt/usim/bullet/bin/python',
        'mujoco': '/opt/usim/mujoco/bin/python',
        'mujoco-warp': '/opt/usim/mujoco-warp/bin/python',
        'isaacsim': '/isaac-sim/python.sh',
    }
    if engine == 'isaacsim' and os.environ.get('ACCEPT_EULA') != 'Y':
        parser.error('Set ACCEPT_EULA=Y after accepting NVIDIA Isaac Sim terms')
    executable = interpreters[engine]
    os.execv(executable, [executable, '-m', 'usim.cli', *sys.argv[1:]])


if __name__ == '__main__':
    main()
