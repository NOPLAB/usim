"""CLI adapter for the mobile simulation library; engines load only at execution."""

from __future__ import annotations

import argparse
from dataclasses import MISSING, fields
from pathlib import Path

from usim.simulation import Ros2Config, SimulationConfig


DEFAULTS = {
    field.name: field.default for field in fields(SimulationConfig) if field.default is not MISSING
}
DEFAULTS.update({field.name: field.default for field in fields(Ros2Config)})
DEFAULTS.update(
    robot_prim_path='/World/Robot',
    environment_prim_path='/World/Environment',
    camera_prim_path='/World/Mobile_Camera',
    ground_name='Ground',
    contact_out=None,
)


def configured(args) -> SimulationConfig:
    """Adapt argparse or legacy arguments to the shared typed library configuration."""
    if isinstance(args, SimulationConfig):
        return args
    values = {**DEFAULTS, **vars(args)}
    ros = (
        None
        if values.get('no_ros', False)
        else Ros2Config(**{field.name: values[field.name] for field in fields(Ros2Config)})
    )
    values.update(
        ros=ros,
        camera_offset=tuple(values['camera_offset']),
        camera_enabled=not values.get('physics_only', False),
    )
    return SimulationConfig(
        **{field.name: values[field.name] for field in fields(SimulationConfig)}
    )


def run(args) -> None:
    configuration = configured(args)
    if args.backend == 'gazebo':
        from usim.ports.gazebo import GazeboSimulator

        GazeboSimulator().run(configuration)
    else:
        from usim.ports.isaac import IsaacSimulator

        IsaacSimulator(
            robot_prim_path=args.robot_prim_path,
            environment_prim_path=args.environment_prim_path,
            camera_prim_path=args.camera_prim_path,
            ground_name=args.ground_name,
            contact_out=args.contact_out,
        ).run(configuration)


def add_arguments(parser, *, defaults: dict | None = None) -> None:
    defaults = {**DEFAULTS, **(defaults or {})}
    parser.add_argument('--world', type=Path, required=True)
    parser.add_argument('--robot-urdf', type=Path, required=True)
    parser.add_argument('--wheel-radius', type=float, default=defaults['wheel_radius'])
    parser.add_argument('--wheel-separation', type=float, default=defaults['wheel_separation'])
    parser.add_argument('--left-joint', default=defaults['left_joint'])
    parser.add_argument('--right-joint', default=defaults['right_joint'])
    parser.add_argument('--base-link', default=defaults['base_link'])
    parser.add_argument(
        '--camera-offset',
        nargs=3,
        type=float,
        metavar=('X', 'Y', 'Z'),
        default=defaults['camera_offset'],
    )
    parser.add_argument('--camera-width', type=int, default=defaults['camera_width'])
    parser.add_argument('--camera-height', type=int, default=defaults['camera_height'])
    parser.add_argument('--camera-hz', type=float, default=defaults['camera_hz'])
    parser.add_argument('--robot-name', default=defaults['robot_name'])
    parser.add_argument(
        '--robot-prim-path',
        '--prim-path',
        dest='robot_prim_path',
        default=defaults['robot_prim_path'],
    )
    parser.add_argument('--environment-prim-path', default=defaults['environment_prim_path'])
    parser.add_argument('--camera-prim-path', default=defaults['camera_prim_path'])
    parser.add_argument('--ground-name', default=defaults['ground_name'])
    for option in ('cmd_vel_topic', 'odom_topic', 'rgb_topic', 'depth_topic', 'motor_service'):
        parser.add_argument('--' + option.replace('_', '-'), default=defaults[option])
    parser.add_argument(
        '--no-ros',
        action='store_true',
        help='run standalone physics and camera without loading ROS',
    )
    parser.add_argument('--headless', action='store_true')
    parser.add_argument(
        '--physics-only', action='store_true', help='run wheels without camera frames'
    )
    parser.add_argument('--max-seconds', type=float, default=0)
    parser.add_argument(
        '--contact-out', type=Path, help='write obstacle-contact onsets from Isaac PhysX'
    )
    parser.set_defaults(node_name=defaults['node_name'])


def convert(args) -> dict:
    from usim.ports.isaac.worlds import convert_world

    return convert_world(args.world, args.out)


def register_world(commands: argparse._SubParsersAction) -> None:
    if 'convert-world' not in commands.choices:
        world = commands.add_parser('convert-world', help='convert static SDF boxes to USD')
        world.add_argument('--world', type=Path, required=True)
        world.add_argument('--out', type=Path, required=True)
        world.set_defaults(handler=convert)


def register(commands: argparse._SubParsersAction) -> None:
    simulate = commands.add_parser('simulate', help='simulate a mobile robot with ROS 2')
    add_arguments(simulate)
    simulate.add_argument('--backend', choices=('isaac', 'gazebo'), default='isaac')
    simulate.set_defaults(handler=run)
    register_world(commands)
