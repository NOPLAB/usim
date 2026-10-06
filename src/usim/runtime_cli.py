"""Native runtime commands; engines load only at execution."""

from __future__ import annotations

import argparse
import sys
import threading
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
    from usim.factory import create_runner

    configuration = configured(args)
    gazebo_options = {
        name: getattr(args, name)
        for name in ('engine', 'image', 'network', 'fastdds_profile')
        if getattr(args, name, None) is not None
    }
    lidar_options = {
        field: getattr(args, 'lidar_' + field)
        for field in (
            'update_rate',
            'horizontal_samples',
            'min_angle',
            'max_angle',
            'range_min',
            'range_max',
        )
        if getattr(args, 'lidar_' + field, None) is not None
    }
    lidar_link = getattr(args, 'lidar_link', None)
    lidar_frame = getattr(args, 'lidar_frame', None)
    lidar_topic = getattr(args, 'lidar_topic', None)
    if args.backend != 'gazebo' and (
        gazebo_options
        or lidar_link is not None
        or lidar_frame is not None
        or lidar_topic is not None
        or lidar_options
    ):
        raise ValueError('container and lidar options require --backend gazebo')
    if lidar_link is None and (
        lidar_frame is not None or lidar_topic is not None or lidar_options
    ):
        raise ValueError('lidar options require --lidar-link')
    if lidar_link is not None:
        if lidar_frame is None or lidar_topic is None:
            raise ValueError('--lidar-link requires --lidar-frame and --lidar-topic')
        from usim_gazebo import GazeboLidarConfig

        gazebo_options['lidar'] = GazeboLidarConfig(
            link_name=lidar_link, frame_name=lidar_frame, topic=lidar_topic, **lidar_options
        )
    if args.backend == 'gazebo':
        runner = create_runner('gazebo', **gazebo_options)
    else:
        runner = create_runner(
            'isaacsim',
            robot_prim_path=args.robot_prim_path,
            environment_prim_path=args.environment_prim_path,
            camera_prim_path=args.camera_prim_path,
            ground_name=args.ground_name,
            contact_out=args.contact_out,
        )
    if not getattr(args, 'stop_on_stdin', False):
        runner.run(configuration)
        return
    stop = threading.Event()

    def read_stop() -> None:
        while not stop.is_set():
            line = sys.stdin.readline()
            if not line or line.strip() == 'stop':
                stop.set()
                return

    threading.Thread(target=read_stop, daemon=True).start()
    runner.run(configuration, stop=stop)
    if not stop.is_set() and configuration.max_seconds == 0:
        raise RuntimeError('simulation runner exited before a stop request')


def add_arguments(
    parser,
    *,
    defaults: dict[
        str,
        str | float | int | bool | Path | Ros2Config | None | tuple[str, ...] | tuple[float, ...],
    ]
    | None = None,
) -> None:
    defaults = {**DEFAULTS, **(defaults or {})}
    parser.add_argument('--world', type=Path, required=True)
    parser.add_argument('--robot-urdf', type=Path, required=True)
    parser.add_argument('--wheel-radius', type=float, default=defaults['wheel_radius'])
    parser.add_argument('--wheel-separation', type=float, default=defaults['wheel_separation'])
    parser.add_argument('--left-joint', default=defaults['left_joint'])
    parser.add_argument('--right-joint', default=defaults['right_joint'])
    parser.add_argument(
        '--left-joints',
        nargs='+',
        default=defaults['left_joints'],
        help='left-side wheel group (overrides --left-joint)',
    )
    parser.add_argument(
        '--right-joints',
        nargs='+',
        default=defaults['right_joints'],
        help='right-side wheel group (overrides --right-joint)',
    )
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


def convert(args) -> dict[str, int | str | list[str]]:
    from usim_isaacsim.worlds import convert_world

    return convert_world(args.world, args.out)


def register_world(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    if 'convert-world' not in commands.choices:
        world = commands.add_parser('convert-world', help='convert static SDF boxes to USD')
        world.add_argument('--world', type=Path, required=True)
        world.add_argument('--out', type=Path, required=True)
        world.set_defaults(handler=convert)


def register(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    simulate = commands.add_parser('simulate', help='simulate a mobile robot with ROS 2')
    add_arguments(simulate)
    simulate.add_argument('--backend', choices=('isaac', 'isaacsim', 'gazebo'), default='isaacsim')
    simulate.add_argument(
        '--stop-on-stdin', action='store_true', help='stop on a stdin stop line or EOF'
    )
    simulate.add_argument('--engine', choices=('docker', 'podman'), help='Gazebo container engine')
    simulate.add_argument('--image', help='Gazebo container image')
    simulate.add_argument('--network', help='Gazebo container network')
    simulate.add_argument('--fastdds-profile', type=Path, help='Gazebo Fast DDS profile')
    simulate.add_argument('--lidar-link', help='enable Gazebo lidar on this robot link')
    simulate.add_argument('--lidar-frame', help='required frame name when lidar is enabled')
    simulate.add_argument('--lidar-topic', help='required ROS topic when lidar is enabled')
    for option in ('update-rate', 'min-angle', 'max-angle', 'range-min', 'range-max'):
        simulate.add_argument('--lidar-' + option, type=float)
    simulate.add_argument('--lidar-horizontal-samples', type=int)
    simulate.set_defaults(handler=run)
    register_world(commands)
