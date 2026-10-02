"""Exercise an external URDF's real motor, odometry and camera paths in Gazebo."""

from __future__ import annotations

import argparse
from pathlib import Path

from usim.ports.gazebo import GazeboSimulator
from usim.simulation import SimulationConfig


def main() -> None:
    """Run the bounded Gazebo smoke for a prepared external robot description."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world', type=Path, required=True)
    parser.add_argument('--robot-urdf', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--wheel-radius', type=float, required=True)
    parser.add_argument('--wheel-separation', type=float, required=True)
    parser.add_argument('--base-link', default='base_link')
    parser.add_argument('--left-joint', default='left_wheel_joint')
    parser.add_argument('--right-joint', default='right_wheel_joint')
    parser.add_argument('--left-joints', nargs='+')
    parser.add_argument('--right-joints', nargs='+')
    parser.add_argument('--image', default='usim-gazebo:local')
    args = parser.parse_args()
    config = SimulationConfig(
        world=args.world,
        robot_urdf=args.robot_urdf,
        base_link=args.base_link,
        left_joint=args.left_joint,
        right_joint=args.right_joint,
        left_joints=tuple(args.left_joints) if args.left_joints else None,
        right_joints=tuple(args.right_joints) if args.right_joints else None,
        wheel_radius=args.wheel_radius,
        wheel_separation=args.wheel_separation,
        headless=True,
        max_seconds=60,
        camera_width=320,
        camera_height=240,
    )
    GazeboSimulator(image=args.image).smoke(config, args.out_dir)


if __name__ == '__main__':
    main()
