"""Native scene commands; engine imports occur only on execution."""

from __future__ import annotations

import argparse
from pathlib import Path


def _run(args: argparse.Namespace) -> dict[str, str | int | list[int]]:
    import numpy as np

    from usim import SimulatorConfig, create_simulator

    if args.steps < 0 or (args.steps > 0 and args.action is None):
        raise ValueError('Positive --steps requires an explicit native --action vector')
    config = SimulatorConfig(
        env_id=args.env_id,
        backend=args.compute,
        render_mode=args.render_mode,
        n_envs=args.num_envs,
        max_episode_steps=max(args.steps, 1),
        robot_path=None if args.robot is None else str(args.robot.resolve()),
        robot_uid=args.robot_uid,
        arm_joint_names=tuple(args.arm_joints),
        gripper_joint_names=tuple(args.gripper_joints),
        obs_mode=args.obs_mode,
        fixed_base=not args.free_base,
        joint_names=tuple(args.joint_names),
        velocity_joint_names=tuple(args.velocity_joints),
    )
    simulator = create_simulator(args.engine, config)
    try:
        observation, _ = simulator.reset(seed=args.seed)
        for _ in range(args.steps):
            action = np.asarray(args.action, dtype=np.float32)
            if args.num_envs > 1:
                action = np.tile(action, (args.num_envs, 1))
            result = simulator.step(action)
            observation = result.observation
        if observation.qpos is None or not np.isfinite(observation.qpos).all():
            raise ValueError('simulation did not produce finite joint positions')
        return {
            'engine': args.engine,
            'steps': args.steps,
            'qpos_shape': list(observation.qpos.shape),
            'rgb_shape': []
            if observation.rgb_image is None
            else list(observation.rgb_image.shape),
        }
    finally:
        simulator.close()


def register(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = commands.add_parser('run', help='run a native robot scene')
    parser.add_argument('--engine', choices=('genesis', 'maniskill', 'isaacsim'), required=True)
    parser.add_argument('--env-id', default='PickPlace-CRANE-X7')
    parser.add_argument('--compute', choices=('cpu', 'gpu'), default='cpu')
    parser.add_argument('--render-mode', choices=('none', 'rgb_array', 'human'), default='none')
    parser.add_argument('--obs-mode', choices=('state', 'rgb', 'rgbd'), default='state')
    parser.add_argument('--robot', type=Path)
    parser.add_argument('--robot-uid', default='CRANE-X7')
    parser.add_argument('--arm-joints', nargs='*', default=[])
    parser.add_argument('--gripper-joints', nargs='*', default=[])
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--joint-names', nargs='*', default=[])
    parser.add_argument('--velocity-joints', nargs='*', default=[])
    parser.add_argument('--free-base', action='store_true')
    parser.add_argument('--action', nargs='+', type=float)
    parser.add_argument('--steps', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.set_defaults(handler=_run)
