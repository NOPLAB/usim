"""Native scene commands; engine imports occur only on execution."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from usim import Observation, SimulatorConfig


def _run_model(args: argparse.Namespace, config: SimulatorConfig) -> Observation:
    """Load caller-owned models through the composed model and control API."""
    from usim import (
        AgentSpec,
        ControlSpec,
        ControlTargets,
        ExecutionSpec,
        ImportRequirements,
        ModelError,
        ModelSource,
        SceneSpec,
        SensorSpec,
        create_backend,
    )

    if config.robot_uid:
        raise ModelError('--robot and --robot-uid select different model sources')
    names = config.joint_names or config.arm_joint_names + config.gripper_joint_names
    if args.steps and len(args.action) != len(names):
        raise ModelError('--action must match the explicit controlled joint order')
    velocities = config.velocity_joint_names
    control = ControlSpec(
        joint_names=names,
        position_joints=tuple(name for name in names if name not in velocities),
        velocity_joints=velocities,
        initial_qpos=config.initial_qpos,
        joint_stiffness=config.joint_stiffness,
        joint_damping=config.joint_damping,
        joint_force_limits=config.joint_force_limits,
    )
    sensors = (
        (SensorSpec('camera', config.obs_mode, rate=config.sim_rate),)
        if config.obs_mode != 'state' and config.render_mode != 'none'
        else ()
    )
    scene = SceneSpec(
        agents=(
            AgentSpec(
                'robot',
                ModelSource.from_path(args.robot),
                control,
                sensors,
                requirements=ImportRequirements(base='fixed' if config.fixed_base else 'floating'),
            ),
        )
    )
    execution = ExecutionSpec(
        backend=config.backend,
        n_envs=config.n_envs,
        max_episode_steps=config.max_episode_steps,
        render_mode=config.render_mode,
        obs_mode=config.obs_mode if sensors else 'state',
        native_task=config.env_id,
    )
    targets = None
    if args.steps:
        values = dict(zip(names, args.action))
        targets = ControlTargets(
            positions={name: value for name, value in values.items() if name not in velocities},
            velocities={name: value for name, value in values.items() if name in velocities},
        )
    with create_backend(args.engine).open(scene, execution) as session:
        observation, _ = session.execution.reset(seed=args.seed)
        if targets is not None:
            for _ in range(args.steps):
                session.control.for_entity('robot').apply(targets)
                session.execution.step()
                observation = session.state.for_entity('robot').get_observation()
        return observation


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
        initial_qpos=None if args.initial_qpos is None else tuple(args.initial_qpos),
        joint_stiffness=None if args.joint_stiffness is None else tuple(args.joint_stiffness),
        joint_damping=None if args.joint_damping is None else tuple(args.joint_damping),
        joint_force_limits=None
        if args.joint_force_limits is None
        else tuple(args.joint_force_limits),
    )
    if args.robot is not None:
        observation = _run_model(args, config)
    else:
        simulator = create_simulator(args.engine, config)
        try:
            observation, _ = simulator.reset(seed=args.seed)
            for _ in range(args.steps):
                action = np.asarray(args.action, dtype=np.float32)
                if args.num_envs > 1:
                    action = np.tile(action, (args.num_envs, 1))
                result = simulator.step(action)
                observation = result.observation
        finally:
            simulator.close()
    if observation.qpos is None or not np.isfinite(observation.qpos).all():
        raise ValueError('simulation did not produce finite joint positions')
    return {
        'engine': args.engine,
        'steps': args.steps,
        'qpos_shape': list(observation.qpos.shape),
        'rgb_shape': [] if observation.rgb_image is None else list(observation.rgb_image.shape),
    }


def register(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    from usim import list_simulators

    parser = commands.add_parser('run', help='run a native robot scene')
    parser.add_argument(
        '--engine',
        choices=list_simulators(),
        required=True,
        help='installed native episode simulator (use "simulate" for continuous execution)',
    )
    parser.add_argument('--env-id', default='JointControl')
    parser.add_argument('--compute', choices=('cpu', 'gpu'), default='cpu')
    parser.add_argument('--render-mode', choices=('none', 'rgb_array', 'human'), default='none')
    parser.add_argument('--obs-mode', choices=('state', 'rgb', 'rgbd'), default='state')
    parser.add_argument('--robot', type=Path)
    parser.add_argument('--robot-uid', default='')
    parser.add_argument('--arm-joints', nargs='*', default=[])
    parser.add_argument('--gripper-joints', nargs='*', default=[])
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--joint-names', nargs='*', default=[])
    parser.add_argument('--velocity-joints', nargs='*', default=[])
    parser.add_argument('--initial-qpos', nargs='+', type=float)
    parser.add_argument('--joint-stiffness', nargs='+', type=float)
    parser.add_argument('--joint-damping', nargs='+', type=float)
    parser.add_argument('--joint-force-limits', nargs='+', type=float)
    parser.add_argument('--free-base', action='store_true')
    parser.add_argument('--action', nargs='+', type=float)
    parser.add_argument('--steps', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.set_defaults(handler=_run)
