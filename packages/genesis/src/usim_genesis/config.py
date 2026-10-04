# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Genesis configuration boundary for native episode scenes."""

from pathlib import Path

from usim.types import SimulatorConfig
from usim.robots.crane_x7 import CraneX7Config


def validate_config(config: SimulatorConfig, names: list[str]) -> None:
    """Reject unsupported engine options before importing native dependencies."""
    crane = config.robot_path is None and config.robot_uid == 'CRANE-X7'
    mimic = crane and names == CraneX7Config.ALL_JOINT_NAMES
    if config.backend not in {'cpu', 'gpu', 'cuda'}:
        raise ValueError('Genesis backend must be cpu, gpu, or cuda')
    if config.control_mode not in {
        'pd_joint_pos',
        'pd_joint_delta_pos',
        'pd_joint_vel',
        'joint_force',
    }:
        raise ValueError(f'Unsupported Genesis control mode: {config.control_mode}')
    if config.render_mode not in {'none', 'human', 'rgb_array'}:
        raise ValueError(f'Unsupported Genesis render mode: {config.render_mode}')
    if config.obs_mode not in {'state', 'rgb', 'rgbd'}:
        raise ValueError(f'Unsupported Genesis observation mode: {config.obs_mode}')
    if config.env_id not in {'JointControl', 'PickPlace-CRANE-X7'}:
        raise ValueError(f'Unknown Genesis environment: {config.env_id}')
    if config.env_id == 'PickPlace-CRANE-X7' and not mimic:
        raise ValueError('PickPlace-CRANE-X7 requires the bundled CRANE-X7 joint mapping')
    if not crane and config.robot_path is None:
        raise ValueError('External robots require robot_path')
    if not names:
        raise ValueError('External robots require explicit joint names')
    if not set(config.velocity_joint_names) <= set(names):
        raise ValueError('Velocity joints must belong to the configured joint order')
    if config.robot_path is not None:
        if Path(config.robot_path).suffix.lower() not in {'.xml', '.urdf'}:
            raise ValueError('Genesis robot_path must be an MJCF .xml or URDF .urdf')
    if config.initial_qpos is not None:
        if len(config.initial_qpos) != len(names):
            raise ValueError('Initial joint positions must match mapped joints')
    if crane:
        if any(name not in CraneX7Config.ALL_JOINT_NAMES for name in names):
            raise ValueError('Unknown bundled CRANE-X7 joint')
        CraneX7Config.controller_force_limits(config.actuator_profile)
    elif config.actuator_profile != 'legacy':
        raise ValueError('CRANE-X7 actuator profiles do not apply to external robots')
