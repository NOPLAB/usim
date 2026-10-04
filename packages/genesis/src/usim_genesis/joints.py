# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Native joint mapping, initial positions, and controller configuration."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from usim.types import JointCommand, SimulatorConfig
from usim.robots.crane_x7 import CraneX7Config


def configure_joints(
    robot: Any, config: SimulatorConfig, names: list[str]
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """Configure a dynamically loaded Genesis entity after scene build."""
    indices = []
    for name in names:
        joint = robot.get_joint(name)
        if joint is None or len(joint.dofs_idx_local) != 1:
            raise ValueError(f'Mapped joint {name!r} must exist and have exactly one DOF')
        indices.append(joint.dofs_idx_local[0])
    dofs = np.asarray(indices, dtype=int)
    crane = config.robot_path is None and config.robot_uid == 'CRANE-X7'
    grippers = set(config.gripper_joint_names)
    if crane:
        grippers = set(CraneX7Config.GRIPPER_JOINT_NAMES) & set(names)
    gripper_indices = np.array([i for i, name in enumerate(names) if name in grippers], dtype=int)
    kp = np.full(len(names), 100.0)
    kv = np.full(len(names), 10.0)
    initial = np.zeros(len(names))
    if crane:
        kp[:] = CraneX7Config.ARM_STIFFNESS
        kv[:] = CraneX7Config.ARM_DAMPING
        kp[gripper_indices] = CraneX7Config.GRIPPER_STIFFNESS
        kv[gripper_indices] = CraneX7Config.GRIPPER_DAMPING
        arm_limit, gripper_limit = CraneX7Config.controller_force_limits(config.actuator_profile)
        limits = np.broadcast_to(arm_limit, (CraneX7Config.NUM_ARM_JOINTS,))
        arm_limits = dict(zip(CraneX7Config.ARM_JOINT_NAMES, limits))
        force = np.array([arm_limits.get(name, gripper_limit) for name in names])
        robot.set_dofs_force_range(-force, force, dofs)
        rest = dict(zip(CraneX7Config.ALL_JOINT_NAMES, CraneX7Config.REST_QPOS))
        initial = np.array([rest[name] for name in names])
    robot.set_dofs_kp(kp=kp, dofs_idx_local=dofs)
    robot.set_dofs_kv(kv=kv, dofs_idx_local=dofs)
    if config.gripper_force_limit is not None and len(gripper_indices):
        force = np.full(len(gripper_indices), config.gripper_force_limit)
        robot.set_dofs_force_range(-force, force, dofs[gripper_indices])
    if config.initial_qpos is not None:
        initial = np.asarray(config.initial_qpos, dtype=float)
    return dofs, initial


def named_targets(
    command: JointCommand, names: list[str], n_envs: int
) -> tuple[
    tuple[NDArray[np.intp], NDArray[np.float64]],
    tuple[NDArray[np.intp], NDArray[np.float64]],
]:
    """Parse all named targets before applying any controller writes."""
    groups = []
    for targets in (command.positions, command.velocities):
        indices = []
        values = []
        for name, value in targets.items():
            if name not in names:
                raise ValueError(f'Unmapped Genesis command joint: {name}')
            column = np.asarray(value, dtype=float)
            if column.ndim > 1 or (column.ndim == 1 and column.shape != (n_envs,)):
                raise ValueError(f'Joint {name!r} needs a scalar or ({n_envs},) batch')
            if not np.all(np.isfinite(column)):
                raise ValueError(f'Joint {name!r} command must be finite')
            indices.append(names.index(name))
            values.append(np.broadcast_to(column, (n_envs,)))
        groups.append(
            (
                np.asarray(indices, dtype=int),
                np.column_stack(values) if values else np.empty((n_envs, 0)),
            )
        )
    return groups[0], groups[1]
