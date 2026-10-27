# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Native joint mapping, initial positions, and controller configuration."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from usim.types import JointCommand, SimulatorConfig


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
    initial = np.zeros(len(names))
    if config.joint_stiffness is not None:
        robot.set_dofs_kp(kp=np.asarray(config.joint_stiffness), dofs_idx_local=dofs)
    if config.joint_damping is not None:
        robot.set_dofs_kv(kv=np.asarray(config.joint_damping), dofs_idx_local=dofs)
    if config.joint_force_limits is not None:
        force = np.asarray(config.joint_force_limits)
        robot.set_dofs_force_range(-force, force, dofs)
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
