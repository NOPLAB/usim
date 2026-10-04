# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Common types for the usim interface layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@dataclass
class SimulatorConfig:
    """Configuration for simulator initialization."""

    env_id: str
    backend: str = 'cpu'
    render_mode: str = 'rgb_array'
    control_mode: str = 'pd_joint_pos'
    sim_rate: float = 30.0
    max_episode_steps: int = 200
    robot_init_qpos_noise: float = 0.02
    n_envs: int = 1  # Number of parallel environments for batch parallelization
    camera_uid: str = 'hand_camera'
    actuator_profile: str = 'legacy'
    gripper_force_limit: float | None = None
    robot_path: str | None = None
    robot_uid: str = 'CRANE-X7'
    arm_joint_names: tuple[str, ...] = ()
    gripper_joint_names: tuple[str, ...] = ()
    initial_qpos: tuple[float, ...] | None = None
    obs_mode: str = 'rgb'
    fixed_base: bool = True
    joint_names: tuple[str, ...] = ()
    velocity_joint_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.env_id.strip() or not self.robot_uid.strip():
            raise ValueError('Environment and robot identifiers must not be empty')
        if not isfinite(self.sim_rate) or self.sim_rate <= 0:
            raise ValueError('Simulation rate must be finite and positive')
        if self.n_envs < 1 or self.max_episode_steps < 1:
            raise ValueError('Environment count and episode length must be positive')
        if not isfinite(self.robot_init_qpos_noise) or self.robot_init_qpos_noise < 0:
            raise ValueError('Joint initialization noise must be finite and nonnegative')
        if self.gripper_force_limit is not None:
            if not isfinite(self.gripper_force_limit) or self.gripper_force_limit <= 0:
                raise ValueError('Gripper force limit must be finite and positive')
        components = self.arm_joint_names + self.gripper_joint_names
        names = self.joint_names or components
        if len(names) != len(set(names)) or any(not name.strip() for name in names):
            raise ValueError('Joint names must be nonempty and unique')
        if len(components) != len(set(components)):
            raise ValueError('Joint component groups must not overlap')
        if self.joint_names and not set(components + self.velocity_joint_names) <= set(names):
            raise ValueError('Component joints must belong to the configured joint order')
        if len(self.velocity_joint_names) != len(set(self.velocity_joint_names)):
            raise ValueError('Velocity joint names must be unique')
        if self.robot_path is not None and not Path(self.robot_path).is_file():
            raise ValueError(f'Missing robot asset: {self.robot_path}')
        if self.initial_qpos is not None:
            if not all(isfinite(value) for value in self.initial_qpos):
                raise ValueError('Initial joint positions must be finite')
            if names and len(self.initial_qpos) != len(names):
                raise ValueError('Initial joint positions must match configured joints')


@dataclass(frozen=True, slots=True)
class JointCommand:
    """Named position and velocity targets for one shared robot articulation."""

    positions: Mapping[str, float | NDArray[np.generic]] = field(default_factory=dict)
    velocities: Mapping[str, float | NDArray[np.generic]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if set(self.positions) & set(self.velocities):
            raise ValueError('A joint cannot receive position and velocity targets simultaneously')


@dataclass
class Observation:
    """Unified observation structure across simulators."""

    rgb_image: Optional[NDArray[np.generic]] = None
    depth_image: Optional[NDArray[np.generic]] = None
    qpos: Optional[NDArray[np.generic]] = None
    qvel: Optional[NDArray[np.generic]] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """Result of a simulation step."""

    observation: Observation
    reward: float | NDArray[np.generic]
    terminated: bool | NDArray[np.generic]
    truncated: bool | NDArray[np.generic]
    info: dict[str, Any] | list[dict[str, Any]]
