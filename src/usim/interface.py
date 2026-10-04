# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Abstract interface for simulators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

from usim.types import JointCommand, Observation, SimulatorConfig, StepResult


class EpisodeSimulator(ABC):
    """Abstract base class for all simulators.

    This interface provides a unified API for different simulation backends
    (ManiSkill, Genesis, Isaac Sim, etc.).
    """

    def __init__(self, config: SimulatorConfig):
        """Initialize the simulator with configuration.

        Args:
            config: Simulator configuration.
        """
        self.config = config
        self._is_running = False

    @property
    def arm_joint_names(self) -> list[str]:
        """Optional arm component metadata; not the full articulation contract."""
        return list(self.config.arm_joint_names)

    @property
    def gripper_joint_names(self) -> list[str]:
        """Optional gripper component metadata."""
        return list(self.config.gripper_joint_names)

    @property
    def all_joint_names(self) -> list[str]:
        """Return list of all joint names (arm + gripper)."""
        return self.joint_names

    @property
    def joint_names(self) -> list[str]:
        """Ordered joints corresponding to state arrays, including mobile joints."""
        return list(self.config.joint_names) or self.arm_joint_names + self.gripper_joint_names

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> tuple[Observation, dict[str, Any]]:
        """Reset the environment and return initial observation.

        Args:
            seed: Optional random seed for reproducibility.

        Returns:
            Tuple of (observation, info dict).
        """
        pass

    @abstractmethod
    def step(self, action: NDArray[np.generic] | JointCommand) -> StepResult:
        """Execute one simulation step.

        Args:
            action: Action array to execute.

        Returns:
            StepResult containing observation, reward, and done flags.
        """
        pass

    @abstractmethod
    def get_observation(self) -> Observation:
        """Get current observation without stepping.

        Returns:
            Current observation.
        """
        pass

    @abstractmethod
    def get_qpos(self) -> NDArray[np.generic]:
        """Get current joint positions.

        Returns:
            Array of joint positions.
        """
        pass

    @abstractmethod
    def get_qvel(self) -> NDArray[np.generic]:
        """Get current joint velocities.

        Returns:
            Array of joint velocities.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Release simulator resources."""
        pass

    @property
    def is_running(self) -> bool:
        """Check if simulation is currently running."""
        return self._is_running

    @is_running.setter
    def is_running(self, value: bool) -> None:
        """Set simulation running state."""
        self._is_running = value
