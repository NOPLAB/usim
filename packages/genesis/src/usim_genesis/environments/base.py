# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Base class for Genesis task environments."""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class GenesisEnvironment(ABC):
    """Abstract base class for Genesis task environments.

    This class defines the interface that all Genesis task environments
    must implement for use with the GenesisSimulator adapter.
    """

    def __init__(self, scene: Any, robot: Any, robot_init_qpos_noise: float = 0.02):
        """Initialize the environment.

        Args:
            scene: Genesis scene instance.
            robot: Genesis robot entity.
            robot_init_qpos_noise: Standard deviation for initial qpos noise.
        """
        self.scene = scene
        self.robot = robot
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self._rng = np.random.default_rng()

    @abstractmethod
    def setup_scene(self) -> None:
        """Add task-specific objects to the scene.

        Called once after the robot is loaded but before scene.build().
        """
        pass

    @abstractmethod
    def reset(
        self, seed: Optional[int] = None, env_ids: Optional[np.ndarray] = None
    ) -> dict[str, Any]:
        """Reset the task state for a new episode.

        Args:
            seed: Optional random seed for reproducibility.

        Returns:
            Info dictionary with task-specific reset information.
        """
        pass

    @abstractmethod
    def compute_reward(self) -> np.ndarray:
        """Compute the current reward.

        Returns:
            Reward array, one value per environment.
        """
        pass

    @abstractmethod
    def is_success(self) -> np.ndarray:
        """Check if the task has been successfully completed.

        Returns:
            Boolean array indicating task completion per environment.
        """
        pass

    @abstractmethod
    def is_terminated(self) -> np.ndarray:
        """Check if the episode should terminate.

        Returns:
            Boolean array indicating episode termination per environment.
        """
        pass

    @abstractmethod
    def get_info(self) -> list[dict[str, Any]]:
        """Get additional task-specific information.

        Returns:
            Task metrics and state information, one dictionary per environment.
        """
        pass
