# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""ManiSkill simulator adapter implementing usim interface."""

from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
import torch

from usim.factory import register_simulator
from usim.interface import EpisodeSimulator
from usim.types import JointCommand, Observation, SimulatorConfig, StepResult


@register_simulator('maniskill')
class ManiSkillSimulator(EpisodeSimulator):
    """ManiSkill simulator implementation."""

    def __init__(self, config: SimulatorConfig):
        super().__init__(config)
        self._env: gym.Env | None = None
        self._current_obs = None
        self._arm_joint_names: list[str] = []
        self._gripper_joint_names: list[str] = []
        self._joint_names: list[str] = []
        try:
            self._init_environment()
        except Exception:
            self.close()
            raise

    def _init_environment(self) -> None:
        """Initialize ManiSkill environment."""
        if self.config.robot_path is not None:
            raise ValueError(
                'ManiSkill robot_path is unsupported; register a ManiSkill agent and use robot_uid'
            )
        if not float(self.config.sim_rate).is_integer():
            raise ValueError('ManiSkill sim_rate must be an integer control frequency in Hz')
        if self.config.n_envs > 1 and self.config.backend in ('cpu', 'physx_cpu'):
            raise ValueError("ManiSkill n_envs > 1 requires backend='gpu' or 'auto'")

        import mani_skill.envs  # noqa: F401 - registers standard environments and agents

        kwargs: dict[str, str | float] = {
            'render_backend': 'cpu' if self.config.backend == 'cpu' else 'gpu'
        }
        render_mode = self.config.render_mode
        if render_mode == 'none':
            render_mode = None
            if self.config.obs_mode in ('none', 'state', 'state_dict'):
                kwargs['render_backend'] = 'none'
        if self.config.robot_uid == 'CRANE-X7':
            from usim_maniskill.agent import CraneX7
            from usim.robots.crane_x7 import CraneX7Config

            CraneX7.arm_force_limit, CraneX7.gripper_force_limit = (
                CraneX7Config.controller_force_limits(self.config.actuator_profile)
            )
            if self.config.gripper_force_limit is not None:
                CraneX7.gripper_force_limit = self.config.gripper_force_limit
        elif (
            self.config.actuator_profile != 'legacy' or self.config.gripper_force_limit is not None
        ):
            raise ValueError('Force overrides are supported only by the CRANE-X7 registered agent')

        if self.config.env_id.startswith('PickPlace-CRANE-X7'):
            from usim_maniskill.environments import PickPlace

            if self.config.robot_uid != 'CRANE-X7':
                raise ValueError("PickPlace-CRANE-X7 requires robot_uid='CRANE-X7'")
            if self.config.camera_uid not in ('hand_camera', 'scene_camera'):
                raise ValueError(f'Unsupported CRANE pick-place camera: {self.config.camera_uid}')
            if self.config.env_id == 'PickPlace-CRANE-X7':
                PickPlace.use_scene_camera = self.config.camera_uid == 'scene_camera'
            kwargs['robot_init_qpos_noise'] = self.config.robot_init_qpos_noise

        self._env = gym.make(
            self.config.env_id,
            render_mode=render_mode,
            sim_backend=self.config.backend,
            robot_uids=self.config.robot_uid,
            obs_mode=self.config.obs_mode,
            control_mode=self.config.control_mode,
            num_envs=self.config.n_envs,
            max_episode_steps=self.config.max_episode_steps,
            sim_config={
                'control_freq': int(self.config.sim_rate),
                'sim_freq': int(self.config.sim_rate) * 5,
            },
            **kwargs,
        )
        self._discover_joints()
        self.reset()
        self._is_running = True

    def _require_env(self) -> gym.Env:
        if self._env is None:
            raise RuntimeError('ManiSkill simulator is closed')
        return self._env

    def _discover_joints(self) -> None:
        agent = self._require_env().unwrapped.agent
        fixed_root = self._numpy(agent.robot.fixed_root_link)
        if not np.all(fixed_root == self.config.fixed_base):
            raise ValueError(
                'ManiSkill fixed_base must match the registered agent root; '
                'choose a registered agent with the required base mobility'
            )
        active_names = [joint.name for joint in agent.robot.get_active_joints()]
        controllers = getattr(agent.controller, 'controllers', {})
        gripper = controllers.get('gripper')
        discovered_gripper = getattr(agent, 'gripper_joint_names', None)
        if discovered_gripper is None:
            discovered_gripper = [] if gripper is None else [j.name for j in gripper.joints]
        self._gripper_joint_names = list(self.config.gripper_joint_names or discovered_gripper)
        arm = controllers.get('arm')
        discovered_arm = getattr(agent, 'arm_joint_names', None)
        if discovered_arm is None:
            discovered_arm = [] if arm is None else [j.name for j in arm.joints]
        self._arm_joint_names = list(self.config.arm_joint_names or discovered_arm)
        groups = self._arm_joint_names + self._gripper_joint_names
        unknown = set(groups + list(self.config.velocity_joint_names)) - set(active_names)
        if unknown:
            raise ValueError(
                f'Unknown ManiSkill joints: {sorted(unknown)}; available: {active_names}'
            )
        if len(set(groups)) != len(groups):
            raise ValueError('Arm and gripper joint selections must not overlap')
        if self.config.joint_names and list(self.config.joint_names) != active_names:
            raise ValueError(
                f'ManiSkill joint_names must match native articulation order: {active_names}'
            )
        self._joint_names = active_names
        if self.config.initial_qpos is not None:
            if len(self.config.initial_qpos) != len(active_names):
                raise ValueError('Initial joint positions must match reported joints')

    @staticmethod
    def _numpy(value) -> NDArray[np.generic]:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value).copy()

    def _unbatch(self, value: NDArray[np.generic]) -> NDArray[np.generic]:
        return value[0] if self.config.n_envs == 1 and value.ndim > 0 else value

    @property
    def joint_names(self) -> list[str]:
        return self._joint_names.copy()

    @property
    def arm_joint_names(self) -> list[str]:
        return self._arm_joint_names.copy()

    @property
    def gripper_joint_names(self) -> list[str]:
        return self._gripper_joint_names.copy()

    def reset(self, seed: int | None = None) -> tuple[Observation, dict[str, Any]]:
        env = self._require_env()
        obs, info = env.reset(seed=seed)
        if self.config.initial_qpos is not None:
            agent = env.unwrapped.agent
            qpos = agent.robot.get_qpos().clone()
            qpos[:] = torch.as_tensor(
                self.config.initial_qpos, device=qpos.device, dtype=qpos.dtype
            )
            agent.reset(qpos)
            if env.unwrapped.gpu_sim_enabled:
                env.unwrapped.scene._gpu_apply_all()
                env.unwrapped.scene.px.gpu_update_articulation_kinematics()
                env.unwrapped.scene._gpu_fetch_all()
            agent.controller.reset()
            info.update(env.unwrapped.get_info())
            obs = env.unwrapped.get_obs(info)
        self._current_obs = obs
        return self._convert_observation(obs), info

    def step(self, action: NDArray[np.generic] | JointCommand) -> StepResult:
        if isinstance(action, JointCommand):
            raise ValueError(
                'ManiSkill does not support named JointCommand; use the registered controller '
                'native action vector with its documented scaling and ordering'
            )
        obs, reward, terminated, truncated, info = self._require_env().step(action)
        self._current_obs = obs
        reward = self._unbatch(self._numpy(reward))
        terminated = self._unbatch(self._numpy(terminated))
        truncated = self._unbatch(self._numpy(truncated))
        return StepResult(
            observation=self._convert_observation(obs),
            reward=float(reward) if self.config.n_envs == 1 else reward,
            terminated=bool(terminated) if self.config.n_envs == 1 else terminated,
            truncated=bool(truncated) if self.config.n_envs == 1 else truncated,
            info=info,
        )

    def get_observation(self) -> Observation:
        self._require_env()
        return self._convert_observation(self._current_obs)

    def get_qpos(self) -> NDArray[np.generic]:
        qpos = self._require_env().unwrapped.agent.robot.get_qpos()
        return self._unbatch(self._numpy(qpos))

    def get_qvel(self) -> NDArray[np.generic]:
        qvel = self._require_env().unwrapped.agent.robot.get_qvel()
        return self._unbatch(self._numpy(qvel))

    def close(self) -> None:
        env, self._env = self._env, None
        self._current_obs = None
        self._is_running = False
        if env is not None:
            env.close()

    def _convert_observation(self, obs: Any) -> Observation:
        """Convert ManiSkill observation to usim Observation."""
        rgb_image = None
        depth_image = None

        if isinstance(obs, dict) and 'sensor_data' in obs:
            sensors = obs['sensor_data']
            if self.config.camera_uid not in sensors:
                raise ValueError(
                    f'Unknown ManiSkill camera {self.config.camera_uid!r}; '
                    f'available: {sorted(sensors)}'
                )
            camera = sensors[self.config.camera_uid]
            rgb = camera.get('rgb')
            if rgb is not None:
                rgb = self._numpy(rgb)
                if np.issubdtype(rgb.dtype, np.floating):
                    rgb = np.rint(np.multiply(np.clip(rgb, 0, 1), 255)).astype(np.uint8)
                else:
                    rgb = rgb.astype(np.uint8)
                rgb_image = self._unbatch(rgb)

            depth = camera.get('depth')
            if depth is not None:
                depth_image = self._unbatch(self._numpy(depth))

        qpos = self.get_qpos()
        qvel = self.get_qvel()

        return Observation(
            rgb_image=rgb_image,
            depth_image=depth_image,
            qpos=qpos,
            qvel=qvel,
            extra={'raw_obs': obs},
        )
