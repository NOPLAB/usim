# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Genesis episode adapter with explicit joint mapping and batched observations."""

from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from usim.factory import register_simulator
from usim.interface import EpisodeSimulator
from usim.types import JointCommand, Observation, SimulatorConfig, StepResult
from usim.robots.crane_x7 import CraneX7Config, get_mjcf_path
from .camera import frame_robot, render_images
from .config import validate_config
from .joints import configure_joints, named_targets


@register_simulator('genesis')
class GenesisSimulator(EpisodeSimulator):
    """Joint-space control for MJCF/URDF robots and the CRANE-X7 pick/place task."""

    def __init__(self, config: SimulatorConfig):
        super().__init__(config)
        self._n_envs = config.n_envs
        # Native Genesis objects are dynamically imported at the optional engine boundary.
        self._gs: Any = None
        self._scene: Any = None
        self._robot: Any = None
        self._camera: Any = None
        self._environment: Any = None
        self._current_obs: Observation | None = None
        self._dofs_idx = np.array([], dtype=int)
        self._episode_steps = np.zeros(self._n_envs, dtype=np.int32)
        self._rng = np.random.default_rng()
        self._crane = config.robot_path is None and config.robot_uid == 'CRANE-X7'
        self._arm_names = list(config.arm_joint_names)
        self._gripper_names = list(config.gripper_joint_names)
        if (
            self._crane
            and not config.joint_names
            and not self._arm_names
            and not self._gripper_names
        ):
            self._arm_names = list(CraneX7Config.ARM_JOINT_NAMES)
            self._gripper_names = list(CraneX7Config.GRIPPER_JOINT_NAMES)
        self._mimic = self._crane and self.all_joint_names == CraneX7Config.ALL_JOINT_NAMES
        validate_config(config, self.all_joint_names)
        try:
            self._init_genesis()
            self._init_scene()
            self._add_robot()
            self._setup_camera()
            self._init_environment()
            self._scene.build(n_envs=self._n_envs)
            self._dofs_idx, self._initial_qpos = configure_joints(
                self._robot, config, self.all_joint_names
            )
            self.reset()
            self._is_running = True
        finally:
            if not self._is_running:
                self.close()

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def arm_joint_names(self) -> list[str]:
        return list(self._arm_names)

    @property
    def gripper_joint_names(self) -> list[str]:
        return list(self._gripper_names)

    def _require_scene(self) -> None:
        if self._scene is None:
            raise RuntimeError('Genesis simulator is closed')

    def reset(
        self, seed: int | None = None, env_ids: np.ndarray | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        """Reset selected environments without advancing any environment's physics."""
        self._require_scene()
        ids: np.ndarray = np.arange(self._n_envs) if env_ids is None else np.asarray(env_ids)
        ids = np.atleast_1d(ids)
        if (
            ids.ndim != 1
            or not np.issubdtype(ids.dtype, np.integer)
            or len(ids) == 0
            or len(np.unique(ids)) != len(ids)
            or np.any((ids < 0) | (ids >= self._n_envs))
        ):
            raise ValueError('env_ids must contain unique valid environment indices')
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        rest = np.tile(self._initial_qpos, (len(ids), 1))
        rest += self._rng.normal(0, self.config.robot_init_qpos_noise, rest.shape)
        self._scene.reset(envs_idx=ids)
        self._robot.set_dofs_position(rest, self._dofs_idx, envs_idx=ids)
        self._robot.set_dofs_velocity(np.zeros_like(rest), self._dofs_idx, envs_idx=ids)
        # Clear the previous episode's command, including velocity/force controllers.
        if self.config.control_mode in {'pd_joint_pos', 'pd_joint_delta_pos'}:
            self._robot.control_dofs_position(rest, self._dofs_idx, envs_idx=ids)
        elif self.config.control_mode == 'pd_joint_vel':
            self._robot.control_dofs_velocity(np.zeros_like(rest), self._dofs_idx, envs_idx=ids)
        else:
            self._robot.control_dofs_force(np.zeros_like(rest), self._dofs_idx, envs_idx=ids)
        if self.config.velocity_joint_names:
            indices = [
                self.all_joint_names.index(name) for name in self.config.velocity_joint_names
            ]
            self._robot.control_dofs_velocity(
                np.zeros((len(ids), len(indices))), self._dofs_idx[indices], envs_idx=ids
            )
        self._episode_steps[ids] = 0
        info = {}
        if self._environment is not None:
            info.update(self._environment.reset(seed=seed, env_ids=ids))
        if self._camera is not None and not self._crane:
            positions = self._numpy(
                self._robot.get_vverts() if self._robot.n_vverts else self._robot.get_links_pos()
            )
            bounds = np.stack([positions.min(axis=1), positions.max(axis=1)], axis=1)
            frame_robot(self._camera, bounds)
        self._current_obs = self._convert_observation()
        return self._current_obs, info

    def step(self, action: np.ndarray | JointCommand) -> StepResult:
        """Apply mapped targets, then advance exactly one step at config.sim_rate."""
        self._require_scene()
        if isinstance(action, JointCommand):
            position, velocity = named_targets(action, self.all_joint_names, self._n_envs)
            if len(position[0]):
                self._robot.control_dofs_position(position[1], self._dofs_idx[position[0]])
            if len(velocity[0]):
                self._robot.control_dofs_velocity(velocity[1], self._dofs_idx[velocity[0]])
        else:
            values = np.asarray(action, dtype=float)
            if values.ndim == 1 and self._n_envs == 1:
                values = values[None, :]
            if values.ndim != 2 or values.shape[0] != self._n_envs:
                raise ValueError(f'Action must have shape ({self._n_envs}, mapped joints)')
            if not np.all(np.isfinite(values)):
                raise ValueError('Action must contain finite values')
            self._apply_action(values)
        self._scene.step()
        self._episode_steps += 1
        self._current_obs = self._convert_observation()
        rewards = np.zeros(self._n_envs, dtype=np.float32)
        terminated = np.zeros(self._n_envs, dtype=bool)
        infos = [{} for _ in range(self._n_envs)]
        if self._environment is not None:
            rewards = self._environment.compute_reward()
            terminated = self._environment.is_terminated()
            infos = self._environment.get_info()
        return StepResult(
            observation=self._current_obs,
            reward=rewards,
            terminated=terminated,
            truncated=self._episode_steps >= self.config.max_episode_steps,
            info=infos,
        )

    def get_observation(self) -> Observation:
        self._require_scene()
        self._current_obs = self._convert_observation()
        return self._current_obs

    @staticmethod
    def _numpy(value: Any) -> np.ndarray:
        if hasattr(value, 'detach'):
            value = value.detach()
        if hasattr(value, 'cpu'):
            value = value.cpu().numpy()
        return np.asarray(value)

    def get_qpos(self) -> np.ndarray:
        self._require_scene()
        return self._numpy(self._robot.get_dofs_position(self._dofs_idx)).reshape(self._n_envs, -1)

    def get_qvel(self) -> np.ndarray:
        self._require_scene()
        return self._numpy(self._robot.get_dofs_velocity(self._dofs_idx)).reshape(self._n_envs, -1)

    def close(self) -> None:
        """Destroy this scene, leaving other users of the Genesis runtime alive."""
        scene, self._scene = self._scene, None
        try:
            if scene is not None:
                scene.destroy()
        finally:
            self._robot = None
            self._camera = None
            self._environment = None
            self._current_obs = None
            self._is_running = False

    def _init_genesis(self) -> None:
        gs = import_module('genesis')
        self._gs = gs
        backend = gs.cpu if self.config.backend == 'cpu' else gs.cuda
        if not gs._initialized:
            gs.init(backend=backend)
        elif gs.backend != backend:
            raise ValueError('Genesis is already initialized with a different backend')

    def _init_scene(self) -> None:
        gs = self._gs
        self._scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=1.0 / self.config.sim_rate),
            vis_options=gs.options.VisOptions(
                split_envs=True, rendered_envs_idx=list(range(self._n_envs))
            ),
            show_viewer=self.config.render_mode == 'human',
        )

    def _add_robot(self) -> None:
        path = self.config.robot_path or get_mjcf_path()
        if Path(path).suffix.lower() == '.urdf':
            morph = self._gs.morphs.URDF(
                file=str(Path(path).resolve()), fixed=self.config.fixed_base
            )
        else:
            morph = self._gs.morphs.MJCF(file=str(Path(path).resolve()))
        self._robot = self._scene.add_entity(morph)

    def _setup_camera(self) -> None:
        if self.config.render_mode == 'none' or self.config.obs_mode == 'state':
            return
        self._camera = self._scene.add_camera(
            res=(640, 480), pos=(0.3, 0.0, 0.4), lookat=(0.15, 0.0, 0.1), fov=69, GUI=False
        )

    def _init_environment(self) -> None:
        if self.config.env_id == 'JointControl':
            self._scene.add_entity(self._gs.morphs.Plane())
            return
        from .environments.pick_place import PickPlace

        self._environment = PickPlace(
            scene=self._scene,
            robot=self._robot,
            robot_init_qpos_noise=self.config.robot_init_qpos_noise,
        )
        self._environment.setup_scene()

    def _apply_action(self, action: np.ndarray) -> None:
        if self._mimic and action.shape[1] == 8:
            action = np.column_stack([action, action[:, 7]])
        if action.shape[1] != len(self._dofs_idx):
            raise ValueError(f'Action must have {len(self._dofs_idx)} mapped joint values')
        mode = self.config.control_mode
        if mode == 'pd_joint_delta_pos':
            action = self.get_qpos() + action
        if mode in {'pd_joint_pos', 'pd_joint_delta_pos'}:
            self._robot.control_dofs_position(action, self._dofs_idx)
        elif mode == 'pd_joint_vel':
            self._robot.control_dofs_velocity(action, self._dofs_idx)
        else:
            self._robot.control_dofs_force(action, self._dofs_idx)

    def _convert_observation(self) -> Observation:
        rgb, depth = None, None
        if self._camera is not None:
            rgb, depth = render_images(self._camera, self._n_envs, self.config.obs_mode == 'rgbd')
        extra = {}
        if self._environment is not None:
            extra['env_info'] = self._environment.get_info()
        return Observation(
            rgb_image=rgb,
            depth_image=depth,
            qpos=self.get_qpos(),
            qvel=self.get_qvel(),
            extra=extra,
        )
