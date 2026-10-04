# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

"""Single-environment native Isaac Sim 6.1 robot adapter."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from numpy.typing import NDArray

from usim.factory import register_simulator
from usim.interface import EpisodeSimulator
from usim.types import JointCommand, Observation, SimulatorConfig, StepResult
from usim.robots.crane_x7 import CraneX7Config, get_mjcf_path


@register_simulator('isaacsim')
class IsaacSimSimulator(EpisodeSimulator):
    """Own one articulation with named position and velocity control.

    Kit is process-global. Use one adapter per dedicated Python process, and
    import native Isaac modules only after constructing SimulationApp.
    """

    def __init__(self, config: SimulatorConfig):
        super().__init__(config)
        if config.env_id not in ('JointControl', 'PickPlace-CRANE-X7'):
            raise ValueError(f'Unsupported Isaac environment: {config.env_id}')
        if config.n_envs != 1:
            raise ValueError('Isaac native episodes currently require n_envs=1')
        if config.control_mode != 'pd_joint_pos':
            raise ValueError('Isaac native episodes support only pd_joint_pos')
        if config.obs_mode not in ('state', 'rgb', 'rgbd'):
            raise ValueError('Isaac obs_mode must be state, rgb, or rgbd')
        if config.render_mode not in ('none', 'human', 'rgb_array'):
            raise ValueError('Isaac render_mode must be none, human, or rgb_array')
        if config.robot_path is None and config.robot_uid != 'CRANE-X7':
            raise ValueError('External robots require robot_path')
        if config.env_id == 'PickPlace-CRANE-X7' and (
            config.robot_path is not None
            or config.robot_uid != 'CRANE-X7'
            or not config.fixed_base
            or config.velocity_joint_names
        ):
            raise ValueError('PickPlace-CRANE-X7 requires the fixed-base bundled CRANE-X7')
        self._path = Path(config.robot_path or get_mjcf_path()).resolve()
        if not self._path.is_file():
            raise ValueError(f'Missing robot asset: {self._path}')
        if self._path.suffix.lower() not in ('.xml', '.urdf', '.usd', '.usda', '.usdc'):
            raise ValueError('Robot asset must be MJCF XML, URDF, or USD')
        self._bundled = config.robot_path is None
        self._joint_names = list(config.joint_names)
        self._arm_names = list(config.arm_joint_names)
        self._gripper_names = list(config.gripper_joint_names)
        if (
            self._bundled
            and not self._joint_names
            and not self._arm_names
            and not self._gripper_names
        ):
            self._arm_names = list(CraneX7Config.ARM_JOINT_NAMES)
            self._gripper_names = list(CraneX7Config.GRIPPER_JOINT_NAMES)
        # Native objects are loaded dynamically after Kit starts.
        self._app: Any = None
        self._world: Any = None
        self._robot: Any = None
        self._camera: Any = None
        self._assets: TemporaryDirectory[str] | None = None
        self._cube = None
        self._fingers = []
        self._steps = 0
        self._rng = np.random.default_rng()
        initialized = False
        try:
            self._initialize()
            self._is_running = True
            self.reset()
            initialized = True
        finally:
            if not initialized:
                self.close()

    @property
    def arm_joint_names(self) -> list[str]:
        return list(self._arm_names)

    @property
    def gripper_joint_names(self) -> list[str]:
        return list(self._gripper_names)

    @property
    def joint_names(self) -> list[str]:
        return list(self._joint_names or (self._arm_names + self._gripper_names))

    @property
    def all_joint_names(self) -> list[str]:
        return self.joint_names

    def _initialize(self) -> None:
        from isaacsim import SimulationApp

        self._app = SimulationApp(
            {
                'headless': self.config.render_mode != 'human',
                'multi_gpu': False,
                'enable_crashreporter': False,
                'width': 640,
                'height': 480,
                'samples_per_pixel_per_frame': 1,
            }
        )
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.utils.types import ArticulationAction
        from pxr import Usd, UsdGeom, UsdPhysics

        self._action_type = ArticulationAction
        self._assets = TemporaryDirectory(prefix='usim-native-')
        robot_usd = self._import_robot()
        asset_stage = Usd.Stage.Open(str(robot_usd))
        if UsdGeom.GetStageUpAxis(asset_stage) != UsdGeom.Tokens.z or not np.isclose(
            UsdGeom.GetStageMetersPerUnit(asset_stage), 1.0
        ):
            raise ValueError('Robot USD must be metre-scale and Z-up')
        self._world = World(
            stage_units_in_meters=1.0,
            physics_dt=1 / self.config.sim_rate,
            rendering_dt=1 / self.config.sim_rate,
        )
        add_reference_to_stage(str(robot_usd), '/World/Robot')
        self._stage = omni.usd.get_context().get_stage()
        self._stage.GetPrimAtPath('/World/Robot').GetVariantSet('Physics').SetVariantSelection(
            'physx'
        )
        self._stage.Load()
        if UsdGeom.GetStageUpAxis(self._stage) != UsdGeom.Tokens.z:
            raise ValueError('Robot stage must be Z-up')
        roots = [
            prim
            for prim in self._stage.Traverse()
            if (
                str(prim.GetPath()) == '/World/Robot'
                or str(prim.GetPath()).startswith('/World/Robot/')
            )
            and prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        ]
        if len(roots) != 1:
            raise ValueError(f'Expected one robot articulation, found {len(roots)}')
        self._robot = self._world.scene.add(
            SingleArticulation(prim_path=str(roots[0].GetPath()), name='robot')
        )
        self._world.scene.add_default_ground_plane()
        if self.config.env_id == 'PickPlace-CRANE-X7':
            self._setup_task()
        self._world.reset()
        self._configure_joints()
        if self.config.obs_mode != 'state' and self.config.render_mode != 'none':
            self._setup_camera()

    def _import_robot(self) -> Path:
        from isaacsim.core.utils.extensions import enable_extension

        assert self._assets is not None
        if self._path.suffix.lower() == '.xml':
            enable_extension('isaacsim.asset.importer.mjcf')
            from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig

            config = MJCFImporterConfig(
                mjcf_path=str(self._path),
                usd_path=self._assets.name,
                fix_base=self.config.fixed_base,
                import_scene=False,
                joint_target_type='position',
                joint_drive_type='force',
            )
            result = MJCFImporter(config).import_mjcf()
        elif self._path.suffix.lower() == '.urdf':
            enable_extension('isaacsim.asset.importer.urdf')
            from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig

            config = URDFImporterConfig(
                urdf_path=str(self._path),
                usd_path=self._assets.name,
                merge_fixed_joints=False,
                fix_base=self.config.fixed_base,
                collision_from_visuals=False,
                joint_target_type='position',
                joint_drive_type='force',
            )
            result = URDFImporter(config).import_urdf()
        else:
            result = self._path
        result = Path(result)
        if not result.is_file():
            raise RuntimeError(f'Robot import produced no USD: {result}')
        return result

    def _configure_joints(self) -> None:
        native_names = list(self._robot.dof_names)
        if not self.all_joint_names:
            self._arm_names = native_names
        self._joint_names = self.all_joint_names
        names = self._joint_names
        self._arm_names = [name for name in names if name not in self._gripper_names]
        if any(name not in names for name in self.config.velocity_joint_names):
            raise ValueError('Velocity joints must be selected articulation joints')
        if not names:
            raise ValueError('Robot articulation contains no movable joints')
        if any(native_names.count(name) != 1 for name in names):
            raise ValueError(
                f'Selected joints missing or ambiguous: {names}; native: {native_names}'
            )
        self._indices = np.array([self._robot.get_dof_index(name) for name in names])
        if len(set(self._indices)) != len(names):
            raise ValueError('Selected joints must have distinct articulation indices')
        self._limits = np.asarray(self._robot.get_dof_limits()).reshape(-1, 2)[self._indices]
        self._velocity_slots = np.array(
            [i for i, name in enumerate(names) if name in self.config.velocity_joint_names],
            dtype=int,
        )
        self._position_slots = np.array(
            [i for i, name in enumerate(names) if name not in self.config.velocity_joint_names],
            dtype=int,
        )
        if self.config.initial_qpos is not None:
            self._initial = np.asarray(self.config.initial_qpos, dtype=float)
        elif self._bundled:
            rest = dict(zip(CraneX7Config.ALL_JOINT_NAMES, CraneX7Config.REST_QPOS))
            self._initial = np.array([rest[name] for name in names])
        else:
            self._initial = np.zeros(len(names))
        if self._initial.shape != (len(names),):
            raise ValueError('Initial joint positions must match selected joints')
        if np.any(self._initial < self._limits[:, 0]) or np.any(
            self._initial > self._limits[:, 1]
        ):
            raise ValueError('Initial joint positions exceed articulation limits')
        controller = self._robot.get_articulation_controller()
        kp, kd = (np.array(value, copy=True) for value in controller.get_gains())
        kp[self._indices] = CraneX7Config.ARM_STIFFNESS
        kd[self._indices] = CraneX7Config.ARM_DAMPING
        kp[self._indices[self._velocity_slots]] = 0.0
        controller.set_gains(kps=kp, kds=kd)
        arm_force, gripper_force = CraneX7Config.controller_force_limits(
            self.config.actuator_profile
        )
        if isinstance(arm_force, tuple) and len(self._arm_names) != len(arm_force):
            raise ValueError('servo_envelope requires seven arm joints')
        force_by_name = dict(
            zip(self._arm_names, np.broadcast_to(arm_force, (len(self._arm_names),)))
        )
        force_by_name.update(
            {
                name: self.config.gripper_force_limit or gripper_force
                for name in self._gripper_names
            }
        )
        efforts = np.array([force_by_name[name] for name in names])
        self._robot.set_max_efforts(efforts, joint_indices=self._indices)
        controller.switch_control_mode('position')
        for index in self._indices[self._velocity_slots]:
            controller.switch_dof_control_mode(dof_index=int(index), mode='velocity')

    def _find_link(self, name: str) -> str:
        paths = [
            str(prim.GetPath())
            for prim in self._stage.Traverse()
            if str(prim.GetPath()).startswith('/World/Robot/') and prim.GetName() == name
        ]
        if len(paths) != 1:
            raise ValueError(f'Required robot link missing or ambiguous: {name}')
        return paths[0]

    def _setup_task(self) -> None:
        from isaacsim.core.api.objects import DynamicCuboid
        from isaacsim.core.prims import SingleRigidPrim

        self._cube = self._world.scene.add(
            DynamicCuboid(
                prim_path='/World/Cube',
                name='cube',
                position=np.array([0.15, 0.02, 0.02]),
                size=0.04,
                mass=0.05,
                color=np.array([12, 42, 160]) / 255,
            )
        )
        for suffix in ('a', 'b'):
            self._fingers.append(
                self._world.scene.add(
                    SingleRigidPrim(
                        prim_path=self._find_link(f'crane_x7_gripper_finger_{suffix}_link'),
                        name=f'finger_{suffix}',
                    )
                )
            )

    def _setup_camera(self) -> None:
        import isaacsim.core.experimental.utils.app as app_utils
        from isaacsim.sensors.experimental.rtx import CameraSensor
        from pxr import UsdGeom, UsdLux

        if self.config.camera_uid == 'scene_camera':
            path = '/World/SceneCamera'
        elif self.config.camera_uid == 'hand_camera' and self._bundled:
            path = self._find_link('crane_x7_gripper_base_link') + '/HandCamera'
        else:
            raise ValueError(
                'Use scene_camera for external robots; CRANE also supports hand_camera'
            )
        UsdGeom.Camera.Define(self._stage, path)
        light = UsdLux.DistantLight.Define(self._stage, '/World/NativeLight')
        light.CreateIntensityAttr(1000)
        self._camera = CameraSensor(
            path, resolution=(480, 640), annotators=['rgb', 'distance_to_image_plane']
        )
        if self.config.camera_uid == 'scene_camera':
            # USD cameras look along local -Z; rotation points toward the work surface.
            self._camera.authoring_object.set_world_poses(
                positions=np.array([[0.4, 0.0, 0.4]]),
                orientations=np.array([[0.9238795, 0.0, 0.3826834, 0.0]]),
            )
        else:
            self._camera.authoring_object.set_local_poses(
                translations=np.array([[0.0, -0.04, 0.02]]),
                orientations=np.array([[0.0, 1.0, 0.0, 0.0]]),
            )
        app_utils.play(commit=True)

    def _require_running(self) -> None:
        if not self._is_running or not self._app.is_running():
            raise RuntimeError('Isaac simulator is closed or stopped')

    def reset(self, seed: int | None = None) -> tuple[Observation, dict[str, Any]]:
        self._require_running()
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._world.reset()
        positions = np.clip(
            self._initial
            + self._rng.normal(0, self.config.robot_init_qpos_noise, self._initial.shape),
            self._limits[:, 0],
            self._limits[:, 1],
        )
        self._robot.set_joint_positions(positions, joint_indices=self._indices)
        self._robot.set_joint_velocities(np.zeros(len(positions)), joint_indices=self._indices)
        if len(self._position_slots):
            self._robot.apply_action(
                self._action_type(
                    joint_positions=positions[self._position_slots].copy(),
                    joint_indices=self._indices[self._position_slots],
                )
            )
        if len(self._velocity_slots):
            self._robot.apply_action(
                self._action_type(
                    joint_velocities=np.zeros(len(self._velocity_slots)),
                    joint_indices=self._indices[self._velocity_slots],
                )
            )
        if self._cube is not None:
            self._cube.set_world_pose(
                position=np.r_[np.array([0.15, 0.02]) + self._rng.uniform(-0.01, 0.01, 2), 0.02],
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            )
            self._cube.set_linear_velocity(np.zeros(3))
            self._cube.set_angular_velocity(np.zeros(3))
        self._steps = 0
        observation = self.get_observation()
        return observation, dict(observation.extra)

    def step(self, action: NDArray[np.generic] | JointCommand) -> StepResult:
        self._require_running()
        if isinstance(action, JointCommand):
            self._apply_named_command(action)
        else:
            if len(self._velocity_slots):
                raise ValueError('Use JointCommand for robots with velocity-controlled joints')
            targets = np.asarray(action, dtype=float)
            if (
                self._bundled
                and targets.shape == (8,)
                and self.all_joint_names == CraneX7Config.ALL_JOINT_NAMES
            ):
                targets = np.r_[targets, targets[-1]]
            if targets.shape != (len(self._indices),) or not np.all(np.isfinite(targets)):
                raise ValueError(f'Action must contain {len(self._indices)} finite joint targets')
            targets = np.clip(targets, self._limits[:, 0], self._limits[:, 1])
            self._robot.apply_action(
                self._action_type(joint_positions=targets, joint_indices=self._indices)
            )
        self._world.step(render=self.config.render_mode == 'human' and self._camera is None)
        self._steps += 1
        observation = self.get_observation()
        info = dict(observation.extra)
        if self._cube is None:
            reward, terminated = 0.0, False
        else:
            distance = info['gripper_to_cube_dist']
            height_progress = np.clip((info['cube_height'] - 0.02) / 0.12, 0, 1)
            terminated = info['success']
            reward = (
                5.0
                if terminated
                else float(1 - np.tanh(5 * distance) + height_progress + np.exp(-10 * distance))
            )
        return StepResult(
            observation, reward, terminated, self._steps >= self.config.max_episode_steps, info
        )

    def _apply_named_command(self, command: JointCommand) -> None:
        if not command.positions and not command.velocities:
            raise ValueError('JointCommand must specify at least one joint target')
        if set(command.positions) & set(command.velocities):
            raise ValueError('Position and velocity command joints must be disjoint')
        selected = set(self._joint_names)
        velocity = set(self.config.velocity_joint_names)
        if not set(command.positions) <= selected - velocity:
            raise ValueError('Position command contains unknown or velocity-controlled joints')
        if not set(command.velocities) <= velocity:
            raise ValueError('Velocity command contains unknown or position-controlled joints')
        # Validate both sets before sending either native command.
        native_actions = []
        for values, mode in (
            (command.positions, 'joint_positions'),
            (command.velocities, 'joint_velocities'),
        ):
            if not values:
                continue
            slots = np.array([self._joint_names.index(name) for name in values])
            raw_targets = [np.asarray(value, dtype=float) for value in values.values()]
            if any(value.size != 1 for value in raw_targets):
                raise ValueError('Single-environment JointCommand requires one target per joint')
            targets = np.array([value.item() for value in raw_targets])
            if not np.all(np.isfinite(targets)):
                raise ValueError('JointCommand targets must be finite')
            if mode == 'joint_positions':
                targets = np.clip(targets, self._limits[slots, 0], self._limits[slots, 1])
            native_actions.append(
                self._action_type(**{mode: targets}, joint_indices=self._indices[slots])
            )
        for action in native_actions:
            self._robot.apply_action(action)

    def _task_info(self) -> dict[str, Any]:
        if self._cube is None:
            return {'task': 'JointControl', 'task_reward': False}
        cube, _ = self._cube.get_world_pose()
        fingertips = []
        for finger in self._fingers:
            position, quaternion = finger.get_world_pose()
            offset = np.array([0.0, 0.0, 0.06])
            vector = quaternion[1:]
            cross = 2 * np.cross(vector, offset)
            fingertips.append(position + offset + quaternion[0] * cross + np.cross(vector, cross))
        distance = float(np.linalg.norm(cube - np.mean(fingertips, axis=0)))
        height_reached = bool(cube[2] >= 0.14)
        is_close = distance <= 0.05
        return {
            'cube_height': float(cube[2]),
            'gripper_to_cube_dist': distance,
            'height_reached': height_reached,
            'is_close': is_close,
            'success': height_reached and is_close,
            'fail': False,
        }

    def get_observation(self) -> Observation:
        self._require_running()
        rgb = depth = None
        if self._camera is not None:
            import omni.replicator.core as rep

            rep.orchestrator.step(rt_subframes=2, pause_timeline=False, delta_time=0.0)
            color, _ = self._camera.get_data('rgb')
            rgb = np.asarray(color)
            if rgb.shape not in ((480, 640, 3), (480, 640, 4)):
                raise RuntimeError(f'Isaac camera returned invalid RGB shape: {rgb.shape}')
            rgb = rgb[..., :3].copy()
            if self.config.obs_mode == 'rgbd':
                data, _ = self._camera.get_data('distance_to_image_plane')
                depth = np.asarray(data).reshape(480, 640).copy()
        return Observation(rgb, depth, self.get_qpos(), self.get_qvel(), self._task_info())

    def get_qpos(self) -> NDArray[np.generic]:
        self._require_running()
        return np.asarray(self._robot.get_joint_positions(joint_indices=self._indices)).copy()

    def get_qvel(self) -> NDArray[np.generic]:
        self._require_running()
        return np.asarray(self._robot.get_joint_velocities(joint_indices=self._indices)).copy()

    def close(self) -> None:
        self._is_running = False
        try:
            if self._world is not None:
                try:
                    self._world.stop()
                finally:
                    self._world.clear_instance()
        finally:
            try:
                if self._app is not None:
                    app, self._app = self._app, None
                    app.close()
            finally:
                self._world = self._robot = self._camera = self._cube = None
                self._fingers = []
                if self._assets is not None:
                    self._assets.cleanup()
                    self._assets = None
