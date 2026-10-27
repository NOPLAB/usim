"""Lazy native construction. Factories are local to an open, never registered globally."""

from collections.abc import Mapping
from typing import TYPE_CHECKING

from usim.capability import assert_never
from usim.model import ModelError, ModelFormat, ModelSource
from usim.specs import ExecutionSpec, Pose, RegisteredAgentRef, SceneSpec
from usim_maniskill.models import Asset
from usim_maniskill.native_agent import _agent_factory

if TYPE_CHECKING:
    import sapien
    from mani_skill.envs.sapien_env import BaseEnv
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.structs import Actor, Articulation


def native_pose(pose: Pose) -> 'sapien.Pose':
    import sapien

    x, y, z, w = pose.orientation_xyzw
    return sapien.Pose(p=pose.position, q=(w, x, y, z))


def build_asset(
    scene: 'ManiSkillScene', asset: Asset, name: str, pose: Pose
) -> 'Actor | Articulation':
    """Parse every builder before building, so a multi-root asset is never truncated."""
    match asset.source.format:
        case ModelFormat.URDF:
            loader = scene.create_urdf_loader()
        case ModelFormat.MJCF:
            loader = scene.create_mjcf_loader()
        case ModelFormat.USD | ModelFormat.SDF:
            raise ModelError('No native ManiSkill loader for this format')
        case unreachable:
            assert_never(unreachable)
    loader.name = name
    loader.fix_root_link = asset.fixed_base
    kwargs = {}
    if asset.source.resources:
        kwargs['package_dir'] = str(asset.source.resources[0])
    parsed = loader.parse(str(asset.source.path), **kwargs)
    articulations = parsed['articulation_builders']
    actors = parsed['actor_builders']
    if len(articulations) + len(actors) != 1:
        raise ModelError('An entity must produce exactly one native actor or articulation')
    if parsed['cameras'] and 'sensors' not in asset.requirements.allow_loss:
        raise ModelError('Authored cameras must be expressed as typed SensorSpec entries')
    builder = (articulations or actors)[0]
    builder.set_name(name)
    builder.initial_pose = native_pose(pose)
    if articulations:
        result = builder.build(fix_root_link=asset.fixed_base)
        actual = tuple(joint.name for joint in result.get_active_joints())
        if set(actual) != {name for name, _ in asset.joints}:
            raise ModelError(f'Native joint mapping differs from the imported asset: {actual}')
        return result
    if asset.joints:
        raise ModelError('The native loader discarded the requested articulation')
    if asset.fixed_base:
        # SAPIEN 3.0.3 / ManiSkill 3.0.1's actor builder applies explicit URDF
        # inertia to static components, which have no mass/inertia setters.
        # Static bodies have no inertial dynamics; retain the authored inertia
        # on dynamic actors and disable only that invalid static assignment.
        builder._auto_inertial = True
        return builder.build_static(name=name)
    return builder.build_dynamic(name=name)


def create_environment(
    spec: SceneSpec, execution: ExecutionSpec, assets: Mapping[str, Asset]
) -> 'BaseEnv':
    """Construct either an existing native task or an unregistered generic scene."""
    import mani_skill.envs  # noqa: F401
    from mani_skill.envs.sapien_env import BaseEnv

    agent_spec = spec.agents[0] if spec.agents else None
    robot = 'none'
    if agent_spec is not None:
        match agent_spec.model:
            case RegisteredAgentRef(uid=uid):
                if uid == 'CRANE-X7':
                    from usim_maniskill.agent import CraneX7  # noqa: F401

                robot = uid
            case ModelSource():
                robot = _agent_factory(agent_spec, assets[agent_spec.name])
            case unreachable:
                assert_never(unreachable)
    cpu = execution.backend in {'cpu', 'physx_cpu'} or (
        execution.backend == 'auto' and execution.n_envs == 1
    )
    render_backend = 'cpu' if cpu else 'gpu'
    if execution.render_mode == 'none' and execution.obs_mode in {'none', 'state', 'state_dict'}:
        render_backend = 'none'
    kwargs = {
        'robot_uids': robot,
        'control_mode': agent_spec.control.native_mode if agent_spec else None,
        'obs_mode': execution.obs_mode,
        'render_mode': None if execution.render_mode == 'none' else execution.render_mode,
        'render_backend': render_backend,
        'sim_backend': execution.backend,
        'num_envs': execution.n_envs,
        'sim_config': {
            'control_freq': int(execution.sim_rate),
            'sim_freq': int(execution.sim_rate) * 5,
        },
    }
    if execution.native_task != 'JointControl':
        import gymnasium as gym

        if execution.native_task.startswith('PickPlace-CRANE-X7'):
            import usim_maniskill.environments  # noqa: F401
        env = gym.make(
            execution.native_task, max_episode_steps=execution.max_episode_steps, **kwargs
        )
        native = env.unwrapped
        step_action = native._step_action
        native.usim_action = None

        def submitted_action(action):
            # Controllers have already received the command. Return it for native
            # action-dependent rewards without applying delta controllers twice.
            result = step_action(action)
            return native.usim_action if action is None else result

        native._step_action = submitted_action
        return env

    class GenericScene(BaseEnv):
        SUPPORTED_REWARD_MODES = ['none']
        SUPPORTED_ROBOTS = None

        def _load_agent(self, options):
            pose = native_pose(agent_spec.pose) if agent_spec else None
            super()._load_agent(options, initial_agent_poses=pose)
            self.usim_entities = {}
            if agent_spec:
                self.usim_entities[agent_spec.name] = self.agent.robot

        def _load_scene(self, options):
            for entity in spec.entities:
                self.usim_entities[entity.name] = build_asset(
                    self.scene, assets[entity.name], entity.name, entity.pose
                )

        def _initialize_episode(self, env_idx, options):
            for entity in spec.entities:
                native = self.usim_entities[entity.name]
                native.set_pose(native_pose(entity.pose))
                if hasattr(native, 'get_active_joints'):
                    zeros = [0.0] * len(native.get_active_joints())
                    native.set_qpos(zeros)
                    native.set_qvel(zeros)
                    native.set_qf(zeros)
            if agent_spec:
                self.agent.robot.set_pose(native_pose(agent_spec.pose))
                qpos = agent_spec.control.initial_qpos
                if qpos is None:
                    rest = self.agent.keyframes.get('rest')
                    qpos = (
                        rest.qpos
                        if rest is not None
                        else ([0.0] * len(self.agent.robot.get_active_joints()))
                    )
                if len(qpos) != len(self.agent.robot.get_active_joints()):
                    raise ModelError('Initial positions must match native joint count')
                self.agent.reset(qpos)

        def _get_obs_agent(self):
            return self.agent.get_proprioception() if self.agent is not None else {}

        def _get_obs_extra(self, info):
            return {name: entity.pose.raw_pose for name, entity in self.usim_entities.items()}

        @property
        def _default_sensor_configs(self):
            from mani_skill.sensors.camera import CameraConfig

            configs = []
            if agent_spec:
                native_names = {config.uid for config in self.agent._sensor_configs}
                for sensor in agent_spec.sensors:
                    if sensor.name in native_names:
                        raise ModelError(
                            f'Sensor name collides with a native sensor: {sensor.name}'
                        )
                    mount = None
                    if sensor.parent is not None:
                        if sensor.parent not in self.agent.robot.links_map:
                            raise ModelError(f'Unknown sensor parent link: {sensor.parent}')
                        mount = self.agent.robot.links_map[sensor.parent]
                    configs.append(
                        CameraConfig(
                            uid=sensor.name,
                            pose=native_pose(sensor.pose),
                            width=sensor.width,
                            height=sensor.height,
                            fov=1.2,
                            near=0.01,
                            far=100.0,
                            mount=mount,
                        )
                    )
            return configs

        def evaluate(self):
            return {}

        @property
        def _default_human_render_camera_configs(self):
            from mani_skill.sensors.camera import CameraConfig
            from mani_skill.utils.sapien_utils import look_at

            return [
                CameraConfig(
                    uid='render_camera',
                    pose=look_at([2, -2, 2], [0, 0, 0.5]),
                    width=512,
                    height=512,
                    fov=1.2,
                    near=0.01,
                    far=100.0,
                )
            ]

        def close(self):
            if hasattr(self, 'usim_entities'):
                self.usim_entities.clear()
            super().close()

    # Retain the partially constructed instance so failures after native allocation
    # can close it too. BaseEnv.close clears the scene and viewer.
    env = GenericScene.__new__(GenericScene)
    succeeded = False
    try:
        GenericScene.__init__(env, reward_mode='none', **kwargs)
        succeeded = True
        return env
    finally:
        if not succeeded and hasattr(env, 'scene'):
            env.close()
