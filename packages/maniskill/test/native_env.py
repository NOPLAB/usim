"""A deterministic subset of the native environment/agent APIs used by contract tests."""

from abc import ABC, abstractmethod
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest

from usim.model import ImportRequirements, ModelSource
from usim.specs import AgentSpec
from packages.maniskill.test.native_models import (
    ControllerConfig,
    NativeFailure,
    NativePose,
    NativeScene,
    Robot,
    SubController,
)


class NativeAgent(ABC):
    keyframes = {}
    robot: Robot

    def __init__(self, scene, control_freq, control_mode, initial_pose=None, **kwargs):
        self.scene = scene
        self._load_articulation(initial_pose)
        self.controller = SimpleNamespace(
            controllers={
                name: SubController(self.robot, name, config)
                for name, config in self._controller_configs[control_mode].items()
                if name != 'balance_passive_force'
            },
            reset=lambda: self.robot.targets.clear(),
        )
        self.native_actions = []
        self.single_action_space = SimpleNamespace(shape=(1,))

    @abstractmethod
    def _load_articulation(self, initial_pose) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def _controller_configs(self):
        raise NotImplementedError

    def reset(self, qpos=None):
        if qpos is not None:
            self.robot.set_qpos(qpos)
        self.robot.set_qvel(np.zeros_like(self.robot.qvel))
        self.robot.set_qf(np.zeros_like(self.robot.qf))

    def get_proprioception(self):
        return {'qpos': self.robot.qpos, 'qvel': self.robot.qvel}

    def set_action(self, action):
        self.native_actions.append(action)

    @property
    def _sensor_configs(self):
        return []


class RegisteredAgent(NativeAgent):
    def _load_articulation(self, initial_pose):
        self.robot = Robot(
            'registered',
            ['shoulder', 'wheel'],
            ['base', 'arm'],
            self.scene.count,
            True,
            initial_pose,
        )

    @property
    def _controller_configs(self):
        return {'native_delta': {'arm': ControllerConfig(['shoulder'])}}


@pytest.fixture
def native(monkeypatch):
    made = []
    settings = SimpleNamespace(fail_build=False, fail_step=False)

    class BaseEnv(ABC):
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.device = 'cpu'
            self.num_envs = kwargs['num_envs']
            self.scene = NativeScene(self.num_envs, settings.fail_build)
            self.gpu_sim_enabled = kwargs['sim_backend'] in {'gpu', 'physx_cuda'}
            self.unwrapped = self
            self.closed = 0
            self.steps = 0
            self.robot_uids = kwargs['robot_uids']
            self._control_freq = kwargs['sim_config']['control_freq']
            self._control_mode = kwargs['control_mode']
            made.append(self)
            self._load_agent({})
            self._load_scene({})
            self.cameras = self._default_sensor_configs
            self.reset()

        @abstractmethod
        def _load_scene(self, options) -> None:
            raise NotImplementedError

        @abstractmethod
        def _initialize_episode(self, env_idx, options) -> None:
            raise NotImplementedError

        @abstractmethod
        def _get_obs_agent(self):
            raise NotImplementedError

        @property
        @abstractmethod
        def _default_sensor_configs(self):
            raise NotImplementedError

        def _load_agent(self, options, initial_agent_poses=None):
            if self.robot_uids == 'none':
                self.agent = None
                return
            cls = RegisteredAgent if self.robot_uids == 'registered' else self.robot_uids
            self.agent = cls(
                self.scene,
                self._control_freq,
                self._control_mode,
                initial_pose=initial_agent_poses,
            )

        def reset(self, seed=None):
            self.seed = seed
            self.steps = 0
            self._initialize_episode(np.arange(self.num_envs), {})
            return self.get_obs(), {'seed': seed}

        def get_obs(self):
            obs = self._get_obs_agent()
            if self.cameras:
                obs = dict(obs)
                obs['sensor_data'] = {
                    camera.uid: {
                        'rgb': np.zeros(
                            (self.num_envs, camera.height, camera.width, 3), dtype=np.uint8
                        ),
                        'depth': np.full(
                            (self.num_envs, camera.height, camera.width, 1), 1500, dtype=np.int16
                        ),
                    }
                    for camera in self.cameras
                }
            return obs

        def step(self, action):
            assert action is None
            if settings.fail_step:
                raise NativeFailure('native step failed')
            self.steps += 1
            if self.agent:
                self.agent.robot.qpos += 0.1
            return (
                self.get_obs(),
                np.zeros(self.num_envs),
                np.zeros(self.num_envs, dtype=bool),
                np.zeros(self.num_envs, dtype=bool),
                {},
            )

        def close(self):
            self.closed += 1

    modules = {
        'mani_skill': {},
        'mani_skill.envs': {},
        'mani_skill.envs.sapien_env': {'BaseEnv': BaseEnv},
        'mani_skill.agents': {},
        'mani_skill.agents.base_agent': {'BaseAgent': NativeAgent},
        'mani_skill.agents.controllers': {
            'PDJointPosControllerConfig': ControllerConfig,
            'PDJointVelControllerConfig': ControllerConfig,
            'PassiveControllerConfig': ControllerConfig,
        },
        'mani_skill.sensors': {},
        'mani_skill.sensors.camera': {'CameraConfig': lambda **kwargs: SimpleNamespace(**kwargs)},
        'mani_skill.utils': {},
        'mani_skill.utils.common': {'to_tensor': lambda value, device: np.asarray(value)},
        'sapien': {'Pose': NativePose},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    settings.BaseEnv = BaseEnv
    return made, settings


@pytest.fixture
def sources(tmp_path):
    urdf = tmp_path / 'robot.urdf'
    urdf.write_text(
        '<robot name="test"><link name="base"/><link name="arm"/><link name="wheel"/>'
        '<joint name="shoulder" type="revolute"><parent link="base"/><child link="arm"/>'
        '<limit lower="-2" upper="2" effort="10" velocity="10"/></joint>'
        '<joint name="wheel" type="continuous"><parent link="arm"/><child link="wheel"/>'
        '</joint></robot>'
    )
    mjcf = tmp_path / 'robot.xml'
    mjcf.write_text(
        '<mujoco><worldbody><body name="base"><freejoint/><body name="arm">'
        '<joint name="shoulder"/><geom type="box" size=".1 .1 .1"/>'
        '<body name="wheel"><joint name="wheel"/><geom type="sphere" size=".1"/>'
        '</body></body></body></worldbody></mujoco>'
    )
    return ModelSource.from_path(urdf), ModelSource.from_path(mjcf)


def direct(source, **kwargs):
    requirements = ImportRequirements()
    if source.format.value == 'mjcf':
        requirements = ImportRequirements(
            preserve=frozenset({'joint_structure', 'collision'}), allow_loss=frozenset({'inertia'})
        )
    return AgentSpec('robot', source, requirements=requirements, **kwargs)
