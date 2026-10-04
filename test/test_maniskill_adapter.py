"""Deterministic adapter contracts without loading SAPIEN or robot assets."""

import sys
from importlib import import_module
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from usim.types import JointCommand, SimulatorConfig

gym = pytest.importorskip('gymnasium')
torch = pytest.importorskip('torch')

ManiSkillSimulator = import_module('usim_maniskill.adapter').ManiSkillSimulator


class FakeEnv:
    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.unwrapped = self
        self.closed = 0
        self.steps = 0
        self.num_envs = kwargs['num_envs']
        self.qpos = torch.arange(self.num_envs * 3, dtype=torch.float32).reshape(-1, 3)
        self.qvel = self.qpos + 10
        self.joints = [SimpleNamespace(name=name) for name in ('finger', 'shoulder', 'elbow')]
        robot = SimpleNamespace(
            fixed_root_link=torch.ones(self.num_envs, dtype=torch.bool),
            get_active_joints=lambda: self.joints,
            get_qpos=lambda: self.qpos,
            get_qvel=lambda: self.qvel,
        )
        self.agent = SimpleNamespace(
            robot=robot,
            controller=SimpleNamespace(
                reset=lambda: None,
                controllers={
                    'gripper': SimpleNamespace(joints=self.joints[:1]),
                    'arm': SimpleNamespace(joints=self.joints[1:]),
                },
            ),
            reset=self.reset_agent,
        )
        self.gpu_sim_enabled = False

    def reset_agent(self, qpos):
        self.qpos = qpos.clone()
        self.qvel.zero_()

    def get_info(self):
        return {'success': torch.zeros(self.num_envs, dtype=torch.bool)}

    def get_obs(self, info=None):
        if self.kwargs['obs_mode'] == 'state':
            return self.qpos.clone()
        rgb = torch.arange(self.num_envs, dtype=torch.uint8)[:, None, None, None]
        rgb = rgb.expand(-1, 2, 4, 3).clone() + 100
        return {
            'sensor_data': {
                'base_camera': {
                    'rgb': rgb,
                    'depth': torch.arange(self.num_envs).reshape(-1, 1, 1, 1).expand(-1, 2, 4, 1),
                }
            }
        }

    def reset(self, seed=None):
        self.seed = seed
        self.steps = 0
        return self.get_obs(), self.get_info()

    def step(self, action):
        assert action.shape == (self.num_envs, 3)
        self.steps += 1
        self.qpos += 1
        return (
            self.get_obs(),
            torch.arange(self.num_envs, dtype=torch.float32) + 0.5,
            torch.arange(self.num_envs) % 2 == 1,
            torch.full((self.num_envs,), self.steps >= self.kwargs['max_episode_steps']),
            self.get_info(),
        )

    def close(self):
        self.closed += 1


@pytest.fixture
def make_env(monkeypatch):
    monkeypatch.setitem(sys.modules, 'mani_skill', ModuleType('mani_skill'))
    monkeypatch.setitem(sys.modules, 'mani_skill.envs', ModuleType('mani_skill.envs'))
    made = []

    def make(env_id, **kwargs):
        assert env_id == 'PickCube-v1' or env_id.startswith('PickPlace-CRANE-X7')
        env = FakeEnv(kwargs)
        made.append(env)
        return env

    monkeypatch.setattr(gym, 'make', make)
    return made


def config(**kwargs):
    return SimulatorConfig(
        env_id='PickCube-v1', robot_uid='panda', camera_uid='base_camera', **kwargs
    )


@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
def test_rendering_uses_the_selected_compute_backend(make_env, backend):
    # Given a compute selection and an image observation request.
    sim = ManiSkillSimulator(config(backend=backend, obs_mode='rgbd', render_mode='rgb_array'))
    try:
        # When the native scene is constructed.
        # Then CPU observations never implicitly request CUDA image buffers.
        assert make_env[0].kwargs['render_backend'] == backend
    finally:
        sim.close()


@pytest.mark.parametrize('count', [1, 3])
def test_registered_robot_batch_and_episode_contract(make_env, count):
    sim = ManiSkillSimulator(config(n_envs=count, backend='auto', max_episode_steps=2))
    env = make_env[0]
    assert env.kwargs['robot_uids'] == 'panda'
    assert env.kwargs['num_envs'] == count
    assert env.kwargs['sim_config'] == {'control_freq': 30, 'sim_freq': 150}
    assert sim.arm_joint_names == ['shoulder', 'elbow']
    assert sim.gripper_joint_names == ['finger']
    obs, info = sim.reset(seed=42)
    assert env.seed == 42
    np.testing.assert_array_equal(obs.qpos, sim.get_qpos())
    assert info['success'].shape == (count,)
    assert obs.qpos.shape == ((3,) if count == 1 else (count, 3))
    assert obs.rgb_image.shape == ((2, 4, 3) if count == 1 else (count, 2, 4, 3))
    assert obs.depth_image.shape == ((2, 4, 1) if count == 1 else (count, 2, 4, 1))
    assert sim.joint_names == ['finger', 'shoulder', 'elbow']
    expected_qpos = np.arange(count * 3).reshape(count, 3)
    np.testing.assert_array_equal(obs.qpos, expected_qpos[0] if count == 1 else expected_qpos)
    result = sim.step(np.zeros((count, 3)))
    if count == 1:
        assert result.reward == 0.5
        assert result.terminated is False
        assert result.truncated is False
    else:
        np.testing.assert_array_equal(result.reward, [0.5, 1.5, 2.5])
        np.testing.assert_array_equal(result.terminated, [False, True, False])
        np.testing.assert_array_equal(result.truncated, [False] * count)
        np.testing.assert_array_equal(result.observation.rgb_image[:, 0, 0, 0], [100, 101, 102])
    final = sim.step(np.zeros((count, 3)))
    assert np.all(final.truncated)
    sim.close()
    sim.close()
    assert env.closed == 1
    assert not sim.is_running
    for call in (sim.get_observation, sim.get_qpos, sim.get_qvel, sim.reset):
        with pytest.raises(RuntimeError, match='closed'):
            call()


def test_initial_qpos_preserves_native_order_independent_of_groups(make_env):
    sim = ManiSkillSimulator(
        config(
            obs_mode='state',
            arm_joint_names=('elbow', 'shoulder'),
            gripper_joint_names=('finger',),
            joint_names=('finger', 'shoulder', 'elbow'),
            initial_qpos=(0.2, 0.3, 0.4),
        )
    )
    np.testing.assert_allclose(sim.get_qpos(), [0.2, 0.3, 0.4])
    np.testing.assert_allclose(make_env[0].qpos.numpy(), [[0.2, 0.3, 0.4]])
    assert sim.get_observation().rgb_image is None
    sim.step(np.zeros((1, 3)))
    obs, _ = sim.reset(seed=10)
    np.testing.assert_allclose(obs.qpos, [0.2, 0.3, 0.4])
    np.testing.assert_array_equal(
        obs.extra['raw_obs'], np.array([[0.2, 0.3, 0.4]], dtype=np.float32)
    )
    sim.close()


def test_missing_camera_closes_constructed_environment(make_env):
    bad = config()
    bad.camera_uid = 'hand_camera'
    with pytest.raises(ValueError, match="available: \\['base_camera'\\]"):
        ManiSkillSimulator(bad)
    assert make_env[0].closed == 1


def test_unknown_joint_closes_constructed_environment(make_env):
    with pytest.raises(ValueError, match='Unknown ManiSkill joints'):
        ManiSkillSimulator(config(arm_joint_names=('not_a_joint',)))
    assert make_env[0].closed == 1


@pytest.mark.parametrize(
    'kwargs, message',
    [
        ({'n_envs': 2}, 'requires'),
        ({'sim_rate': 29.5}, 'integer'),
        ({'gripper_force_limit': 1}, 'Force overrides'),
        ({'initial_qpos': (0.0,)}, 'reported joints'),
    ],
)
def test_unsupported_configs_are_explicit(make_env, kwargs, message):
    with pytest.raises(ValueError, match=message):
        ManiSkillSimulator(config(**kwargs))
    if make_env:
        assert make_env[0].closed == 1


def test_robot_path_rejected_before_loading_environment(make_env, tmp_path):
    path = tmp_path / 'robot.urdf'
    path.touch()
    with pytest.raises(ValueError, match='register a ManiSkill agent'):
        ManiSkillSimulator(config(robot_path=str(path)))
    assert not make_env


def test_rgb_float_conversion_preserves_batch(make_env):
    sim = ManiSkillSimulator(config(n_envs=2, backend='auto'))
    raw = {
        'sensor_data': {
            'base_camera': {
                'rgb': torch.tensor([0.0, 1.0]).reshape(2, 1, 1, 1).expand(-1, 1, 1, 3),
            }
        }
    }
    obs = sim._convert_observation(raw)
    assert obs.rgb_image.dtype == np.uint8
    np.testing.assert_array_equal(obs.rgb_image[:, 0, 0, 0], [0, 255])
    sim.close()


def test_headless_state_mode_disables_renderer(make_env):
    sim = ManiSkillSimulator(config(obs_mode='state', render_mode='none'))
    assert make_env[0].kwargs['render_mode'] is None
    assert make_env[0].kwargs['render_backend'] == 'none'
    sim.close()


def test_joint_names_reject_controller_order_guessing(make_env):
    with pytest.raises(ValueError, match='native articulation order'):
        ManiSkillSimulator(config(joint_names=('shoulder', 'elbow', 'finger')))
    assert make_env[0].closed == 1


def test_full_native_joint_order_includes_mobile_dofs(make_env, monkeypatch):
    original_make = gym.make

    def make(*args, **kwargs):
        env = original_make(*args, **kwargs)
        env.joints.append(SimpleNamespace(name='wheel'))
        env.qpos = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        env.qvel = env.qpos + 10
        env.agent.robot.fixed_root_link = torch.zeros(1, dtype=torch.bool)
        return env

    monkeypatch.setattr(gym, 'make', make)
    sim = ManiSkillSimulator(
        config(obs_mode='state', fixed_base=False, velocity_joint_names=('wheel',))
    )
    assert sim.joint_names == ['finger', 'shoulder', 'elbow', 'wheel']
    assert sim.arm_joint_names == ['shoulder', 'elbow']
    assert sim.gripper_joint_names == ['finger']
    np.testing.assert_array_equal(sim.get_qpos(), [0, 1, 2, 3])
    native_action = np.zeros((1, 3))
    result = sim.step(native_action)
    np.testing.assert_array_equal(result.observation.qpos, [1, 2, 3, 4])
    sim.close()


def test_named_joint_command_is_rejected_without_stepping(make_env):
    sim = ManiSkillSimulator(config(obs_mode='state'))
    with pytest.raises(ValueError, match='native action vector'):
        sim.step(JointCommand(positions={'elbow': 0.4}, velocities={'shoulder': 1.0}))
    assert make_env[0].steps == 0
    sim.close()


def test_fixed_base_must_match_registered_robot(make_env):
    with pytest.raises(ValueError, match='registered agent root'):
        ManiSkillSimulator(config(fixed_base=False))
    assert make_env[0].closed == 1


def test_initial_qpos_syncs_gpu_kinematics_before_controller_reset(make_env, monkeypatch):
    original_make = gym.make
    events = []

    def make(*args, **kwargs):
        env = original_make(*args, **kwargs)
        env.gpu_sim_enabled = True
        env.scene = SimpleNamespace(
            _gpu_apply_all=lambda: events.append('apply'),
            px=SimpleNamespace(
                gpu_update_articulation_kinematics=lambda: events.append('kinematics')
            ),
            _gpu_fetch_all=lambda: events.append('fetch'),
        )
        env.agent.controller.reset = lambda: events.append('controller')
        return env

    monkeypatch.setattr(gym, 'make', make)
    sim = ManiSkillSimulator(
        config(obs_mode='state', n_envs=2, backend='gpu', initial_qpos=(0.2, 0.3, 0.4))
    )
    assert events == ['apply', 'kinematics', 'fetch', 'controller']
    np.testing.assert_allclose(sim.get_qpos(), [[0.2, 0.3, 0.4]] * 2)
    sim.close()


@pytest.mark.parametrize('visual_domain', [False, True])
def test_crane_force_camera_and_visual_domain_settings_are_preserved(
    make_env, monkeypatch, visual_domain
):
    class CraneAgent:
        arm_force_limit = 0
        gripper_force_limit = 0

    class PickPlace:
        use_scene_camera = False

    agent_module = ModuleType('usim_maniskill.agent')
    agent_module.CraneX7 = CraneAgent
    env_module = ModuleType('usim_maniskill.environments')
    env_module.PickPlace = PickPlace
    monkeypatch.setitem(sys.modules, 'usim_maniskill.agent', agent_module)
    monkeypatch.setitem(sys.modules, 'usim_maniskill.environments', env_module)
    env_id = 'PickPlace-CRANE-X7-visual-domain' if visual_domain else 'PickPlace-CRANE-X7'
    sim = ManiSkillSimulator(
        SimulatorConfig(
            env_id=env_id,
            camera_uid='scene_camera',
            obs_mode='state',
            robot_init_qpos_noise=0.05,
            actuator_profile='servo_envelope',
            gripper_force_limit=3.0,
        )
    )
    assert CraneAgent.arm_force_limit == (10.0, 10.0, 4.0, 4.0, 4.0, 4.0, 4.0)
    assert CraneAgent.gripper_force_limit == 3.0
    assert make_env[0].kwargs['robot_init_qpos_noise'] == 0.05
    assert make_env[0].kwargs['robot_uids'] == 'CRANE-X7'
    assert PickPlace.use_scene_camera is (not visual_domain)
    sim.close()
