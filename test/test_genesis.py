"""Deterministic Genesis boundary tests without installing native dependencies."""

import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from usim.types import JointCommand, SimulatorConfig
from usim_genesis import GenesisSimulator
from usim.robots.crane_x7 import CraneX7Config


class Entity:
    """In-memory batched state with persistent position, velocity, and force commands."""

    def __init__(self, names, position=(0, 0, 0)):
        self.names = names
        self.position = position
        self.force_ranges = []
        self.n_vverts = 2

    def build(self, count):
        self.qpos = np.zeros((count, len(self.names)))
        self.qvel = np.zeros_like(self.qpos)
        self.target = np.zeros_like(self.qpos)
        self.mode = np.full(self.qpos.shape, 'position', dtype=object)
        self.pos = np.tile(self.position, (count, 1))
        self.quat = np.tile([1, 0, 0, 0], (count, 1))

    def get_joint(self, name):
        if name not in self.names:
            return None
        return SimpleNamespace(dofs_idx_local=[self.names.index(name)])

    def set_dofs_position(self, values, indices, envs_idx):
        self.qpos[np.ix_(envs_idx, indices)] = values

    def set_dofs_velocity(self, values, indices, envs_idx):
        self.qvel[np.ix_(envs_idx, indices)] = values

    def command(self, mode, values, indices, envs_idx):
        ids = np.arange(len(self.qpos)) if envs_idx is None else envs_idx
        self.target[np.ix_(ids, indices)] = values
        self.mode[np.ix_(ids, indices)] = mode

    def control_dofs_position(self, values, indices, envs_idx=None):
        self.command('position', values, indices, envs_idx)

    def control_dofs_velocity(self, values, indices, envs_idx=None):
        self.command('velocity', values, indices, envs_idx)

    def control_dofs_force(self, values, indices, envs_idx=None):
        self.command('force', values, indices, envs_idx)

    def get_dofs_position(self, indices):
        return self.qpos[:, indices].copy()

    def get_dofs_velocity(self, indices):
        return self.qvel[:, indices].copy()

    def set_dofs_kp(self, kp, dofs_idx_local):
        self.kp = kp

    def set_dofs_kv(self, kv, dofs_idx_local):
        self.kv = kv

    def set_dofs_force_range(self, lower, upper, indices):
        self.force_ranges.append((np.asarray(lower), np.asarray(upper), indices))

    def set_pos(self, values, envs_idx):
        self.pos[envs_idx] = values

    def set_quat(self, values, envs_idx):
        self.quat[envs_idx] = values

    def get_pos(self):
        return self.pos

    def get_vverts(self):
        return np.tile([[-1, -1, 0], [1, 1, 2]], (len(self.pos), 1, 1))

    def get_link(self, name):
        assert name == 'crane_x7_gripper_base_link'
        return SimpleNamespace(get_pos=lambda: np.tile([0.15, 0.02, 0.1], (len(self.pos), 1)))


class Scene:
    def __init__(self, native, **options):
        self.native = native
        self.options = options
        self.entities = []
        self.steps = 0
        self.destroyed = 0
        native.scenes.append(self)

    def add_entity(self, morph):
        names = self.native.names if hasattr(morph, 'file') else []
        entity = Entity(names, getattr(morph, 'pos', (0, 0, 0)))
        self.entities.append(entity)
        self.native.morphs_added.append(morph)
        return entity

    def add_camera(self, **options):
        self.camera_pose = {}
        return SimpleNamespace(
            render=self.render, set_pose=lambda **pose: self.camera_pose.update(pose)
        )

    def render(self, rgb, depth):
        colors = np.arange(self.n_envs, dtype=np.uint8)[:, None, None, None]
        image = np.broadcast_to(colors, (self.n_envs, 2, 3, 3)).copy()
        return image, np.ones((self.n_envs, 2, 3)) if depth else None, None, None

    def build(self, n_envs):
        self.n_envs = n_envs
        for entity in self.entities:
            entity.build(n_envs)

    def reset(self, envs_idx):
        for entity in self.entities:
            entity.qpos[envs_idx] = 0
            entity.qvel[envs_idx] = 0

    def step(self):
        self.steps += 1
        for entity in self.entities:
            entity.qvel = np.where(
                entity.mode == 'position',
                entity.target - entity.qpos,
                np.where(entity.mode == 'velocity', entity.target, entity.qvel + entity.target),
            )
            entity.qpos += entity.qvel

    def destroy(self):
        self.destroyed += 1


@pytest.fixture
def native(monkeypatch):
    module = SimpleNamespace(
        _initialized=False,
        cpu='cpu',
        cuda='cuda',
        backend=None,
        names=['second', 'first', 'finger'],
        scenes=[],
        morphs_added=[],
        init_count=0,
    )

    def init(backend):
        assert not module._initialized
        module._initialized = True
        module.backend = backend
        module.init_count += 1

    module.init = init
    module.Scene = lambda **options: Scene(module, **options)
    module.options = SimpleNamespace(SimOptions=SimpleNamespace, VisOptions=SimpleNamespace)
    module.morphs = SimpleNamespace(
        MJCF=SimpleNamespace, URDF=SimpleNamespace, Plane=SimpleNamespace, Box=SimpleNamespace
    )
    monkeypatch.setitem(sys.modules, 'genesis', module)
    return module


@pytest.fixture
def config(tmp_path):
    path = tmp_path / 'external.urdf'
    path.write_text('<robot name="external"/>')
    return SimulatorConfig(
        env_id='JointControl',
        robot_path=str(path),
        robot_uid='external',
        arm_joint_names=('first', 'second'),
        gripper_joint_names=('finger',),
        initial_qpos=(0.2, -0.1, 0.3),
        robot_init_qpos_noise=0,
        render_mode='none',
        max_episode_steps=2,
    )


@pytest.mark.parametrize(
    'mode', ['pd_joint_pos', 'pd_joint_delta_pos', 'pd_joint_vel', 'joint_force']
)
def test_mapped_action_advances_configured_controller(native, config, mode):
    # Given a model whose DOF order differs from the explicit action mapping.
    sim = GenesisSimulator(replace(config, control_mode=mode))
    before = sim.get_qpos()
    # When executing one mapped action.
    result = sim.step(np.array([0.4, 0.5, 0.6]))
    # Then the observation reflects the requested mode and joint order.
    expected = [0.4, 0.5, 0.6] if mode == 'pd_joint_pos' else before + [0.4, 0.5, 0.6]
    np.testing.assert_allclose(result.observation.qpos, np.atleast_2d(expected))
    assert result.reward.tolist() == [0]
    assert result.truncated.tolist() == [False]
    sim.close()


def test_partial_reset_clears_targets_without_advancing_other_environment(native, config):
    # Given two environments after an action that reaches the episode limit.
    sim = GenesisSimulator(replace(config, n_envs=2, max_episode_steps=1))
    sim.step(np.array([[1, 2, 3], [4, 5, 6]]))
    scene = native.scenes[0]
    # When resetting only environment zero.
    observation, _ = sim.reset(seed=5, env_ids=np.array([0]))
    # Then the selected state is exact and the other environment is untouched.
    np.testing.assert_allclose(observation.qpos, [[0.2, -0.1, 0.3], [4, 5, 6]])
    np.testing.assert_allclose(scene.entities[0].target, [[-0.1, 0.2, 0.3], [5, 4, 6]])
    assert scene.steps == 1
    assert sim._episode_steps.tolist() == [0, 1]
    sim.close()


def test_seeded_reset_reproduces_noise(native, config):
    # Given noise enabled for a generic mapped robot.
    sim = GenesisSimulator(replace(config, robot_init_qpos_noise=0.1))
    expected, _ = sim.reset(seed=42)
    # When replaying the same seed.
    actual, _ = sim.reset(seed=42)
    # Then every mapped joint has the same noisy initial state.
    np.testing.assert_array_equal(actual.qpos, expected.qpos)
    assert not np.array_equal(actual.qpos, [[0.2, -0.1, 0.3]])
    sim.close()


def test_rgbd_contains_distinct_environment_frames(native, config):
    # Given separate rendering for two environments.
    sim = GenesisSimulator(replace(config, n_envs=2, render_mode='rgb_array', obs_mode='rgbd'))
    # When requesting an observation.
    obs = sim.get_observation()
    # Then each environment retains its own image rather than a duplicated first frame.
    assert obs.rgb_image.shape == (2, 2, 3, 3)
    assert obs.rgb_image[:, 0, 0, 0].tolist() == [0, 1]
    assert obs.depth_image.shape == (2, 2, 3)
    assert native.scenes[0].options['vis_options'].split_envs
    np.testing.assert_allclose(native.scenes[0].camera_pose['lookat'], [0, 0, 1])
    sim.close()


def test_close_is_idempotent_and_reuses_runtime(native, config):
    # Given two scenes sharing a process-wide native runtime.
    first = GenesisSimulator(config)
    second = GenesisSimulator(config)
    # When closing one scene twice.
    first.close()
    first.close()
    # Then its native resources are destroyed once and the second scene remains usable.
    assert native.scenes[0].destroyed == 1
    assert native.init_count == 1
    assert not first.is_running
    np.testing.assert_allclose(second.step(np.zeros(3)).observation.qpos, [[0, 0, 0]])
    with pytest.raises(RuntimeError, match='closed'):
        first.reset()
    second.close()


def test_missing_joint_destroys_failed_construction(native, config):
    # Given an explicit mapping absent from the loaded model.
    native.names = ['different']
    # When building the adapter.
    with pytest.raises(ValueError, match='exactly one DOF'):
        GenesisSimulator(config)
    # Then scene allocation is released even though construction failed.
    assert native.scenes[0].destroyed == 1


@pytest.mark.parametrize('action', [[0, 0], [[0, 0, 0], [0, 0, 0]], [0, np.nan, 0]])
def test_invalid_action_does_not_advance_physics(native, config, action):
    # Given a built generic robot.
    sim = GenesisSimulator(config)
    # When an invalid action crosses the boundary.
    with pytest.raises(ValueError):
        sim.step(np.asarray(action))
    # Then no physics step is executed.
    assert native.scenes[0].steps == 0
    sim.close()


def test_crane_pick_place_preserves_mimic_reward_and_force_limits(native):
    # Given the original CRANE-X7 task and its independent finger joints.
    native.names = CraneX7Config.ALL_JOINT_NAMES
    sim = GenesisSimulator(
        SimulatorConfig(
            env_id='PickPlace-CRANE-X7',
            render_mode='none',
            robot_init_qpos_noise=0,
            gripper_force_limit=2,
        )
    )
    # When commanding the compact seven-arm-plus-one-gripper action.
    result = sim.step(np.arange(8) * 0.01)
    # Then both fingers follow the compact value and task metrics remain batched.
    np.testing.assert_allclose(result.observation.qpos[0, -2:], [0.07, 0.07])
    assert result.reward.shape == (1,)
    assert result.info[0]['cube_height'] == pytest.approx(0.02)
    assert result.observation.extra['env_info'][0]['height_reached'] is False
    np.testing.assert_array_equal(native.scenes[0].entities[0].force_ranges[-1][1], [2, 2])
    sim.close()


def test_mobile_arm_command_uses_one_free_articulation(native, config):
    # Given a free-base model with explicitly ordered wheel and arm joints.
    sim = GenesisSimulator(
        replace(
            config,
            joint_names=('second', 'first', 'finger'),
            velocity_joint_names=('second',),
            fixed_base=False,
        )
    )
    # When driving a wheel's velocity and an arm's position in the same step.
    result = sim.step(JointCommand(positions={'first': 0.7}, velocities={'second': 0.4}))
    # Then both joint modes work in the same articulation and URDF base stays free.
    np.testing.assert_allclose(result.observation.qpos, [[0.6, 0.7, 0.3]])
    np.testing.assert_allclose(result.observation.qvel, [[0.4, 0.8, 0]])
    assert native.morphs_added[0].fixed is False
    assert len(native.scenes[0].entities[0].names) == 3
    sim.close()


def test_invalid_named_velocity_leaves_position_target_unchanged(native, config):
    # Given a previous position command.
    sim = GenesisSimulator(config)
    before = native.scenes[0].entities[0].target.copy()
    # When a mixed command contains a valid position and an unknown velocity joint.
    with pytest.raises(ValueError, match='Unmapped'):
        sim.step(JointCommand(positions={'first': 1}, velocities={'missing': 2}))
    # Then the entire command is rejected before either control path changes state.
    np.testing.assert_array_equal(native.scenes[0].entities[0].target, before)
    assert native.scenes[0].steps == 0
    sim.close()
