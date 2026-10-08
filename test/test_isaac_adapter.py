"""Deterministic Isaac adapter fixtures; these do not claim GPU physics validation."""

import sys
import tomllib
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from usim.types import JointCommand, SimulatorConfig
import usim_isaacsim.adapter as adapter
from usim_isaacsim.adapter import IsaacSimSimulator
from usim.robots.crane_x7 import CraneX7Config


@pytest.fixture
def native(monkeypatch):
    """Preserve articulation indexing, native stepping, and lifetime boundaries."""
    state = SimpleNamespace(
        events=[],
        imports=[],
        apps=[],
        worlds=[],
        robots=[],
        cameras=[],
        names=list(reversed(CraneX7Config.ALL_JOINT_NAMES)),
        roots=1,
        missing_output=False,
        plugins=[],
    )

    def module(name, **attributes):
        result = ModuleType(name)
        result.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, result)
        if '.' in name:
            parent, child = name.rsplit('.', 1)
            if parent not in sys.modules:
                module(parent)
            monkeypatch.setattr(sys.modules[parent], child, result, raising=False)
        return result

    class App:
        def __init__(self, settings, *, experience=''):
            state.events.append('app')
            self.settings, self.closed = settings, False
            self.experience = experience
            state.apps.append(self)

        def is_running(self):
            return not self.closed

        def close(self):
            if self.settings.get('fast_shutdown', True):
                raise SystemExit(0)
            self.closed = True
            state.events.append('close')

    class Prim:
        def __init__(self, path, root=False):
            self.path, self.root = path, root

        def GetPath(self):
            return self.path

        def GetName(self):
            return self.path.rsplit('/', 1)[-1]

        def HasAPI(self, api):
            return self.root

        def GetVariantSet(self, name):
            return SimpleNamespace(SetVariantSelection=lambda value: state.events.append(value))

    links = [
        'crane_x7_gripper_base_link',
        'crane_x7_gripper_finger_a_link',
        'crane_x7_gripper_finger_b_link',
    ]

    class Stage:
        def Traverse(self):
            return [
                *[Prim(f'/World/Robot/root{i}', True) for i in range(state.roots)],
                *[Prim('/World/Robot/' + name) for name in links],
            ]

        def GetPrimAtPath(self, path):
            return Prim(path)

        def Load(self):
            state.events.append('load')

    stage = Stage()

    class Robot:
        def __init__(self, prim_path, name):
            self.dof_names = list(state.names)
            self.qpos = np.zeros(len(self.dof_names))
            self.qvel = self.qpos.copy()
            self.targets = self.qpos.copy()
            self.velocity_targets = self.qpos.copy()
            self.modes = np.full(len(self.dof_names), 'position', dtype=object)
            self.kp = self.qpos.copy()
            self.kd = self.qpos.copy()
            state.robots.append(self)

        def get_dof_index(self, name):
            return self.dof_names.index(name)

        @property
        def dof_properties(self):
            properties = np.zeros(len(self.dof_names), dtype=[('lower', float), ('upper', float)])
            properties['lower'] = -3.0
            properties['upper'] = 3.0
            return properties

        def get_articulation_controller(self):
            return Controller(self)

        def set_joint_positions(self, positions, *, joint_indices):
            self.qpos[joint_indices] = positions

        def set_joint_velocities(self, velocities, *, joint_indices):
            self.qvel[joint_indices] = velocities

        def get_joint_positions(self, *, joint_indices):
            return self.qpos[joint_indices]

        def get_joint_velocities(self, *, joint_indices):
            return self.qvel[joint_indices]

        def apply_action(self, action):
            if hasattr(action, 'joint_positions'):
                self.targets[action.joint_indices] = action.joint_positions
            if hasattr(action, 'joint_velocities'):
                self.velocity_targets[action.joint_indices] = action.joint_velocities

    class Controller:
        def __init__(self, robot):
            self.robot = robot

        def get_gains(self):
            return self.robot.kp, self.robot.kd

        def set_gains(self, *, kps, kds):
            self.robot.kp, self.robot.kd = kps.copy(), kds.copy()

        def set_max_efforts(self, efforts, *, joint_indices):
            self.robot.efforts = np.zeros(len(self.robot.dof_names))
            self.robot.efforts[joint_indices] = efforts

        def switch_control_mode(self, mode):
            assert mode == 'position'
            self.robot.modes[:] = mode

        def switch_dof_control_mode(self, *, dof_index, mode):
            self.robot.modes[dof_index] = mode

    class Body:
        def __init__(self, prim_path, name, position=None, **kwargs):
            self.position = np.array(position if position is not None else [0.15, 0.02, 0.08])
            self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])

        def get_world_pose(self):
            return self.position.copy(), self.quaternion.copy()

        def set_world_pose(self, *, position, orientation):
            self.position, self.quaternion = position.copy(), orientation.copy()

        def set_linear_velocity(self, velocity):
            self.linear_velocity = velocity.copy()

        def set_angular_velocity(self, velocity):
            self.angular_velocity = velocity.copy()

    class World:
        def __init__(self, **settings):
            assert state.events[0] == 'app'
            self.settings, self.steps, self.resets = settings, 0, 0
            self.instance_cleared = False
            self.scene = SimpleNamespace(
                add=lambda item: item, add_default_ground_plane=lambda: None
            )
            state.worlds.append(self)

        def reset(self):
            self.resets += 1
            for robot in state.robots:
                robot.qpos[:] = 0
                robot.qvel[:] = 0

        def step(self, *, render):
            self.steps += 1
            for robot in state.robots:
                robot.qvel[:] = (robot.targets - robot.qpos) / self.settings['physics_dt']
                velocity = robot.modes == 'velocity'
                robot.qvel[velocity] = robot.velocity_targets[velocity]
                robot.qpos += robot.qvel * self.settings['physics_dt']

        def stop(self):
            state.events.append('stop')

        def clear_instance(self):
            self.instance_cleared = True
            state.events.append('clear')

    class ImporterConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Importer:
        def __init__(self, config):
            self.config = config

        def convert(self):
            assert state.events[0] == 'app'
            state.imports.append(self.config)
            path = Path(self.config.usd_path) / 'robot.usd'
            if not state.missing_output:
                path.write_text('#usda 1.0\n', encoding='utf-8')
            return str(path)

        import_mjcf = import_urdf = convert

    class Camera:
        def __init__(self, path, *, resolution, annotators):
            self.path = path
            self.authoring_object = SimpleNamespace(
                set_world_poses=lambda **kwargs: None, set_local_poses=lambda **kwargs: None
            )
            state.cameras.append(self)

        def get_data(self, name):
            if name == 'rgb':
                return np.full((480, 640, 4), 23, dtype=np.uint8), {}
            return np.ones((480, 640), dtype=np.float32), {}

    def capture(**kwargs):
        assert kwargs['delta_time'] == 0.0
        state.events.append('capture')

    module('isaacsim', SimulationApp=App)
    module(
        'carb',
        get_framework=lambda: SimpleNamespace(get_plugins=lambda: state.plugins),
    )
    module('isaacsim.core.api', World=World)
    module('isaacsim.core.api.objects', DynamicCuboid=Body)
    module('isaacsim.core.prims', SingleArticulation=Robot, SingleRigidPrim=Body)
    module('isaacsim.core.utils.stage', add_reference_to_stage=lambda *args: None)
    module('isaacsim.core.utils.types', ArticulationAction=SimpleNamespace)
    module('isaacsim.core.utils.extensions', enable_extension=lambda name: None)
    module('isaacsim.core.experimental.utils.app', play=lambda **kwargs: None)
    module(
        'isaacsim.asset.importer.mjcf', MJCFImporter=Importer, MJCFImporterConfig=ImporterConfig
    )
    module(
        'isaacsim.asset.importer.urdf', URDFImporter=Importer, URDFImporterConfig=ImporterConfig
    )
    module('isaacsim.sensors.experimental.rtx', CameraSensor=Camera)
    module('omni.usd', get_context=lambda: SimpleNamespace(get_stage=lambda: stage))
    module('omni.replicator.core', orchestrator=SimpleNamespace(step=capture))
    module(
        'pxr',
        Usd=SimpleNamespace(Stage=SimpleNamespace(Open=lambda path: stage)),
        UsdGeom=SimpleNamespace(
            GetStageUpAxis=lambda stage: 'z',
            GetStageMetersPerUnit=lambda stage: 1.0,
            Tokens=SimpleNamespace(z='z'),
            Camera=SimpleNamespace(Define=lambda *args: None),
        ),
        UsdPhysics=SimpleNamespace(ArticulationRootAPI=object()),
        UsdLux=SimpleNamespace(
            DistantLight=SimpleNamespace(
                Define=lambda *args: SimpleNamespace(CreateIntensityAttr=lambda value: None)
            )
        ),
    )
    return state


def config(**kwargs):
    return SimulatorConfig(
        env_id=kwargs.pop('env_id', 'JointControl'),
        obs_mode=kwargs.pop('obs_mode', 'state'),
        render_mode=kwargs.pop('render_mode', 'none'),
        robot_init_qpos_noise=kwargs.pop('robot_init_qpos_noise', 0),
        **kwargs,
    )


def test_native_joint_mapping_targets_step_observation_and_close(native):
    sim = IsaacSimSimulator(config(sim_rate=60, max_episode_steps=2))
    robot, world = native.robots[0], native.worlds[0]
    np.testing.assert_allclose(sim.get_qpos(), CraneX7Config.REST_QPOS)
    assert sim.all_joint_names == CraneX7Config.ALL_JOINT_NAMES
    np.testing.assert_array_equal(sim._indices, np.arange(8, -1, -1))
    assert np.all(robot.kp == 1000) and np.all(robot.kd == 100)
    assert world.steps == 0
    action = np.arange(8) / 10
    result = sim.step(action)
    np.testing.assert_allclose(result.observation.qpos, np.r_[action, action[-1]])
    np.testing.assert_allclose(
        result.observation.qvel, (np.r_[action, action[-1]] - CraneX7Config.REST_QPOS) * 60
    )
    assert world.steps == 1 and result.reward == 0 and not result.terminated
    assert not result.truncated and result.info['task_reward'] is False
    result.observation.qpos[:] = 999
    assert sim.get_qpos()[0] == 0
    assert sim.step(np.full(9, 9)).truncated
    np.testing.assert_allclose(sim.get_qpos(), 3)
    output = Path(native.imports[0].usd_path)
    sim.close()
    sim.close()
    assert not output.exists() and native.apps[0].closed and not sim.is_running
    assert native.events[-3:] == ['stop', 'clear', 'close']
    with pytest.raises(RuntimeError, match='closed'):
        sim.get_qpos()


def test_seeded_reset_noise_velocity_and_targets(native):
    sim = IsaacSimSimulator(config(robot_init_qpos_noise=0.02))
    first, _ = sim.reset(seed=42)
    sim.step(np.zeros(9))
    second, _ = sim.reset(seed=42)
    np.testing.assert_array_equal(first.qpos, second.qpos)
    assert not np.array_equal(first.qpos, CraneX7Config.REST_QPOS)
    np.testing.assert_array_equal(second.qvel, np.zeros(9))
    np.testing.assert_array_equal(native.robots[0].targets[sim._indices], second.qpos)
    assert native.worlds[0].steps == 1
    sim.close()


def test_shutdown_failure_still_releases_app_singleton_and_imports(native, monkeypatch):
    sim = IsaacSimSimulator(config())
    directory = Path(native.imports[0].usd_path)

    def fail_stop():
        raise RuntimeError('native stop failed')

    monkeypatch.setattr(native.worlds[0], 'stop', fail_stop)
    with pytest.raises(RuntimeError, match='native stop failed'):
        sim.close()
    assert native.apps[0].closed and 'clear' in native.events
    assert not directory.exists() and not sim.is_running
    sim.close()


def test_native_resources_released_before_app_shutdown(native, monkeypatch):
    sim = IsaacSimSimulator(config())
    close_app = native.apps[0].close

    def close_without_native_references():
        assert all(
            vars(sim)[name] is None for name in ('_world', '_robot', '_stage', '_camera', '_cube')
        )
        assert vars(sim)['_fingers'] == []
        assert native.worlds[0].instance_cleared
        close_app()

    monkeypatch.setattr(native.apps[0], 'close', close_without_native_references)
    sim.close()


@pytest.mark.parametrize(
    ('obs_mode', 'render_mode', 'minimal'),
    [('state', 'none', True), ('rgb', 'none', False), ('state', 'human', False)],
)
def test_headless_state_uses_native_physics_experience(native, obs_mode, render_mode, minimal):
    sim = IsaacSimSimulator(config(obs_mode=obs_mode, render_mode=render_mode))
    experience = native.apps[0].experience
    if minimal:
        settings = tomllib.loads(Path(experience).read_text())
        assert set(settings['dependencies']) == {
            'omni.isaac.ml_archive',
            'omni.kit.renderer.core',
            'isaacsim.core.api',
            'isaacsim.asset.importer.urdf',
            'omni.kit.loop-isaac',
        }
    else:
        assert experience == ''
    sim.close()


def test_linux_shutdown_keeps_only_mapped_provider_code_until_exit(native, monkeypatch, tmp_path):
    sim = IsaacSimSimulator(config())
    loaded = str((tmp_path / 'loaded.plugin.so').resolve())
    unloaded = str((tmp_path / 'unloaded.plugin.so').resolve())
    native.plugins = [
        SimpleNamespace(libPath=loaded),
        SimpleNamespace(libPath=unloaded),
        SimpleNamespace(libPath=''),
    ]
    monkeypatch.setattr(adapter.sys, 'platform', 'linux')
    for name, value in (
        ('RTLD_NOW', 2),
        ('RTLD_LOCAL', 0),
        ('RTLD_NOLOAD', 4),
        ('RTLD_NODELETE', 4096),
    ):
        monkeypatch.setattr(adapter.os, name, value, raising=False)

    def maps(path):
        assert path == Path('/proc/self/maps')
        return f'1000-2000 r-xp 0000 00:01 1 {loaded}\n2000-3000 rw-p 0000 00:00 0\n'

    monkeypatch.setattr(Path, 'read_text', maps)
    retained = []

    def keep_provider(path, *, mode):
        retained.append((path, mode))
        native.events.append('retain')

    monkeypatch.setattr(adapter.ctypes, 'CDLL', keep_provider)
    sim.close()
    sim.close()
    assert retained == [(loaded, 2 | 4 | 4096)]
    assert native.events[-4:] == ['stop', 'clear', 'retain', 'close']
    assert native.apps[0].closed and not sim.is_running


def test_provider_retention_failure_is_not_reported_as_success(native, monkeypatch, tmp_path):
    sim = IsaacSimSimulator(config())
    loaded = str((tmp_path / 'loaded.plugin.so').resolve())
    native.plugins = [SimpleNamespace(libPath=loaded)]
    monkeypatch.setattr(adapter.sys, 'platform', 'linux')
    for name in ('RTLD_NOW', 'RTLD_LOCAL', 'RTLD_NOLOAD', 'RTLD_NODELETE'):
        monkeypatch.setattr(adapter.os, name, 0, raising=False)
    monkeypatch.setattr(Path, 'read_text', lambda path: f'1000-2000 r-xp 0000 00:01 1 {loaded}\n')

    def fail_retention(*args, **kwargs):
        raise OSError('provider could not be retained')

    monkeypatch.setattr(adapter.ctypes, 'CDLL', fail_retention)
    with pytest.raises(OSError, match='provider could not be retained'):
        sim.close()
    assert not native.apps[0].closed
    monkeypatch.setattr(adapter.ctypes, 'CDLL', lambda *args, **kwargs: None)
    sim.close()
    assert native.apps[0].closed


@pytest.mark.parametrize('suffix', ['.xml', '.urdf', '.usd'])
def test_external_robot_explicit_mapping_and_import(native, tmp_path, suffix):
    native.names = ['wrist', 'unused', 'shoulder']
    path = tmp_path / ('custom' + suffix)
    path.write_text('fixture', encoding='utf-8')
    sim = IsaacSimSimulator(
        config(
            robot_path=str(path),
            robot_uid='custom',
            arm_joint_names=('shoulder', 'wrist'),
            initial_qpos=(0.1, 0.2),
        )
    )
    np.testing.assert_allclose(sim.get_qpos(), [0.1, 0.2])
    result = sim.step(np.array([0.4, -0.2]))
    np.testing.assert_allclose(result.observation.qpos, [0.4, -0.2])
    assert native.robots[0].qpos[1] == 0
    if suffix != '.usd':
        assert native.imports[0].fix_base
        assert native.imports[0].joint_target_type == 'position'
    else:
        assert not native.imports
    sim.close()
    assert path.exists()


def test_external_joint_discovery(native, tmp_path):
    native.names = ['second', 'first']
    path = tmp_path / 'robot.usd'
    path.write_text('fixture', encoding='utf-8')
    sim = IsaacSimSimulator(config(robot_path=str(path), robot_uid='custom'))
    assert sim.arm_joint_names == native.names and sim.gripper_joint_names == []
    assert sim.joint_names == native.names and sim.all_joint_names == sim.joint_names
    sim.close()


@pytest.mark.parametrize('suffix', ['.xml', '.urdf'])
@pytest.mark.parametrize('fixed_base', [True, False])
def test_import_respects_fixed_and_free_base(native, tmp_path, suffix, fixed_base):
    native.names = ['wheel', 'shoulder']
    path = tmp_path / ('combined' + suffix)
    path.write_text('fixture', encoding='utf-8')
    sim = IsaacSimSimulator(
        config(
            robot_path=str(path),
            robot_uid='combined',
            fixed_base=fixed_base,
            joint_names=('shoulder', 'wheel'),
        )
    )
    assert native.imports[0].fix_base is fixed_base
    assert len(native.robots) == 1
    assert sim.joint_names == ['shoulder', 'wheel']
    assert sim.all_joint_names == ['shoulder', 'wheel']
    sim.close()


def test_one_free_articulation_accepts_mixed_named_commands(native, tmp_path):
    native.names = ['right_wheel', 'unused', 'shoulder', 'left_wheel']
    path = tmp_path / 'combined.urdf'
    path.write_text('fixture', encoding='utf-8')
    sim = IsaacSimSimulator(
        config(
            robot_path=str(path),
            robot_uid='combined',
            fixed_base=False,
            sim_rate=10,
            joint_names=('left_wheel', 'shoulder', 'right_wheel'),
            velocity_joint_names=('left_wheel', 'right_wheel'),
            initial_qpos=(0.0, 0.1, 0.0),
        )
    )
    result = sim.step(
        JointCommand(
            positions={'shoulder': 0.6}, velocities={'left_wheel': 2.0, 'right_wheel': -1.0}
        )
    )
    assert len(native.robots) == 1 and native.worlds[0].steps == 1
    np.testing.assert_allclose(result.observation.qpos, [0.2, 0.6, -0.1])
    np.testing.assert_allclose(result.observation.qvel, [2.0, 5.0, -1.0])
    robot = native.robots[0]
    np.testing.assert_array_equal(robot.modes, ['velocity', 'position', 'position', 'velocity'])
    np.testing.assert_allclose(robot.kp, [0, 0, 1000, 0])
    # Unspecified native DOFs are retained; named commands don't address them.
    assert robot.qpos[1] == 0
    result = sim.step(JointCommand(velocities={'left_wheel': 0.0, 'right_wheel': 0.0}))
    np.testing.assert_allclose(result.observation.qvel, [0, 0, 0])
    sim.reset(seed=0)
    np.testing.assert_allclose(sim.get_qpos(), [0, 0.1, 0])
    np.testing.assert_allclose(sim.get_qvel(), [0, 0, 0])
    np.testing.assert_allclose(robot.velocity_targets, 0)
    with pytest.raises(ValueError, match='Use JointCommand'):
        sim.step(np.zeros(3))
    before = native.worlds[0].steps
    with pytest.raises(ValueError, match='Position command'):
        sim.step(JointCommand(positions={'left_wheel': 1.0}))
    with pytest.raises(ValueError, match='Velocity command'):
        sim.step(JointCommand(velocities={'shoulder': 1.0}))
    with pytest.raises(ValueError, match='Position command'):
        sim.step(JointCommand(positions={'missing': 1.0}))
    original_targets = robot.targets.copy()
    with pytest.raises(ValueError, match='finite'):
        sim.step(JointCommand(positions={'shoulder': 1.0}, velocities={'left_wheel': np.nan}))
    np.testing.assert_array_equal(robot.targets, original_targets)
    with pytest.raises(ValueError, match='one target'):
        sim.step(JointCommand(velocities={'left_wheel': np.array([1.0, 2.0])}))
    assert native.worlds[0].steps == before
    sim.close()


def test_named_position_command_preserves_other_targets(native):
    sim = IsaacSimSimulator(config())
    name = CraneX7Config.ARM_JOINT_NAMES[0]
    result = sim.step(JointCommand(positions={name: np.array([0.25])}))
    expected = CraneX7Config.REST_QPOS.copy()
    expected[0] = 0.25
    np.testing.assert_allclose(result.observation.qpos, expected)
    sim.close()


def test_pick_place_physical_reset_metrics_and_reward(native):
    sim = IsaacSimSimulator(config(env_id='PickPlace-CRANE-X7'))
    sim.reset(seed=7)
    first = sim._cube.position.copy()
    sim.reset(seed=7)
    np.testing.assert_array_equal(first, sim._cube.position)
    assert np.all(sim._cube.linear_velocity == 0) and np.all(sim._cube.angular_velocity == 0)
    sim._cube.position = np.array([0.15, 0.02, 0.14])
    success = sim.step(CraneX7Config.REST_QPOS)
    assert success.reward == 5 and success.terminated and success.info['success']
    sim._cube.position = np.array([0.15, 0.02, 0.02])
    failure = sim.step(CraneX7Config.REST_QPOS)
    distance = failure.info['gripper_to_cube_dist']
    assert failure.reward == pytest.approx(1 - np.tanh(5 * distance) + np.exp(-10 * distance))
    assert not failure.terminated
    sim.close()


def test_rgbd_camera_captures_without_extra_physics(native):
    sim = IsaacSimSimulator(
        config(obs_mode='rgbd', render_mode='rgb_array', camera_uid='scene_camera')
    )
    observation = sim.step(np.zeros(9)).observation
    assert observation.rgb_image.shape == (480, 640, 3)
    assert observation.rgb_image.dtype == np.uint8
    assert np.all(observation.rgb_image == 23)
    assert np.all(observation.depth_image == 1)
    assert native.worlds[0].steps == 1
    sim.get_observation()
    assert native.worlds[0].steps == 1
    sim.close()


@pytest.mark.parametrize(
    'change,error',
    [
        ({'env_id': 'unknown'}, 'Unsupported Isaac environment'),
        ({'n_envs': 2}, 'n_envs=1'),
        ({'control_mode': 'velocity'}, 'pd_joint_pos'),
        ({'obs_mode': 'invalid'}, 'obs_mode'),
        ({'robot_uid': 'custom'}, 'robot_path'),
    ],
)
def test_unsupported_choices_fail_before_starting_app(native, change, error):
    with pytest.raises(ValueError, match=error):
        IsaacSimSimulator(config(**change))
    assert native.apps == []


@pytest.mark.parametrize('fault', ['joint', 'roots', 'output', 'camera', 'initial'])
def test_initialization_errors_release_app_and_imports(native, fault):
    options = {}
    if fault == 'joint':
        native.names.pop()
    elif fault == 'roots':
        native.roots = 0
    elif fault == 'output':
        native.missing_output = True
    elif fault == 'camera':
        options.update(obs_mode='rgb', render_mode='rgb_array', camera_uid='invalid')
    else:
        options['initial_qpos'] = tuple([10.0] * 9)
    with pytest.raises((ValueError, RuntimeError)):
        IsaacSimSimulator(config(**options))
    assert native.apps[0].closed
    assert all(not Path(item.usd_path).exists() for item in native.imports)


@pytest.mark.parametrize(
    'action', [np.zeros(7), np.zeros((1, 9)), np.full(9, np.nan), np.full(9, np.inf)]
)
def test_bad_action_does_not_advance(native, action):
    sim = IsaacSimSimulator(config())
    with pytest.raises(ValueError, match='finite joint targets'):
        sim.step(action)
    assert native.worlds[0].steps == 0
    sim.close()
