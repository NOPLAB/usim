"""Bullet contracts; native tests run only in the isolated Bullet SDK environment."""

import subprocess
import sys

import pytest

from usim.capability import CapabilityRequest, Phase, UnsupportedOperation
from usim.facets import ControlTargets
from usim.model import ImportRequirements, ModelError, ModelSource
from usim.runtime import RuntimeSession
from usim.scene import SetJointState, SetPose, SpawnEntity
from usim.specs import AgentSpec, ControlSpec, EntitySpec, ExecutionSpec, Pose, SceneSpec
from usim.types import JointCommand, SimulatorConfig
from usim_bullet.backend import Backend


@pytest.fixture
def source(tmp_path):
    path = tmp_path / 'hinge.urdf'
    inertia = (
        '<inertial><mass value="1"/>'
        '<inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/></inertial>'
    )
    path.write_text(
        f'<robot name="bullet_test_hinge"><link name="base">{inertia}</link>'
        f'<link name="arm">{inertia}</link>'
        '<joint name="hinge" type="revolute"><parent link="base"/><child link="arm"/>'
        '<axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="20" velocity="10"/>'
        '</joint></robot>',
        encoding='utf-8',
    )
    return ModelSource.from_path(path)


@pytest.fixture
def native():
    return pytest.importorskip('pybullet')


def agent(source, **control):
    return AgentSpec(
        'robot',
        source,
        control=ControlSpec(**control),
        requirements=ImportRequirements(base='fixed'),
    )


def test_discovery_and_adapter_import_are_lazy():
    script = (
        'import sys; from usim_bullet import Backend, BulletSimulator; '
        'assert Backend().describe().name == "bullet"; '
        'assert "episode" in Backend().describe().execution_modes; '
        'assert "pybullet" not in sys.modules; assert "numpy" not in sys.modules'
    )
    import os

    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(sys.path)
    subprocess.run([sys.executable, '-c', script], env=env, check=True)


def test_native_formats_and_conservative_mjcf_report(source, tmp_path):
    backend = Backend()
    assert backend.models.plan(source).report.preserved >= {'joint_structure', 'inertia'}
    path = tmp_path / 'robot.xml'
    path.write_text(
        '<mujoco><worldbody><body name="base"><body name="arm">'
        '<joint name="hinge" type="hinge" axis="0 0 1"/>'
        '<geom type="sphere" size=".1" mass="1"/>'
        '</body></body></worldbody></mujoco>'
    )
    mjcf = ModelSource.from_path(path)
    assert not backend.models.assess(mjcf).supported
    plan = backend.models.plan(mjcf, ImportRequirements(preserve=frozenset({'joint_structure'})))
    assert plan.report.unknown >= {'inertia', 'collision'}


@pytest.mark.parametrize(
    'options',
    [
        {'n_envs': 2},
        {'backend': 'gpu'},
        {'mode': 'continuous'},
        {'obs_mode': 'rgb'},
        {'render_mode': 'rgb_array'},
        {'native_task': 'FakeTask'},
    ],
)
def test_unsupported_execution_is_rejected_without_native_import(source, options):
    with pytest.raises(UnsupportedOperation):
        Backend().open(SceneSpec(agents=(agent(source),)), ExecutionSpec(**options))


def test_real_control_reset_edits_and_teardown(native, source):
    import numpy as np

    session = Backend().open(
        SceneSpec(agents=(agent(source, initial_qpos=(0.2,)),)),
        ExecutionSpec(sim_rate=240, max_episode_steps=120),
    )
    client = session._require()
    state = session.state
    runtime_session = isinstance(session, RuntimeSession)
    assert runtime_session
    assert not state.schema.batched
    assert state.schema.joints[0].name == 'hinge'
    np.testing.assert_allclose(np.asarray(state.get_qpos(), dtype=float), [0.2])
    session.control.set_positions({'hinge': 0.8})
    np.testing.assert_allclose(np.asarray(state.get_qpos(), dtype=float), [0.2])
    result = session.execution.step()
    for _ in range(119):
        result = session.execution.step()
    assert result.truncated
    assert abs(state.get_qpos()[0] - 0.8) < 0.01
    assert isinstance(result.info, dict)
    assert result.info['sim_time'] == 0.5
    session.scene.apply(session.scene.plan(SetJointState('robot', (('hinge', 0.4),))))
    np.testing.assert_allclose(np.asarray(state.get_qpos(), dtype=float), [0.4])
    session.scene.apply(session.scene.plan(SetPose('robot', Pose((1, 2, 3)))))
    assert state.get_observation().extra['pose'][:3] == (1.0, 2.0, 3.0)
    session.execution.reset(seed=7)
    assert session.phase is Phase.READY
    np.testing.assert_allclose(np.asarray(state.get_qpos(), dtype=float), [0.2])
    assert state.get_observation().extra['pose'][:3] == (0.0, 0.0, 0.0)
    assert not session.scene.plan(SpawnEntity(EntitySpec('new', source))).support.supported
    assert not session.assess(CapabilityRequest('control.velocity')).supported
    session.close()
    session.close()
    assert not native.isConnected(client)
    with pytest.raises(ModelError, match='closed'):
        state.get_qpos()


@pytest.mark.parametrize('mode', ['velocity', 'effort'])
def test_native_velocity_and_persistent_torque(native, source, mode):
    control = {f'{mode}_joints': ('hinge',)}
    with Backend().open(
        SceneSpec(agents=(agent(source, **control),)), ExecutionSpec(sim_rate=240)
    ) as session:
        targets = {'velocities' if mode == 'velocity' else 'efforts': {'hinge': 1.0}}
        session.control.apply(ControlTargets(**targets))
        assert session.state.get_qvel()[0] == 0
        session.execution.step()
        first = session.state.get_qvel()[0]
        session.execution.step()
        second = session.state.get_qvel()[0]
        assert first > 0
        assert second > first if mode == 'effort' else second == pytest.approx(1)
        session.execution.reset()
        session.execution.step()
        assert session.state.get_qvel()[0] == pytest.approx(0)


def test_validation_is_atomic_and_clients_are_isolated(native, source):
    with (
        Backend().open(SceneSpec(agents=(agent(source),))) as first,
        Backend().open(SceneSpec(agents=(agent(source),))) as second,
    ):
        assert first._require() != second._require()
        with pytest.raises(ModelError):
            first.control.apply(
                ControlTargets(
                    positions={'hinge': 0.5},
                    velocities={'unknown': 1.0},
                )
            )
        assert first.phase is Phase.READY
        for _ in range(10):
            first.execution.step()
        assert first.state.get_qpos()[0] == pytest.approx(0)
        assert second.state.get_qpos()[0] == 0
        first.close()
        second.execution.step()


def test_native_failure_disconnects_client(native, source, monkeypatch):
    session = Backend().open(SceneSpec(agents=(agent(source),)))
    client = session._require()

    def fail(**kwargs):
        raise RuntimeError('native step failed')

    monkeypatch.setattr(native, 'stepSimulation', fail)
    with pytest.raises(RuntimeError, match='native step failed'):
        session.execution.step()
    assert session.phase is Phase.FAILED
    assert not native.isConnected(client)


def test_native_mjcf_load_and_step(native, tmp_path):
    path = tmp_path / 'robot.xml'
    path.write_text(
        '<mujoco><worldbody><body name="base"><geom type="sphere" size=".1" mass="1"/>'
        '<body name="arm" pos="0 0 .2"><joint name="hinge" type="hinge" axis="0 0 1"/>'
        '<geom type="sphere" size=".1" mass="1"/></body></body></worldbody></mujoco>'
    )
    spec = AgentSpec(
        'robot',
        ModelSource.from_path(path),
        requirements=ImportRequirements(preserve=frozenset({'joint_structure'})),
        control=ControlSpec(joint_names=('hinge',), joint_force_limits=(10,)),
    )
    with Backend().open(SceneSpec(agents=(spec,))) as session:
        session.control.set_positions({'hinge': 0.5})
        for _ in range(20):
            session.execution.step()
        assert session.state.get_qpos()[0] > 0


def test_episode_adapter_uses_same_native_world(native, source):
    from usim_bullet import BulletSimulator

    config = SimulatorConfig(
        'JointControl',
        robot_path=str(source.path),
        render_mode='none',
        obs_mode='state',
        robot_init_qpos_noise=0,
        joint_names=('hinge',),
        max_episode_steps=1,
    )
    simulator = BulletSimulator(config)
    assert simulator.joint_names == ['hinge']
    assert simulator.step(JointCommand(positions={'hinge': 0.4})).truncated
    assert simulator.get_qpos()[0] > 0
    simulator.reset()
    assert simulator.get_qpos()[0] == 0
    simulator.close()
    simulator.close()
    assert not simulator.is_running


def test_native_cli_model_route(native, source):
    import json
    import os

    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(sys.path)
    result = subprocess.run(
        [
            sys.executable,
            '-m',
            'usim.cli',
            'run',
            '--engine',
            'bullet',
            '--robot',
            str(source.path),
            '--joint-names',
            'hinge',
            '--render-mode',
            'none',
            '--obs-mode',
            'state',
            '--steps',
            '20',
            '--action',
            '0.3',
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(result.stdout.strip().splitlines()[-1])
    assert summary == {'engine': 'bullet', 'steps': 20, 'qpos_shape': [1], 'rgb_shape': []}
