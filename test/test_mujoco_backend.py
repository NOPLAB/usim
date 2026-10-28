"""Portable contracts at the MuJoCo native boundary, without GPU computation."""

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from usim.capability import CapabilityRequest, Phase, SupportStatus, UnsupportedOperation
from usim.facets import ControlTargets
from usim.model import ImportRequirements, ModelError, ModelSource
from usim.scene import SetJointState
from usim.specs import AgentSpec, ControlSpec, ExecutionSpec, SceneSpec
from usim.types import JointCommand, SimulatorConfig
from usim_mujoco import Backend, MuJoCoSimulator


@pytest.fixture
def scene(tmp_path):
    path = tmp_path / 'model.xml'
    path.write_text(
        '<mujoco><worldbody><body name="link"><joint name="hinge"/>'
        '<geom size="0.1"/></body></worldbody></mujoco>'
    )
    return SceneSpec(
        agents=(
            AgentSpec(
                'robot',
                ModelSource.from_path(path),
                ControlSpec(joint_names=('hinge',), joint_force_limits=(2.0,)),
                requirements=ImportRequirements(base='fixed'),
            ),
        )
    )


@pytest.fixture
def native(monkeypatch):
    """Stateful native boundary preserves control/state/time, not native physics."""
    model = SimpleNamespace(
        opt=SimpleNamespace(timestep=0.002),
        njnt=1,
        nq=1,
        nv=1,
        nu=1,
        jnt_type=np.array([3]),
        jnt_qposadr=np.array([0]),
        jnt_dofadr=np.array([0]),
    )
    calls = SimpleNamespace(cpu_steps=0, gpu_steps=0, uploads=0, downloads=0)

    def data(model):
        return SimpleNamespace(
            qpos=np.zeros(1),
            qvel=np.zeros(1),
            ctrl=np.zeros(1),
            qfrc_applied=np.zeros(1),
            qfrc_actuator=np.zeros(1),
            time=0.0,
        )

    def reset(model, state):
        state.qpos[:] = state.qvel[:] = state.ctrl[:] = state.qfrc_applied[:] = 0
        state.time = 0.0

    def step(model, state):
        calls.cpu_steps += 1
        state.qvel[:] += (state.qfrc_applied + state.qfrc_actuator) * model.opt.timestep
        state.qpos[:] += state.qvel * model.opt.timestep
        state.time += model.opt.timestep

    sdk = SimpleNamespace(
        MjModel=SimpleNamespace(from_xml_path=lambda path: model),
        MjData=data,
        mj_resetData=reset,
        mj_forward=lambda *args: None,
        mj_step=step,
        mj_id2name=lambda *args: 'hinge',
        mjtObj=SimpleNamespace(mjOBJ_JOINT=3),
        mjtJoint=SimpleNamespace(mjJNT_FREE=0),
    )
    monkeypatch.setitem(sys.modules, 'mujoco', sdk)
    return calls, model, sdk


def test_discovery_does_not_import_numpy_or_native_sdks():
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join([str(root / 'src'), str(root / 'packages/mujoco/src')])
    result = subprocess.run(
        [
            sys.executable,
            '-c',
            'import sys; from usim_mujoco import Backend, WarpBackend; '
            "assert Backend().describe().name == 'mujoco'; "
            "assert WarpBackend().describe().name == 'mujoco-warp'; "
            "assert not {'numpy', 'mujoco', 'mujoco_warp', 'warp'} & sys.modules.keys()",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


def test_grouped_controls_do_not_step_and_native_state_is_copied(native, scene):
    calls, model, _ = native
    with Backend().open(scene, ExecutionSpec(max_episode_steps=1)) as session:
        session.control.for_entity('robot').apply(ControlTargets(positions={'hinge': 0.3}))
        assert calls.cpu_steps == 0
        result = session.execution.step()
        assert calls.cpu_steps == session.substeps
        assert result.observation.qpos is not None
        assert result.observation.qpos[0] > 0
        assert result.observation.extra['time'] == pytest.approx(1 / 30)
        assert abs(session.data.qfrc_applied[0]) <= 2.0
        assert model.opt.timestep <= 0.002
        assert result.truncated
        assert not result.terminated
        assert session.phase is Phase.PAUSED
        assert session.state.schema.batched is False
        session.state.get_qpos()[0] = -999
        assert session.state.get_qpos()[0] != -999
    assert session.phase is Phase.CLOSED
    session.close()
    with pytest.raises(ModelError, match='closed'):
        session.state.get_qpos()


def test_invalid_batch_is_atomic_and_reset_restores_initial_state(native, scene):
    control = replace(scene.agents[0].control, initial_qpos=(0.2,))
    scene = replace(scene, agents=(replace(scene.agents[0], control=control),))
    with Backend().open(scene) as session:
        session.control.set_efforts({'hinge': 0.5})
        previous = session.targets.copy()
        with pytest.raises(ModelError):
            session.control.apply(ControlTargets(positions={'hinge': [0.1, 0.2]}))
        assert session.targets == previous
        session.step()
        observation, _ = session.reset(seed=42)
        assert observation.qpos is not None
        assert observation.qvel is not None
        np.testing.assert_array_equal(observation.qpos, [0.2])
        np.testing.assert_array_equal(observation.qvel, [0])
        assert session.targets == {}
        assert session.steps == 0


def test_state_edits_do_not_advance_and_unknown_operations_are_rejected(native, scene):
    calls, _, _ = native
    with Backend().open(scene) as session:
        edit = SetJointState('robot', positions=(('hinge', 0.4),), velocities=(('hinge', 0.2),))
        session.scene.apply(session.scene.plan(edit))
        np.testing.assert_array_equal(session.state.get_qpos(), [0.4])
        np.testing.assert_array_equal(session.state.get_qvel(), [0.2])
        assert calls.cpu_steps == 0
        assert session.assess(CapabilityRequest('scene.spawn')).status is SupportStatus.UNSUPPORTED
        assert session.assess(CapabilityRequest('sensors.rgb')).status is SupportStatus.UNSUPPORTED


def test_native_step_failure_closes_owned_resources(native, scene):
    _, _, sdk = native
    with Backend().open(scene) as session:
        owned = session.owned
        assert owned is not None

        def fail(*args):
            raise RuntimeError('native failure')

        sdk.mj_step = fail
        with pytest.raises(RuntimeError, match='native failure'):
            session.step()
        assert session.phase is Phase.FAILED
        assert owned.closed
        assert session.model is session.data is None


def test_episode_actions_validate_shape_and_keep_joint_order(native, scene):
    config = SimulatorConfig(
        env_id='JointControl',
        robot_path=str(scene.agents[0].model.path),
        render_mode='none',
        obs_mode='state',
        robot_init_qpos_noise=0,
        joint_names=('hinge',),
    )
    simulator = MuJoCoSimulator(config)
    try:
        assert simulator.joint_names == ['hinge']
        with pytest.raises(ModelError, match='shape'):
            simulator.step(np.array([1.0, 2.0]))
        simulator.step(JointCommand(velocities={'hinge': 0.2}))
        assert simulator.get_qvel()[0] > 0
        simulator.reset()
        np.testing.assert_array_equal(simulator.get_qpos(), [0])
    finally:
        simulator.close()
    simulator.close()
    assert not simulator.is_running


@pytest.mark.parametrize(
    'options',
    [
        ExecutionSpec(n_envs=2),
        ExecutionSpec(render_mode='rgb_array'),
        ExecutionSpec(mode='continuous'),
    ],
)
def test_unsupported_execution_is_rejected_before_native_import(scene, options):
    with pytest.raises(UnsupportedOperation):
        Backend().open(scene, options)


def test_base_mode_is_checked_against_native_model(native, scene):
    floating = replace(
        scene, agents=(replace(scene.agents[0], requirements=ImportRequirements(base='floating')),)
    )
    with pytest.raises(ModelError, match='base mode'):
        Backend().open(floating)
