"""Opt-in real CPU integration checks; never initialize Warp or GPU computation.

Run in the native Python 3.12 environment with USIM_MUJOCO_NATIVE_TESTS=1.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from usim.facets import ControlTargets
from usim.model import ModelSource
from usim.specs import AgentSpec, ControlSpec, ExecutionSpec, SceneSpec
from usim.types import SimulatorConfig
from usim_mujoco import Backend, MuJoCoSimulator

pytestmark = pytest.mark.skipif(
    os.environ.get('USIM_MUJOCO_NATIVE_TESTS') != '1',
    reason='Native CPU checks require explicit opt-in in the engine environment',
)


@pytest.fixture
def mjcf(tmp_path):
    pytest.importorskip('mujoco')
    path = tmp_path / 'hinge.xml'
    path.write_text(
        '<mujoco><option gravity="0 0 0" timestep="0.002"/>'
        '<worldbody><body name="arm"><joint name="hinge" damping="0.1"/>'
        '<geom type="capsule" size="0.05" fromto="0 0 0 0 0 1" mass="1"/>'
        '</body></worldbody><actuator><motor joint="hinge"/></actuator></mujoco>'
    )
    return path


def test_real_cpu_position_control_reset_and_native_actuator(mjcf):
    scene = SceneSpec(
        agents=(
            AgentSpec(
                'robot',
                ModelSource.from_path(mjcf),
                ControlSpec(joint_names=('hinge',), joint_force_limits=(1.0,)),
            ),
        )
    )
    with Backend().open(scene, ExecutionSpec(sim_rate=100, max_episode_steps=100)) as session:
        session.control.apply(ControlTargets(positions={'hinge': 0.3}))
        result = session.step()
        for _ in range(99):
            result = session.step()
        assert result.observation.qpos is not None
        assert np.isfinite(np.asarray(result.observation.qpos, dtype=float)).all()
        assert result.observation.qpos[0] > 0.01
        assert result.observation.extra['time'] == pytest.approx(1.0)
        observation, _ = session.reset()
        assert observation.qpos is not None
        assert observation.qvel is not None
        np.testing.assert_array_equal(observation.qpos, [0])
        np.testing.assert_array_equal(observation.qvel, [0])
    config = SimulatorConfig(
        env_id='JointControl',
        robot_path=str(mjcf),
        robot_init_qpos_noise=0,
        joint_names=('hinge',),
        render_mode='none',
        obs_mode='state',
        control_mode='native',
    )
    simulator = MuJoCoSimulator(config)
    try:
        observation = simulator.step(np.array([0.1])).observation
        assert observation.qvel is not None
        assert observation.qvel[0] > 0
    finally:
        simulator.close()


def test_physical_target_overrides_authored_position_actuator(mjcf):
    # Given a native position actuator holding zero and a different initial position.
    mjcf.write_text(
        mjcf.read_text().replace('<motor joint="hinge"/>', '<position joint="hinge" kp="1000"/>')
    )
    scene = SceneSpec(
        agents=(
            AgentSpec(
                'robot',
                ModelSource.from_path(mjcf),
                ControlSpec(
                    joint_names=('hinge',),
                    initial_qpos=(0.2,),
                    joint_stiffness=(1.0,),
                    joint_damping=(0.1,),
                    joint_force_limits=(1.0,),
                ),
            ),
        )
    )
    with Backend().open(scene, ExecutionSpec()) as session:
        session.control.set_positions({'hinge': 0.3})

        # When explicit physical control advances the native model.
        for _ in range(20):
            session.step()

        # Then the authored servo does not pull the joint back to zero.
        position = np.asarray(session.state.get_qpos(), dtype=float)[0]
        assert 0.25 < position < 0.31


def test_real_urdf_import(tmp_path, mjcf):
    path = tmp_path / 'hinge.urdf'
    path.write_text(
        '<robot name="test"><link name="base"/><link name="arm">'
        '<inertial><mass value="1"/><origin xyz="0 0 0.5"/>'
        '<inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/></inertial>'
        '<collision><geometry><box size="0.1 0.1 1"/></geometry></collision></link>'
        '<joint name="hinge" type="revolute"><parent link="base"/><child link="arm"/>'
        '<axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="10" velocity="10"/>'
        '</joint></robot>'
    )
    with Backend().open(
        SceneSpec(agents=(AgentSpec('robot', ModelSource.from_path(path)),))
    ) as session:
        assert session.state.schema.joints[0].name == 'hinge'
        session.control.set_efforts({'hinge': 0.1})
        observation = session.step().observation
        assert observation.qvel is not None
        assert observation.qvel[0] > 0


def test_real_free_base_schema_and_episode_scalar_state(mjcf):
    path = mjcf.with_name('free.xml')
    path.write_text(
        mjcf.read_text()
        .replace('<joint name="hinge"', '<freejoint/><body name="tip"><joint name="hinge"')
        .replace('</body></worldbody>', '</body></body></worldbody>')
        .replace('<freejoint/>', '<freejoint/><geom size="0.1" mass="1"/>')
    )
    with Backend().open(
        SceneSpec(agents=(AgentSpec('robot', ModelSource.from_path(path)),))
    ) as session:
        assert session.state.get_qpos().shape == (8,)
        assert session.state.get_qvel().shape == (7,)
        assert session.state.schema.quaternion_order == 'wxyz'
        assert session.state.schema.joints[0].qpos_range == (0, 7)
        assert session.state.schema.joints[0].qvel_range == (0, 6)
    config = SimulatorConfig(
        env_id='JointControl',
        robot_path=str(path),
        robot_init_qpos_noise=0,
        render_mode='none',
        obs_mode='state',
        fixed_base=False,
    )
    simulator = MuJoCoSimulator(config)
    try:
        assert simulator.joint_names == ['hinge']
        assert simulator.get_qpos().shape == (1,)
    finally:
        simulator.close()


def test_real_grouped_cli_robot_route(mjcf):
    result = subprocess.run(
        [
            sys.executable,
            '-m',
            'usim.cli',
            'run',
            '--engine',
            'mujoco',
            '--robot',
            str(mjcf),
            '--joint-names',
            'hinge',
            '--render-mode',
            'none',
            '--obs-mode',
            'state',
            '--steps',
            '5',
            '--action',
            '0.3',
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary['engine'] == 'mujoco'
