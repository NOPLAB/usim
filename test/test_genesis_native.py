"""Opt-in native Genesis smoke; run with USIM_GENESIS_NATIVE=1."""

import os

import numpy as np
import pytest

from usim.types import JointCommand, SimulatorConfig
from usim_genesis import GenesisSimulator


@pytest.mark.skipif(os.environ.get('USIM_GENESIS_NATIVE') != '1', reason='Native smoke is opt-in')
@pytest.mark.parametrize('robot', ['mjcf', 'urdf', 'mobile_arm', 'crane'])
def test_native_reset_action_observation_close(tmp_path, robot):
    # Given a real Genesis scene, optionally with an external one-joint robot.
    options = {}
    if robot == 'mjcf':
        path = tmp_path / 'robot.xml'
        path.write_text(
            '<mujoco><worldbody><body pos="0 0 0.5">'
            '<joint name="hinge" type="hinge" axis="0 0 1"/>'
            '<geom type="box" size="0.1 0.03 0.03" mass="1"/>'
            '</body></worldbody></mujoco>'
        )
        options = {'robot_path': str(path), 'arm_joint_names': ('hinge',)}
    elif robot == 'mobile_arm':
        path = tmp_path / 'mobile_arm.urdf'
        inertia = (
            '<inertial><mass value="1"/>'
            '<inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>'
            '</inertial>'
        )
        path.write_text(
            '<robot name="mobile_arm"><link name="base">'
            + inertia
            + '<visual><geometry><box size="0.2 0.2 0.1"/></geometry></visual>'
            '<collision><geometry><box size="0.2 0.2 0.1"/></geometry></collision></link>'
            + ''.join(
                f'<link name="{name}">{inertia}<visual><geometry>'
                '<box size="0.1 0.04 0.04"/></geometry></visual></link>'
                f'<joint name="{name}" type="continuous"><parent link="base"/>'
                f'<child link="{name}"/><origin xyz="0 0 {height}"/><axis xyz="0 0 1"/>'
                '<limit effort="10" velocity="2"/></joint>'
                for name, height in [('wheel', 0.1), ('arm', 0.3)]
            )
            + '</robot>'
        )
        options = {
            'robot_path': str(path),
            'joint_names': ('arm', 'wheel'),
            'velocity_joint_names': ('wheel',),
            'fixed_base': False,
        }
    elif robot == 'urdf':
        path = tmp_path / 'robot.urdf'
        path.write_text(
            '<robot name="one_joint"><link name="base"/>'
            '<link name="arm"><inertial><mass value="1"/>'
            '<inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>'
            '</inertial><visual><geometry><box size="0.2 0.06 0.06"/></geometry></visual>'
            '<collision><geometry><box size="0.2 0.06 0.06"/></geometry></collision></link>'
            '<joint name="hinge" type="revolute"><parent link="base"/><child link="arm"/>'
            '<origin xyz="0 0 0.5"/><axis xyz="0 0 1"/>'
            '<limit lower="-1" upper="1" effort="10" velocity="2"/></joint></robot>'
        )
        options = {'robot_path': str(path), 'arm_joint_names': ('hinge',)}
    backend = os.environ.get('USIM_GENESIS_BACKEND', 'cpu')
    render = os.environ.get('USIM_GENESIS_RENDER', 'none')
    sim = GenesisSimulator(
        SimulatorConfig(
            env_id='PickPlace-CRANE-X7' if robot == 'crane' else 'JointControl',
            backend=backend,
            render_mode=render,
            obs_mode='rgbd',
            robot_init_qpos_noise=0,
            n_envs=2,
            **options,
        )
    )
    try:
        initial, _ = sim.reset(seed=7)
        expected = initial.qpos.copy()
        action = expected.copy()
        action[:, 0] += [0.1, -0.1]
        if robot == 'mobile_arm':
            action = JointCommand(
                positions={'arm': np.array([0.1, -0.1])},
                velocities={'wheel': np.array([0.2, 0.3])},
            )
        # When applying real native actions and resetting one environment.
        for _ in range(5):
            result = sim.step(action)
        other = result.observation.qpos[1].copy()
        reset, _ = sim.reset(seed=7, env_ids=np.array([0]))
        # Then states are finite, the selected reset is exact, and the other state is untouched.
        assert np.all(np.isfinite(result.observation.qpos))
        assert np.any(np.abs(result.observation.qpos[:, 0] - expected[:, 0]) > 1e-6)
        if robot == 'mobile_arm':
            assert np.all(result.observation.qvel[:, 1] > 0)
        np.testing.assert_allclose(reset.qpos[0], expected[0], atol=1e-6)
        np.testing.assert_allclose(reset.qpos[1], other, atol=1e-6)
        if render != 'none':
            assert reset.rgb_image.shape == (2, 480, 640, 3)
            assert reset.depth_image.shape == (2, 480, 640)
            assert not np.array_equal(reset.rgb_image[0], reset.rgb_image[1])
            from PIL import Image

            image = tmp_path / f'{robot}.png'
            Image.fromarray(reset.rgb_image[0]).save(image)
            print(f'NATIVE image={image}')
        print(
            f'NATIVE {robot} {backend}: reset={initial.qpos.tolist()} '
            f'action={action} observation={result.observation.qpos.tolist()} '
            f'rgb={None if reset.rgb_image is None else reset.rgb_image.shape}'
        )
    finally:
        sim.close()
    assert not sim.is_running
    print(f'NATIVE {robot}: close={not sim.is_running}')
