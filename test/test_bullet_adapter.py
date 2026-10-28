"""Flat Bullet actions preserve explicitly selected velocity joints."""

import numpy as np

from test.test_bullet_backend import native as native, source as source
from usim.types import SimulatorConfig
from usim_bullet import BulletSimulator


def test_flat_velocity_action_reaches_velocity_controller(native, source):
    # Given a caller-owned model with its joint explicitly assigned to velocity control.
    simulator = BulletSimulator(
        SimulatorConfig(
            'JointControl',
            robot_path=str(source.path),
            robot_init_qpos_noise=0,
            render_mode='none',
            obs_mode='state',
            joint_names=('hinge',),
            velocity_joint_names=('hinge',),
        )
    )
    try:
        # When the flat action is submitted in the declared joint order.
        simulator.step(np.array([0.4]))

        # Then native joint velocity follows that target, not a position target.
        np.testing.assert_allclose(np.asarray(simulator.get_qvel(), dtype=float), [0.4], atol=1e-6)
    finally:
        simulator.close()
