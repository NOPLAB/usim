"""Physical control must not fight native servos or lose per-joint modes."""

from dataclasses import replace

import numpy as np
import pytest

from test.test_mujoco_backend import native as native, scene as scene
from usim.specs import ExecutionSpec
from usim.types import SimulatorConfig
from usim_mujoco import Backend, MuJoCoSimulator


def test_physical_target_overrides_native_position_servo(native, scene):
    # Given an authored actuator holding zero against a nonzero initial position.
    _, _, sdk = native

    def forward(model, state):
        state.qfrc_actuator[:] = -1000 * state.qpos

    sdk.mj_forward = forward
    agent = replace(scene.agents[0], control=replace(scene.agents[0].control, initial_qpos=(0.2,)))
    with Backend().open(replace(scene, agents=(agent,)), ExecutionSpec()) as session:
        session.control.set_positions({'hinge': 0.3})

        # When the physical controller advances the model.
        session.step()

        # Then native servo bias does not pull the joint away from its target.
        assert session.state.get_qpos()[0] > 0.2


@pytest.mark.parametrize(
    ('mode', 'expected'),
    [('pd_joint_pos', 'velocity'), ('pd_joint_vel', 'velocity'), ('effort', 'effort')],
)
def test_flat_actions_follow_selected_joint_modes(native, scene, mode, expected):
    # Given an explicitly ordered velocity joint and the selected array control mode.
    config = SimulatorConfig(
        env_id='JointControl',
        robot_path=str(scene.agents[0].model.path),
        render_mode='none',
        obs_mode='state',
        robot_init_qpos_noise=0,
        joint_names=('hinge',),
        velocity_joint_names=('hinge',),
        control_mode=mode,
    )
    simulator = MuJoCoSimulator(config)
    try:
        # When a flat action is submitted.
        simulator.step(np.array([0.3]))

        # Then its physical target uses the configured mode.
        assert simulator._require().targets == {'hinge': (expected, 0.3)}
    finally:
        simulator.close()


def test_flat_action_infers_native_joint_order(native, scene):
    # Given a model whose joint order was not overridden by the caller.
    simulator = MuJoCoSimulator(
        SimulatorConfig(
            env_id='JointControl',
            robot_path=str(scene.agents[0].model.path),
            render_mode='none',
            obs_mode='state',
            robot_init_qpos_noise=0,
        )
    )
    try:
        # When an action follows the native joint order.
        simulator.step(np.array([0.3]))

        # Then it reaches the inferred position controller.
        assert simulator._require().targets == {'hinge': ('position', 0.3)}
    finally:
        simulator.close()
