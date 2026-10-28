"""Episode API boundary and portable registration regressions."""

import subprocess
import sys

import numpy as np
import pytest

from usim import EpisodeSimulator, Observation, SimulatorConfig, create_simulator
from usim.factory import list_simulators, register_simulator
from usim.cli import build_parser


@pytest.mark.parametrize(
    'options',
    [
        {'sim_rate': 0},
        {'sim_rate': float('nan')},
        {'n_envs': 0},
        {'max_episode_steps': -1},
        {'robot_init_qpos_noise': -0.01},
        {'joint_names': ('joint',), 'joint_force_limits': (float('inf'),)},
        {'joint_names': ('joint',), 'joint_force_limits': (0,)},
        {'joint_names': ('joint',), 'joint_stiffness': (-1,)},
        {'joint_names': ('joint',), 'joint_damping': (float('nan'),)},
        {'joint_names': ('joint',), 'joint_stiffness': (1, 2)},
        {'joint_force_limits': (1,)},
        {'arm_joint_names': ('joint',), 'gripper_joint_names': ('joint',)},
        {'arm_joint_names': ('',)},
        {'robot_path': 'missing-robot.xml'},
        {'initial_qpos': (float('nan'),)},
        {'arm_joint_names': ('joint',), 'initial_qpos': (0.0, 1.0)},
    ],
)
def test_invalid_configuration_is_rejected_before_native_execution(options):
    # Given invalid boundary inputs.
    # When constructing a simulation configuration.
    # Then native engines are not needed to reject them.
    with pytest.raises(ValueError):
        SimulatorConfig(env_id='JointControl', **options)


def test_core_and_simulator_help_do_not_import_native_dependencies():
    # Given an interpreter where optional imports are forbidden.
    code = """
import importlib.abc
import sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'numpy', 'torch', 'genesis', 'mani_skill', 'sapien', 'isaacsim', 'omni'
        }:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Guard())
import usim
from usim.cli import build_parser
build_parser().parse_args(['run', '--help'])
"""
    # When the user requests help.
    result = subprocess.run(
        [sys.executable, '-c', code], capture_output=True, text=True, timeout=30
    )
    # Then help succeeds without execution dependencies.
    assert result.returncode == 0, result.stdout + result.stderr


def test_simulator_cli_closes_registered_simulator_on_step_failure():
    # Given a registered simulator that fails during physics execution.
    instances = []

    class FailingSimulator(EpisodeSimulator):
        def __init__(self, config):
            super().__init__(config)
            self.closed = False
            instances.append(self)

        @property
        def arm_joint_names(self):
            return ['joint']

        @property
        def gripper_joint_names(self):
            return []

        def reset(self, seed=None):
            return Observation(qpos=np.zeros(1)), {}

        def step(self, action):
            raise RuntimeError('physics failed')

        def get_observation(self):
            return Observation(qpos=self.get_qpos())

        def get_qpos(self):
            return np.zeros(1)

        def get_qvel(self):
            return np.zeros(1)

        def close(self):
            self.closed = True

    from usim.factory import _SIMULATORS

    name = 'test_cli_lifecycle'
    register_simulator(name)(FailingSimulator)
    arguments = build_parser().parse_args(
        ['run', '--engine', name, '--steps', '1', '--action', '0']
    )
    # When execution raises.
    try:
        with pytest.raises(RuntimeError, match='physics failed'):
            arguments.handler(arguments)
        # Then the acquired simulator is closed.
        assert instances[0].closed
    finally:
        _SIMULATORS.pop(name)


def test_unknown_backend_is_rejected_without_loading_an_engine():
    # Given a valid task and an unknown backend name.
    config = SimulatorConfig(env_id='JointControl')
    # When selecting the backend.
    with pytest.raises(ValueError, match='Unknown simulator'):
        create_simulator('not-a-simulator', config)
    # Then discovery still lists the native choices without importing them.
    assert {'genesis', 'maniskill', 'isaacsim'} <= set(list_simulators())


def test_mixed_joint_commands_reject_competing_control_targets():
    from usim import JointCommand

    # Given one joint requested by two competing controllers.
    # When constructing the command.
    # Then the ambiguity is rejected before an engine runs.
    with pytest.raises(ValueError, match='simultaneously'):
        JointCommand(positions={'wheel': 0.1}, velocities={'wheel': 1.0})


def test_external_robot_configuration_has_no_registered_robot_default(tmp_path):
    # Given a caller-owned model and explicitly ordered controller settings.
    model = tmp_path / 'robot.urdf'
    model.write_text('<robot name="external"/>')
    # When configuring a file-backed robot without a registry identifier.
    config = SimulatorConfig(
        env_id='JointControl',
        robot_path=str(model),
        joint_names=('hinge',),
        joint_stiffness=(100.0,),
        joint_damping=(10.0,),
        joint_force_limits=(5.0,),
    )
    # Then file-backed execution does not select an implicit built-in robot.
    assert config.robot_uid == ''
    assert config.robot_path == str(model)
    assert config.joint_force_limits == (5.0,)
