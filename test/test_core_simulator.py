"""Episode API boundary and portable registration regressions."""

import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

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
        {'gripper_force_limit': float('inf')},
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
        ['run', '--engine', 'genesis', '--steps', '1', '--action', '0']
    )
    arguments.engine = name
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


def test_bundled_robot_model_resolves_every_mesh_without_consumer_checkout():
    # Given the model resolved by the installed simulator package.
    from usim.robots.crane_x7 import get_mjcf_path

    model = Path(get_mjcf_path())
    # When the model's native asset references are resolved.
    tree = ElementTree.parse(model)
    compiler = tree.find('compiler')
    assert compiler is not None
    mesh_directory = model.parent / compiler.attrib.get('meshdir', '')
    meshes = [mesh_directory / mesh.attrib['file'] for mesh in tree.findall('asset/mesh')]
    # Then every mesh required to construct the robot is packaged with it.
    assert meshes
    assert all(mesh.is_file() and mesh.stat().st_size > 0 for mesh in meshes)
