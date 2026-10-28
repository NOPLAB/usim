"""New engine distributions remain discoverable without their native SDKs."""

import subprocess
import sys

import pytest

from usim import capabilities, create_backend, list_backends, list_simulators
from usim.cli import build_parser


@pytest.mark.parametrize('engine', ['bullet', 'mujoco', 'mujoco-warp'])
def test_added_engine_is_selectable_for_model_episodes(engine):
    # Given installed engine base packages, with no native SDK requirement.
    assert engine in list_backends()
    assert engine in list_simulators()
    assert capabilities(engine) == frozenset({'episode'})

    # When selecting a caller-owned model through the shared command.
    args = build_parser().parse_args(
        [
            'run',
            '--engine',
            engine,
            '--robot',
            'robot.urdf',
            '--joint-names',
            'hinge',
        ]
    )

    # Then metadata discovery preserves the selected engine.
    assert args.engine == engine


def test_added_provider_discovery_never_imports_native_sdks():
    # Given an interpreter that rejects native SDK and array imports.
    code = """
import importlib.abc
import sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'numpy', 'pybullet', 'mujoco', 'mujoco_warp', 'warp',
        }:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Guard())
from usim import create_backend, list_backends, list_simulators
from usim.cli import build_parser
for name in ('bullet', 'mujoco', 'mujoco-warp'):
    assert name in list_backends()
    assert name in list_simulators()
    assert create_backend(name).describe().name == name
build_parser().parse_args(['run', '--help'])
"""

    # When loading provider descriptions and CLI help.
    result = subprocess.run(
        [sys.executable, '-B', '-c', code], capture_output=True, text=True, timeout=30
    )

    # Then only base-package contracts were needed.
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('engine', ['bullet', 'mujoco', 'mujoco-warp'])
def test_added_engine_reports_episode_execution(engine):
    # Given a provider loaded independently of its native SDK.
    provider = create_backend(engine)

    # When inspecting its public execution capabilities.
    description = provider.describe()

    # Then native episode stepping is advertised without a continuous ROS runner.
    assert description.name == engine
    assert description.execution_modes == frozenset({'episode'})
