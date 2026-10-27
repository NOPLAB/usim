"""Verify a separately installed core wheel without numerical/native dependencies."""

import importlib.abc
import json
import sys
from importlib.util import find_spec


class NativeImportGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'numpy',
            'torch',
            'genesis',
            'mani_skill',
            'sapien',
            'isaacsim',
            'omni',
            'usim_genesis',
            'usim_maniskill',
            'usim_isaacsim',
            'usim_gazebo',
        }:
            raise AssertionError(f'core unexpectedly imports engine dependency: {fullname}')


sys.meta_path.insert(0, NativeImportGuard())

import usim  # noqa: E402
from usim.cli import build_parser  # noqa: E402

assert find_spec('usim.robots') is None
assert usim.SimulatorConfig(env_id='JointControl').robot_uid == ''
print(
    json.dumps(
        {
            'core': usim.__file__,
            'providers': usim.list_simulators(),
            'bundled_robots': False,
        }
    )
)
build_parser().parse_args(['run', '--help'])
