"""Verify a separately installed core wheel without numerical/native dependencies."""

import importlib.abc
import json
import sys
from importlib.resources import files
from xml.etree import ElementTree


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

assets = files('usim').joinpath('robots/assets')
model = ElementTree.parse(str(assets.joinpath('crane_x7.xml')))
compiler = model.find('compiler')
assert compiler is not None
mesh_directory = assets.joinpath(compiler.attrib.get('meshdir', ''))
meshes = [mesh_directory.joinpath(mesh.attrib['file']) for mesh in model.findall('asset/mesh')]
assert meshes and all(mesh.is_file() for mesh in meshes)
print(
    json.dumps(
        {
            'core': usim.__file__,
            'providers': usim.list_simulators(),
            'packaged_meshes': len(meshes),
        }
    )
)
build_parser().parse_args(['run', '--help'])
