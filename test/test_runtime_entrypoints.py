"""Import and help remain portable without native execution dependencies."""

from dataclasses import replace
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class EntryPointTest(unittest.TestCase):
    def test_builtin_help_and_authoring_without_optional_packages(self):
        guard = """
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'PIL', 'pxr', 'rclpy', 'isaacsim', 'omni'}:
            raise AssertionError('unexpected execution import: ' + fullname)
sys.meta_path.insert(0, Guard())
"""
        code = (
            'import usim; import usim.bridges.ros2; import usim.ports.isaac; '
            'from usim.cli import build_parser; '
            "build_parser().parse_args(['simulate', '--help'])"
        )
        child = subprocess.run(
            [sys.executable, '-c', guard + code], capture_output=True, text=True, timeout=30
        )
        self.assertEqual(child.returncode, 0, child.stdout + child.stderr)


class LibraryPortTest(unittest.TestCase):
    def test_typed_configuration_and_standalone_dispatch(self):
        from usim.ports.isaac import IsaacSimulator, SimulationConfig, simulate
        from usim_isaacsim.contacts import ObstacleContactLog

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            world, robot = root / 'room.usd', root / 'robot.urdf'
            world.write_bytes(b'fixture')
            robot.write_bytes(b'fixture')
            configuration = SimulationConfig(
                world=world, robot_urdf=robot, ros=None, camera_enabled=False
            )
            with patch('usim_isaacsim.runner.run') as execute:
                simulate(configuration)
                self.assertIs(execute.call_args.args[0], configuration)
                self.assertIsNone(execute.call_args.kwargs['stop'])
            with self.assertRaises(ValueError):
                simulate(replace(configuration, wheel_radius=0))
            with self.assertRaises(ValueError):
                IsaacSimulator(robot_prim_path='/World/Environment/Robot')
            contacts = ObstacleContactLog(
                {'Wall'},
                clock=lambda: 12,
                robot_prim_path='/World/Custom',
                scene_prim_path='/World/Room',
            )
            contacts.record(
                '/World/Custom/Base',
                '/World/Room/Wall/Body',
                '/World/Custom/Shape',
                '/World/Room/Wall/Shape',
                1,
                2,
            )
            contacts.record(
                '/World/Other/Base',
                '/World/Room/Wall/Body',
                '/World/Other/Shape',
                '/World/Room/Wall/Shape',
                1,
                2,
            )
            self.assertEqual(contacts.report()['collisions'], 1)
