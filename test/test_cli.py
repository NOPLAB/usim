"""Validate built-in command dispatch without starting a native engine."""

import tempfile
import unittest
from pathlib import Path

from usim.cli import build_parser
from usim.ports.isaac.cli import configured


class CliTest(unittest.TestCase):
    def test_custom_geometry_and_backend_configuration(self):
        parser = build_parser()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'robot.urdf'
            args = parser.parse_args(
                [
                    'create-robot',
                    '--out',
                    str(output),
                    '--name',
                    'delivery',
                    '--wheel-radius',
                    '0.09',
                    '--wheel-separation',
                    '0.36',
                ]
            )
            result = args.handler(args)
            self.assertEqual(result['robot'], 'delivery')
            self.assertTrue(output.is_file())
        args = parser.parse_args(
            [
                'simulate',
                '--backend',
                'gazebo',
                '--world',
                'room.sdf',
                '--robot-urdf',
                'robot.urdf',
                '--left-joint',
                'drive_l',
                '--right-joint',
                'drive_r',
                '--camera-offset',
                '0.3',
                '0.2',
                '0.5',
                '--cmd-vel-topic',
                '/drive',
                '--odom-topic',
                '/state',
                '--headless',
                '--max-seconds',
                '5',
            ]
        )
        config = configured(args)
        self.assertEqual((config.left_joint, config.right_joint), ('drive_l', 'drive_r'))
        self.assertEqual(config.camera_offset, (0.3, 0.2, 0.5))
        self.assertEqual((config.ros.cmd_vel_topic, config.ros.odom_topic), ('/drive', '/state'))
        self.assertEqual((config.headless, config.max_seconds), (True, 5))

    def test_no_ros_and_invalid_geometry(self):
        args = build_parser().parse_args(
            [
                'simulate',
                '--world',
                'room.usd',
                '--robot-urdf',
                'robot.urdf',
                '--no-ros',
                '--physics-only',
            ]
        )
        config = configured(args)
        self.assertIsNone(config.ros)
        self.assertFalse(config.camera_enabled)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'robot.urdf'
            args = build_parser().parse_args(
                ['create-robot', '--out', str(output), '--wheel-radius', '0']
            )
            with self.assertRaises(ValueError):
                args.handler(args)
            self.assertFalse(output.exists())
