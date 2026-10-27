"""Validate built-in command dispatch without starting a native engine."""

import tempfile
import unittest
from pathlib import Path

from usim.cli import build_parser
from usim.runtime_cli import configured


class CliTest(unittest.TestCase):
    def test_mobile_navigation_options_reach_shared_configuration(self):
        args = build_parser().parse_args(
            [
                'simulate',
                '--world',
                'room.usd',
                '--robot-urdf',
                'robot.urdf',
                '--initial-position',
                '300',
                '-190',
                '0.2',
                '--initial-yaw',
                '1.5',
                '--real-time',
                '--physics-only',
                '--lidar-enabled',
                '--lidar-offset',
                '0.144',
                '0',
                '0.105',
                '--lidar-samples',
                '720',
                '--scan-topic',
                '/scan_raw',
            ]
        )
        config = configured(args)
        self.assertEqual(config.initial_position, (300.0, -190.0, 0.2))
        self.assertEqual(config.initial_yaw, 1.5)
        self.assertTrue(config.real_time and config.lidar_enabled)
        self.assertFalse(config.camera_enabled)
        self.assertEqual(config.lidar_offset, (0.144, 0.0, 0.105))
        self.assertEqual(config.lidar_samples, 720)
        self.assertIsNotNone(config.ros)
        self.assertEqual(config.ros.scan_topic, '/scan_raw')

    def test_wheel_groups_override_singular_flags(self):
        args = build_parser().parse_args(
            [
                'simulate',
                '--world',
                'room.sdf',
                '--robot-urdf',
                'robot.urdf',
                '--left-joint',
                'legacy_l',
                '--right-joint',
                'legacy_r',
                '--left-joints',
                'lf',
                'lr',
                '--right-joints',
                'rf',
                'rr',
            ]
        )
        config = configured(args)
        self.assertEqual(config.left_joints, ('lf', 'lr'))
        self.assertEqual(config.right_joints, ('rf', 'rr'))
        self.assertEqual((config.left_joint, config.right_joint), ('legacy_l', 'legacy_r'))

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
