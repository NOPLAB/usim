"""Verify shared drive contracts without either engine or wall-clock sleeps."""

import math
import unittest
from dataclasses import asdict, replace
import json
from pathlib import Path

from usim.simulation import (
    CommandGate,
    ConfigurationError,
    Ros2Config,
    SimulationConfig,
    Velocity,
    wheel_velocities,
)


class SimulationTest(unittest.TestCase):
    def test_navigation_fields_validate_and_roundtrip_json_lists(self):
        config = SimulationConfig(
            Path('world'),
            Path('robot'),
            initial_position=[300.0, -190.0, 0.2],
            lidar_offset=[0.144, 0.0, 0.105],
            lidar_vertical_angles=(0.0, math.pi / 9),
            real_time=True,
        )
        self.assertEqual(config.initial_position, (300.0, -190.0, 0.2))
        self.assertEqual(config.lidar_offset, (0.144, 0.0, 0.105))
        self.assertEqual(config.lidar_vertical_angles, (0.0, math.pi / 9))
        self.assertIsNone(SimulationConfig(Path('world'), Path('robot')).lidar_vertical_angles)
        for change in (
            dict(initial_position=(1, 2)),
            dict(initial_yaw=math.inf),
            dict(lidar_offset=(0, math.nan, 0)),
            dict(lidar_samples=1),
            dict(lidar_samples=3.5),
            dict(lidar_hz=0),
            dict(lidar_angle_min=1, lidar_angle_max=0),
            dict(lidar_min_range=-1),
            dict(lidar_min_range=30, lidar_max_range=30),
            dict(lidar_vertical_angles=[]),
            dict(lidar_vertical_angles='bad'),
            dict(lidar_vertical_angles=(math.nan,)),
            dict(lidar_vertical_angles=(math.inf,)),
            dict(lidar_vertical_angles=(math.pi,)),
        ):
            with self.subTest(change=change), self.assertRaises(ConfigurationError):
                replace(config, **change)

    def test_wheel_groups_preserve_singular_defaults_and_roundtrip(self):
        config = SimulationConfig(Path('world'), Path('robot'))
        self.assertEqual(config.left_wheel_joints, ('left_wheel_joint',))
        self.assertEqual(config.right_wheel_joints, ('right_wheel_joint',))
        self.assertEqual(replace(config, left_joint='custom').left_wheel_joints, ('custom',))
        paired = replace(config, left_joints=('lf', 'lr'), right_joints=('rf', 'rr'))
        data = asdict(paired)
        data.update(world='world', robot_urdf='robot', ros=None)
        restored = SimulationConfig(**json.loads(json.dumps(data)))
        self.assertEqual(restored.left_joints, ('lf', 'lr'))
        self.assertEqual(restored.right_joints, ('rf', 'rr'))
        self.assertEqual(restored.left_joint, config.left_joint)

    def test_wheel_groups_reject_invalid_selections(self):
        for left, right in (
            ((), ('r',)),
            (('l',), ()),
            (('l', 'l'), ('r', 's')),
            (('l', 'm'), ('r', 'r')),
            (('l',), ('l',)),
            (('l', 'm'), ('r',)),
            (('',), ('r',)),
            (('l',), (' ',)),
        ):
            with self.subTest(left=left, right=right), self.assertRaises(ConfigurationError):
                SimulationConfig(
                    Path('world'), Path('robot'), left_joints=left, right_joints=right
                )
        with self.assertRaises(ConfigurationError):
            SimulationConfig(Path('world'), Path('robot'), left_joints=json.loads('"left"'))

    def test_motor_toggle_clears_preexisting_commands(self):
        gate = CommandGate(clock=lambda: 0.0)
        gate.command(Velocity(0.3, 0.2))
        gate.set_motor(True)
        self.assertEqual(gate.sample(), Velocity())
        gate.command(Velocity(0.3, 0.2))
        self.assertEqual(gate.sample(), Velocity(0.3, 0.2))
        gate.set_motor(False)
        gate.set_motor(True)
        self.assertEqual(gate.sample(), Velocity())

    def test_watchdog_and_limits_use_independent_clock(self):
        current = [10.0]
        gate = CommandGate(clock=lambda: current[0])
        gate.set_motor(True)
        gate.command(Velocity(5.0, -3.0))
        self.assertEqual(gate.sample(), Velocity(0.4, -1.0))
        current[0] = 10.5
        self.assertEqual(gate.sample(), Velocity(0.4, -1.0))
        current[0] = 10.5001
        self.assertEqual(gate.sample(), Velocity())

    def test_drive_geometry_maps_forward_and_yaw(self):
        self.assertEqual(wheel_velocities(0.2, 0.0, 0.1, 0.4), (2.0, 2.0))
        self.assertEqual(wheel_velocities(0.0, 1.0, 0.1, 0.4), (-2.0, 2.0))

    def test_invalid_boundary_values_are_rejected(self):
        with self.assertRaises(ConfigurationError):
            Velocity(math.nan, 0.0)
        with self.assertRaises(ConfigurationError):
            SimulationConfig(Path('world'), Path('robot'), wheel_radius=0.0)
        with self.assertRaises(ConfigurationError):
            SimulationConfig(Path('world'), Path('robot'), camera_offset=(0.0, 0.0, math.inf))
        with self.assertRaises(ConfigurationError):
            SimulationConfig(Path('world'), Path('robot'), max_seconds=-1.0)
        with self.assertRaises(ConfigurationError):
            Ros2Config(odom_topic='')
        with self.assertRaises(ConfigurationError):
            Ros2Config(points_topic=' ')
