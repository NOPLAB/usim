"""Run an odometry-guided square survey while SLAM builds a VTC occupancy map."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
import math
from pathlib import Path
import time
from typing import TypedDict

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from std_srvs.srv import SetBool

from errors import VtcExampleError


class ExplorationResult(TypedDict):
    """Machine-readable evidence from one SLAM mapping run."""

    duration_seconds: float
    distance_m: float
    map_width: int
    map_height: int
    known_cells: int
    map_updates: int
    scan_updates: int
    waypoints_completed: int


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Return planar yaw for an odometry quaternion."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class Explorer:
    """Drive a bounded square path and observe the live SLAM map."""

    def __init__(self) -> None:
        self.node = rclpy.create_node('vtc_slam_explorer')
        self.command = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.motor = self.node.create_client(SetBool, '/motor_power')
        self.odometry: tuple[float, float, float] | None = None
        self.map_shape: tuple[int, int] | None = None
        self.map_known_cells = 0
        self.map_updates = 0
        self.scan_updates = 0
        self.motor_enabled = False
        self.node.create_subscription(Odometry, '/odom', self.on_odometry, 10)
        self.node.create_subscription(OccupancyGrid, '/map', self.on_map, 10)
        self.node.create_subscription(LaserScan, '/scan', self.on_scan, 10)

    def on_odometry(self, message: Odometry) -> None:
        pose = message.pose.pose
        orientation = pose.orientation
        self.odometry = (
            pose.position.x,
            pose.position.y,
            yaw_from_quaternion(
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
            ),
        )

    def on_map(self, message: OccupancyGrid) -> None:
        self.map_shape = (message.info.width, message.info.height)
        self.map_known_cells = sum(value >= 0 for value in message.data)
        self.map_updates += 1

    def on_scan(self, _message: LaserScan) -> None:
        self.scan_updates += 1

    def spin_until(
        self,
        ready: Callable[[], bool],
        timeout_seconds: float,
        reason: str,
    ) -> None:
        """Spin callbacks until a subscribed state is observed or deadline expires."""
        deadline = time.monotonic() + timeout_seconds
        while rclpy.ok() and not ready():
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Timed out waiting for {reason}')
            rclpy.spin_once(self.node, timeout_sec=0.1)
        if not rclpy.ok():
            raise VtcExampleError('SLAM', f'ROS shut down while waiting for {reason}')

    def set_motor_power(self, enabled: bool) -> None:
        """Request the upstream Raspberry Pi Cat motor gate state."""
        if not self.motor.wait_for_service(timeout_sec=10.0):
            raise TimeoutError('The /motor_power service did not appear')
        future = self.motor.call_async(SetBool.Request(data=enabled))
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=10.0)
        if not future.done():
            raise TimeoutError('The /motor_power service did not respond')
        response = future.result()
        if response is None or not response.success:
            message = 'no response' if response is None else response.message
            raise VtcExampleError(
                'SLAM',
                f'Could not set motor power to {enabled}: {message}',
            )
        self.motor_enabled = enabled

    def drive_square(self) -> ExplorationResult:
        """Drive four 1.5 m sides using live odometry feedback."""
        self.spin_until(
            lambda: self.odometry is not None and self.scan_updates > 0,
            30.0,
            'initial odometry and lidar scan',
        )
        origin = self.odometry
        if origin is None:
            raise VtcExampleError('SLAM', 'Initial odometry disappeared')
        start_x, start_y, start_yaw = origin
        cosine, sine = math.cos(start_yaw), math.sin(start_yaw)
        local_targets = ((1.5, 0.0), (1.5, 1.5), (0.0, 1.5), (0.0, 0.0))
        targets = [
            (
                start_x + x * cosine - y * sine,
                start_y + x * sine + y * cosine,
            )
            for x, y in local_targets
        ]
        starting_map_updates = self.map_updates
        self.set_motor_power(True)
        start_time = time.monotonic()
        path_length = 0.0
        previous = self.odometry
        for target_x, target_y in targets:
            deadline = time.monotonic() + 120.0
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError('The robot did not reach a mapping waypoint')
                rclpy.spin_once(self.node, timeout_sec=0.1)
                pose = self.odometry
                if pose is None:
                    continue
                x, y, yaw = pose
                if previous is not None:
                    path_length += math.hypot(x - previous[0], y - previous[1])
                previous = pose
                dx, dy = target_x - x, target_y - y
                distance = math.hypot(dx, dy)
                if distance < 0.14:
                    break
                heading_error = math.atan2(
                    math.sin(math.atan2(dy, dx) - yaw),
                    math.cos(math.atan2(dy, dx) - yaw),
                )
                twist = Twist()
                if abs(heading_error) > 0.25:
                    twist.angular.z = max(-0.65, min(0.65, 1.4 * heading_error))
                else:
                    twist.linear.x = min(0.16, 0.5 * distance)
                    twist.angular.z = max(-0.45, min(0.45, 1.0 * heading_error))
                self.command.publish(twist)

        self.command.publish(Twist())
        self.set_motor_power(False)
        self.spin_until(
            lambda: self.map_updates > starting_map_updates,
            15.0,
            'updated SLAM occupancy grid',
        )
        if path_length < 4.0:
            raise VtcExampleError(
                'SLAM',
                f'Robot traveled only {path_length:.2f} m during mapping',
            )
        if self.map_shape is None or self.map_known_cells < 100:
            raise VtcExampleError(
                'SLAM',
                f'Map contains too few known cells: {self.map_known_cells}',
            )
        return {
            'duration_seconds': time.monotonic() - start_time,
            'distance_m': path_length,
            'map_width': self.map_shape[0],
            'map_height': self.map_shape[1],
            'known_cells': self.map_known_cells,
            'map_updates': self.map_updates,
            'scan_updates': self.scan_updates,
            'waypoints_completed': len(targets),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rclpy.init()
    explorer = Explorer()
    try:
        result = explorer.drive_square()
        args.output.with_name('exploration.json').write_text(
            json.dumps(result, indent=2),
            encoding='utf-8',
        )
        print(json.dumps(result), flush=True)
    finally:
        try:
            if explorer.motor_enabled and explorer.motor.service_is_ready():
                explorer.set_motor_power(False)
        finally:
            explorer.command.publish(Twist())
            explorer.node.destroy_node()
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
