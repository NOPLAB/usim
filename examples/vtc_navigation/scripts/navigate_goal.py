"""Set an initial pose and measure a real Nav2 NavigateToPose result."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Callable
from typing import TypedDict

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from lifecycle_msgs.msg import TransitionEvent
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.action import ActionClient
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.parameter import Parameter
from std_msgs.msg import Header
from std_srvs.srv import SetBool

from errors import VtcExampleError


class NavigationResult(TypedDict):
    """Machine-readable evidence from one Nav2 goal."""

    action_status: int
    success: bool
    goal_x_m: float
    goal_y_m: float
    localized_x_m: float
    localized_y_m: float
    goal_error_m: float
    map_width: int
    map_height: int
    odom_x_m: float
    odom_y_m: float


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Return planar yaw for an odometry quaternion."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class Navigator:
    """Observe localization and drive a goal through the upstream Nav2 stack."""

    def __init__(self) -> None:
        self.node = rclpy.create_node('vtc_nav2_goal')
        self.node.set_parameters([Parameter('use_sim_time', value=True)])
        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.map: OccupancyGrid | None = None
        self.odometry: tuple[float, float, float] | None = None
        self.localized_pose: tuple[float, float, float] | None = None
        self.localization_updates = 0
        self.node.create_subscription(OccupancyGrid, '/map', self.on_map, map_qos)
        self.node.create_subscription(Odometry, '/odom', self.on_odometry, 10)
        self.node.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.on_localized_pose,
            10,
        )
        self.initial_pose = self.node.create_publisher(
            PoseWithCovarianceStamped,
            '/initialpose',
            10,
        )
        self.motor = self.node.create_client(SetBool, '/motor_power')
        self.action = ActionClient(self.node, NavigateToPose, '/navigate_to_pose')
        self.navigation_nodes = ('controller_server', 'bt_navigator')
        self.active_navigation_nodes: set[str] = set()
        self.navigation_state_clients = {
            name: self.node.create_client(GetState, f'/{name}/get_state')
            for name in self.navigation_nodes
        }
        for name in self.navigation_nodes:
            self.node.create_subscription(
                TransitionEvent,
                f'/{name}/transition_event',
                lambda event, node_name=name: self.on_navigation_transition(node_name, event),
                10,
            )
        self.motor_enabled = False

    def on_map(self, message: OccupancyGrid) -> None:
        self.map = message

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

    def on_localized_pose(self, message: PoseWithCovarianceStamped) -> None:
        pose = message.pose.pose
        orientation = pose.orientation
        self.localized_pose = (
            pose.position.x,
            pose.position.y,
            yaw_from_quaternion(
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
            ),
        )
        self.localization_updates += 1

    def spin_until(
        self,
        ready: Callable[[], bool],
        timeout_seconds: float,
        reason: str,
    ) -> None:
        """Process ROS events until a state predicate succeeds or a deadline expires."""
        deadline = time.monotonic() + timeout_seconds
        while rclpy.ok() and not ready():
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Timed out waiting for {reason}')
            rclpy.spin_once(self.node, timeout_sec=0.1)
        if not rclpy.ok():
            raise VtcExampleError('Nav2', f'ROS shut down while waiting for {reason}')

    def set_motor_power(self, enabled: bool) -> None:
        """Enable or disable the simulated Raspberry Pi Cat motor service."""
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
                'Nav2',
                f'Could not set motor power to {enabled}: {message}',
            )
        self.motor_enabled = enabled

    def navigate(self, goal_x: float, goal_y: float) -> NavigationResult:
        """Send one bounded Nav2 goal after receiving the map and initial pose."""
        self.spin_until(
            lambda: self.map is not None and self.odometry is not None,
            30.0,
            'latched occupancy map and odometry',
        )
        if not self.action.wait_for_server(timeout_sec=30.0):
            raise TimeoutError('The /navigate_to_pose action server did not appear')
        self.spin_until(
            lambda: self.initial_pose.get_subscription_count() > 0,
            15.0,
            '/initialpose subscriber',
        )
        before_pose = self.localization_updates
        initial = PoseWithCovarianceStamped()
        initial.header = Header(
            stamp=self.node.get_clock().now().to_msg(),
            frame_id='map',
        )
        initial.pose.pose.position.x = 0.0
        initial.pose.pose.position.y = 0.0
        initial.pose.pose.orientation.w = 1.0
        initial.pose.covariance[0] = 0.25
        initial.pose.covariance[7] = 0.25
        initial.pose.covariance[35] = 0.07
        self.initial_pose.publish(initial)
        self.spin_until(
            lambda: self.localization_updates > before_pose,
            30.0,
            'AMCL localization after initial pose',
        )
        self.refresh_navigation_states()
        self.spin_until(
            lambda: self.active_navigation_nodes.issuperset(self.navigation_nodes),
            60.0,
            'Nav2 lifecycle activation',
        )

        self.set_motor_power(True)
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.node.get_clock().now().to_msg()
        goal.pose.pose.position.x = goal_x
        goal.pose.pose.position.y = goal_y
        goal.pose.pose.orientation.w = 1.0
        before_navigation_pose = self.localization_updates
        goal_future = self.action.send_goal_async(goal)
        rclpy.spin_until_future_complete(self.node, goal_future, timeout_sec=30.0)
        if not goal_future.done():
            raise TimeoutError('Nav2 did not accept or reject the goal')
        goal_handle = goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise VtcExampleError('Nav2', 'Rejected the VTC goal')

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future, timeout_sec=240.0)
        if not result_future.done():
            raise TimeoutError('Nav2 did not finish the VTC goal')
        wrapped_result = result_future.result()
        if wrapped_result is None:
            raise VtcExampleError('Nav2', 'Returned no action result')
        self.spin_until(
            lambda: self.localization_updates > before_navigation_pose,
            15.0,
            'AMCL pose after navigation',
        )
        if self.localized_pose is None or self.map is None or self.odometry is None:
            raise VtcExampleError('Nav2', 'Navigation observations were lost')
        distance = math.hypot(
            self.localized_pose[0] - goal_x,
            self.localized_pose[1] - goal_y,
        )
        success = wrapped_result.status == 4 and distance <= 0.4
        result: NavigationResult = {
            'action_status': int(wrapped_result.status),
            'success': success,
            'goal_x_m': float(goal_x),
            'goal_y_m': float(goal_y),
            'localized_x_m': float(self.localized_pose[0]),
            'localized_y_m': float(self.localized_pose[1]),
            'goal_error_m': float(distance),
            'map_width': int(self.map.info.width),
            'map_height': int(self.map.info.height),
            'odom_x_m': float(self.odometry[0]),
            'odom_y_m': float(self.odometry[1]),
        }
        return result

    def on_navigation_transition(self, name: str, event: TransitionEvent) -> None:
        """Track readiness from lifecycle transition events."""
        if event.goal_state.id == 3:
            self.active_navigation_nodes.add(name)
        else:
            self.active_navigation_nodes.discard(name)

    def refresh_navigation_states(self) -> None:
        """Seed readiness when a node activated before its transition subscription."""
        for name, client in self.navigation_state_clients.items():
            if not client.wait_for_service(timeout_sec=10.0):
                raise TimeoutError(f'The {name} lifecycle state service did not appear')
            future = client.call_async(GetState.Request())
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=10.0)
            if not future.done():
                raise TimeoutError(f'The {name} lifecycle state service did not respond')
            response = future.result()
            if response is None:
                raise VtcExampleError('Nav2', f'{name} returned no lifecycle state')
            self.on_navigation_state(name, response.current_state.id)

    def on_navigation_state(self, name: str, state: int) -> None:
        """Synchronize one lifecycle state query with transition tracking."""
        if state == 3:
            self.active_navigation_nodes.add(name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--goal-x', type=float, default=1.0)
    parser.add_argument('--goal-y', type=float, default=1.0)
    args = parser.parse_args()
    if not args.map.is_file():
        raise FileNotFoundError(args.map)
    rclpy.init()
    navigator = Navigator()
    try:
        result = navigator.navigate(args.goal_x, args.goal_y)
        args.result.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result), flush=True)
        if not result['success']:
            return 1
    finally:
        try:
            if navigator.motor_enabled:
                navigator.set_motor_power(False)
        finally:
            navigator.node.destroy_node()
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
