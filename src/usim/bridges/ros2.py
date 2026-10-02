"""Standard ROS 2 I/O; ROS message modules load only when a bridge is created."""

from __future__ import annotations

import time

from usim.simulation import CommandGate, RobotState, Ros2Config, Velocity


def Bridge(
    *,
    camera_enabled: bool,
    configuration: Ros2Config | None = None,
    base_link: str = 'base_link',
    optical_frame: str = 'camera_link',
):
    """Create a ROS node at the execution boundary, leaving module imports portable."""
    configuration = configuration if configuration is not None else Ros2Config()
    import numpy as np
    from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from rclpy.time import Time
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import Image
    from std_srvs.srv import SetBool
    from tf2_msgs.msg import TFMessage

    class Bridge(Node):
        def __init__(self):
            super().__init__(configuration.node_name)
            self.gate = CommandGate(clock=lambda: time.monotonic())
            self.last_command = 0.0
            self.command_count = 0
            self.motor_requests = 0
            self.create_subscription(Twist, configuration.cmd_vel_topic, self.on_command, 10)
            self.create_service(SetBool, configuration.motor_service, self.on_motor)
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.image_pub = None
            self.depth_pub = None
            if camera_enabled:
                self.image_pub = self.create_publisher(Image, configuration.rgb_topic, qos)
                self.depth_pub = self.create_publisher(Image, configuration.depth_topic, qos)
            self.clock_pub = self.create_publisher(Clock, '/clock', 10)
            self.odom_pub = self.create_publisher(Odometry, configuration.odom_topic, 10)
            self.pose_pub = self.create_publisher(PoseStamped, '/sim/ground_truth_pose', 10)
            self.tf_pub = self.create_publisher(TFMessage, '/tf', 10)

        @property
        def command(self):
            motion = self.gate.sample()
            return motion.linear, motion.angular

        @property
        def motor_on(self):
            return self.gate.enabled

        def on_command(self, message):
            self.command_count += 1
            try:
                motion = Velocity(float(message.linear.x), float(message.angular.z))
            except ValueError as error:
                self.get_logger().warning(str(error))
                return
            self.gate.command(motion)
            self.last_command = time.monotonic()

        def on_motor(self, request, response):
            self.motor_requests += 1
            self.gate.set_motor(bool(request.data))
            response.success = True
            response.message = 'motor enabled' if self.motor_on else 'motor disabled'
            return response

        def publish_state(self, state: RobotState):
            """Publish clock, body odometry, ground truth and the odom transform."""
            stamp = Time(seconds=state.sim_time).to_msg()
            clock = Clock()
            clock.clock = stamp
            self.clock_pub.publish(clock)
            odom = Odometry()
            odom.header.stamp = stamp
            odom.header.frame_id = 'odom'
            odom.child_frame_id = base_link
            odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = (
                float(v) for v in state.position
            )
            (
                odom.pose.pose.orientation.x,
                odom.pose.pose.orientation.y,
                odom.pose.pose.orientation.z,
                odom.pose.pose.orientation.w,
            ) = state.orientation_xyzw
            odom.twist.twist.linear.x = state.velocity.linear
            odom.twist.twist.angular.z = state.velocity.angular
            self.odom_pub.publish(odom)
            pose = PoseStamped()
            pose.header = odom.header
            pose.pose = odom.pose.pose
            self.pose_pub.publish(pose)
            transform = TransformStamped()
            transform.header = odom.header
            transform.child_frame_id = base_link
            transform.transform.translation.x = odom.pose.pose.position.x
            transform.transform.translation.y = odom.pose.pose.position.y
            transform.transform.translation.z = odom.pose.pose.position.z
            transform.transform.rotation = odom.pose.pose.orientation
            self.tf_pub.publish(TFMessage(transforms=[transform]))
            return stamp

        def publish_color(self, stamp, rgba):
            image = Image()
            image.header.stamp = stamp
            image.header.frame_id = optical_frame
            image.height, image.width = rgba.shape[:2]
            image.encoding = 'rgb8'
            image.step = image.width * 3
            image.data = rgba[:, :, :3].astype('uint8').tobytes()
            self.image_pub.publish(image)
            return image

        def publish_depth(self, header, depth):
            depth_image = Image()
            depth_image.header = header
            depth_image.height, depth_image.width = depth.shape[:2]
            depth_image.encoding = '32FC1'
            depth_image.step = depth_image.width * 4
            depth_image.data = np.asarray(depth, dtype='<f4').tobytes()
            self.depth_pub.publish(depth_image)

    return Bridge()


class Ros2Client:
    """Bounded ROS control/state client with a private executor and ROS context."""

    def __init__(self, configuration: Ros2Config | None = None) -> None:
        import threading
        from usim.bridges.ros_runtime import load_ros_python

        load_ros_python()
        import rclpy
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from std_srvs.srv import SetBool

        self.configuration = configuration if configuration is not None else Ros2Config()
        self._condition = threading.Condition()
        self._state = None
        self._closed = False
        self._Twist, self._SetBool = Twist, SetBool
        self._context = Context()
        rclpy.init(context=self._context)
        self._node = Node(self.configuration.node_name + '_client', context=self._context)
        self._commands = self._node.create_publisher(Twist, self.configuration.cmd_vel_topic, 10)
        self._node.create_subscription(Odometry, self.configuration.odom_topic, self._on_odom, 10)
        self._motor = self._node.create_client(SetBool, self.configuration.motor_service)
        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self._node)
        self._thread = threading.Thread(
            target=self._executor.spin, daemon=True, name='usim-ros2-client'
        )
        self._thread.start()

    def _on_odom(self, message) -> None:
        from usim.simulation import RobotState, Velocity

        pose = message.pose.pose
        stamp = message.header.stamp
        state = RobotState(
            sim_time=float(stamp.sec) + float(stamp.nanosec) * 1e-9,
            position=(float(pose.position.x), float(pose.position.y), float(pose.position.z)),
            orientation_xyzw=(
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            ),
            velocity=Velocity(
                float(message.twist.twist.linear.x), float(message.twist.twist.angular.z)
            ),
        )
        with self._condition:
            self._state = state
            self._condition.notify_all()

    def wait_ready(self, timeout: float = 30) -> RobotState:
        """Wait for state, motor service and command matching within one deadline."""
        deadline = time.monotonic() + timeout
        state = self.observe(timeout=timeout)
        if not self._motor.wait_for_service(timeout_sec=max(0, deadline - time.monotonic())):
            raise TimeoutError('ROS motor service is unavailable')
        while self._commands.get_subscription_count() == 0:
            state = self.observe(after=state.sim_time, timeout=max(0, deadline - time.monotonic()))
        return state

    def command(self, velocity: Velocity) -> None:
        """Publish a body velocity without inventing simulator-specific actions."""
        if self._closed:
            raise RuntimeError('ROS client is closed')
        message = self._Twist()
        message.linear.x, message.angular.z = velocity.linear, velocity.angular
        self._commands.publish(message)

    def set_motor(self, enabled: bool, timeout: float = 5) -> None:
        import threading

        if self._closed:
            raise RuntimeError('ROS client is closed')
        deadline = time.monotonic() + timeout
        if not self._motor.wait_for_service(timeout_sec=timeout):
            raise TimeoutError('ROS motor service is unavailable')
        completed = threading.Event()
        future = self._motor.call_async(self._SetBool.Request(data=enabled))
        future.add_done_callback(lambda result: completed.set())
        if not completed.wait(max(0, deadline - time.monotonic())):
            future.cancel()
            raise TimeoutError('ROS motor request timed out')
        response = future.result()
        if response is None or not response.success:
            raise RuntimeError('ROS motor request was rejected')

    def observe(self, after: float | None = None, timeout: float = 5) -> RobotState:
        """Wait for an exact odometry update newer than the supplied simulation time."""
        with self._condition:
            ready = self._condition.wait_for(
                lambda: (
                    self._closed
                    or (
                        self._state is not None and (after is None or self._state.sim_time > after)
                    )
                ),
                timeout=timeout,
            )
            if self._closed:
                raise RuntimeError('ROS client is closed')
            if not ready:
                raise TimeoutError('ROS odometry timed out')
            return self._state

    def close(self) -> None:
        """Stop the executor, join its exact shutdown, and release this client's context."""
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()
        self._executor.shutdown(timeout_sec=5)
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise RuntimeError('ROS client executor did not stop')
        self._node.destroy_node()
        self._context.shutdown()
