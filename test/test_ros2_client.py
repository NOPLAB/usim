"""Exact-event ROS client tests without installed ROS modules or timing sleeps."""

import concurrent.futures
import sys
import threading
import types
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from usim.bridges.ros2 import Ros2Client
from usim.simulation import Ros2Config, Velocity


def odometry(seconds):
    return NS(
        header=NS(stamp=NS(sec=seconds, nanosec=250000000)),
        pose=NS(
            pose=NS(position=NS(x=1.0, y=2.0, z=3.0), orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0))
        ),
        twist=NS(twist=NS(linear=NS(x=0.2), angular=NS(z=0.1))),
    )


class Ros2ClientTest(unittest.TestCase):
    def setUp(self):
        self.actions = []
        self.sent = []
        self.responses = []
        self.service_ready = True
        self.subscriber_count = 1
        outer = self

        class Context:
            def shutdown(self):
                outer.actions.append('context_shutdown')

        class Executor:
            def __init__(self, context):
                self.stopped = threading.Event()

            def add_node(self, node):
                pass

            def spin(self):
                self.stopped.wait()

            def shutdown(self, timeout_sec):
                outer.actions.append('executor_shutdown')
                self.stopped.set()
                return True

        class Motor:
            def wait_for_service(self, timeout_sec):
                return outer.service_ready

            def call_async(self, request):
                outer.actions.append(('motor', request.data))
                return outer.responses.pop(0)

        class Node:
            def __init__(self, name, context):
                outer.actions.append(('node', name))

            def create_publisher(self, kind, topic, qos):
                outer.command_topic = topic
                return NS(
                    publish=outer.sent.append,
                    get_subscription_count=lambda: outer.subscriber_count,
                )

            def create_subscription(self, kind, topic, callback, qos):
                outer.callback = callback
                outer.odom_topic = topic

            def create_client(self, kind, service):
                outer.motor_service = service
                return Motor()

            def destroy_node(self):
                outer.actions.append('destroy_node')

        values = {
            'rclpy': dict(init=lambda **kw: None),
            'rclpy.context': dict(Context=Context),
            'rclpy.executors': dict(SingleThreadedExecutor=Executor),
            'rclpy.node': dict(Node=Node),
            'geometry_msgs.msg': dict(Twist=lambda: NS(linear=NS(x=0.0), angular=NS(z=0.0))),
            'nav_msgs.msg': dict(Odometry=NS),
            'std_srvs.srv': dict(SetBool=NS(Request=NS)),
        }
        modules = {}
        for name, attributes in values.items():
            module = types.ModuleType(name)
            module.__dict__.update(attributes)
            modules[name] = module
        patches = patch.dict(sys.modules, modules)
        patches.start()
        self.addCleanup(patches.stop)
        loader = patch('usim.bridges.ros_runtime.load_ros_python')
        loader.start()
        self.addCleanup(loader.stop)
        self.client = Ros2Client(
            Ros2Config(cmd_vel_topic='/drive', odom_topic='/state', motor_service='/power')
        )
        self.addCleanup(self.client.close)

    def test_commands_motor_future_and_ready_state(self):
        self.callback(odometry(1))
        state = self.client.wait_ready(timeout=1)
        self.assertEqual(state.sim_time, 1.25)
        self.assertEqual(state.position, (1.0, 2.0, 3.0))
        self.assertEqual(state.orientation_xyzw, (0.0, 0.0, 0.0, 1.0))
        self.assertEqual(state.velocity, Velocity(0.2, 0.1))
        self.client.command(Velocity(0.3, -0.2))
        self.assertEqual((self.sent[0].linear.x, self.sent[0].angular.z), (0.3, -0.2))
        self.assertEqual(
            (self.command_topic, self.odom_topic, self.motor_service),
            ('/drive', '/state', '/power'),
        )
        done = concurrent.futures.Future()
        done.set_result(NS(success=True))
        self.responses.append(done)
        self.client.set_motor(True, timeout=1)
        self.client.close()
        self.assertEqual(
            self.actions[-3:], ['executor_shutdown', 'destroy_node', 'context_shutdown']
        )
        with self.assertRaises(RuntimeError):
            self.client.command(Velocity())

    def test_observe_waits_for_exact_newer_odometry(self):
        self.callback(odometry(1))
        waiting = threading.Event()

        class SignalledCondition(threading.Condition):
            def wait(self, timeout=None):
                waiting.set()
                return super().wait(timeout)

        self.client._condition = SignalledCondition()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.client.observe, after=1.25, timeout=5)
            self.assertTrue(waiting.wait(timeout=5))
            self.callback(odometry(2))
            self.assertEqual(future.result(timeout=5).sim_time, 2.25)

    def test_ready_waits_for_command_subscriber(self):
        self.callback(odometry(1))
        self.subscriber_count = 0
        waiting = threading.Event()

        class SignalledCondition(threading.Condition):
            def wait(self, timeout=None):
                waiting.set()
                return super().wait(timeout)

        self.client._condition = SignalledCondition()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.client.wait_ready, timeout=5)
            self.assertTrue(waiting.wait(timeout=5))
            self.subscriber_count = 1
            self.callback(odometry(2))
            self.assertEqual(future.result(timeout=5).sim_time, 2.25)

    def test_timeouts_cancel_exact_future_and_service_failure(self):
        with self.assertRaises(TimeoutError):
            self.client.observe(timeout=0)
        pending = concurrent.futures.Future()
        self.responses.append(pending)
        with self.assertRaises(TimeoutError):
            self.client.set_motor(True, timeout=0)
        self.assertTrue(pending.cancelled())
        self.service_ready = False
        with self.assertRaises(TimeoutError):
            self.client.set_motor(True, timeout=0)
