"""ROS boundary regressions without an Isaac or ROS installation."""

import importlib.util
from array import array
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

from usim.bridges import ros_runtime
from usim.simulation import RobotState, Velocity


def vector():
    return NS(x=0.0, y=0.0, z=0.0)


def pose():
    return NS(position=vector(), orientation=NS(w=1.0, x=0.0, y=0.0, z=0.0))


def twist():
    return NS(linear=vector(), angular=vector())


def header():
    return NS(stamp=None, frame_id='')


class ImageMessage:
    """Model the generated uint8 sequence setter's native-array fast path."""

    def __init__(self):
        self.header = header()
        self.elementwise_bytes = 0
        self._data = array('B')

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if isinstance(value, array):
            assert value.typecode == 'B'
            self._data = value
        else:
            self.elementwise_bytes += len(value)
            self._data = array('B', value)


class Node:
    def __init__(self, name):
        self.name = name
        self.subscriptions = []
        self.services = []
        self.publishers = {}

    def create_subscription(self, kind, topic, callback, qos):
        self.subscriptions.append((kind, topic, callback, qos))

    def create_service(self, kind, topic, callback):
        self.services.append((kind, topic, callback))

    def create_publisher(self, kind, topic, qos):
        publisher = NS(kind=kind, qos=qos, messages=[])
        publisher.publish = publisher.messages.append
        self.publishers[topic] = publisher
        return publisher


def runtime_modules():
    values = {
        'geometry_msgs.msg': dict(
            PoseStamped=lambda: NS(header=header(), pose=pose()),
            TransformStamped=lambda: NS(
                header=header(),
                transform=NS(translation=vector(), rotation=NS(x=0, y=0, z=0, w=0)),
            ),
            Twist=twist,
        ),
        'nav_msgs.msg': dict(
            Odometry=lambda: NS(header=header(), pose=NS(pose=pose()), twist=NS(twist=twist())),
            Path=NS,
        ),
        'rclpy.node': dict(Node=Node),
        'rclpy.qos': dict(
            QoSProfile=NS,
            ReliabilityPolicy=NS(BEST_EFFORT=1),
            DurabilityPolicy=NS(TRANSIENT_LOCAL=2),
        ),
        'rclpy.time': dict(Time=lambda **kw: NS(to_msg=lambda: kw)),
        'rosgraph_msgs.msg': dict(Clock=NS),
        'sensor_msgs.msg': dict(
            Image=ImageMessage,
            LaserScan=lambda: NS(header=header()),
            PointCloud2=lambda: NS(header=header()),
            PointField=type('PointField', (NS,), {'FLOAT32': 7}),
        ),
        'std_srvs.srv': dict(SetBool=NS(Request=NS)),
        'tf2_msgs.msg': dict(TFMessage=NS),
        'rclpy': {},
    }
    modules = {}
    for name, attributes in values.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        modules[name] = module
    return modules


def load_boundary(name):
    path = Path(ros_runtime.__file__).with_name(name + '.py')
    spec = importlib.util.spec_from_file_location('_test_' + name, path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {**runtime_modules(), spec.name: module}):
        spec.loader.exec_module(module)
    return module


class BridgeTest(unittest.TestCase):
    def setUp(self):
        self.module = load_boundary('ros2')
        modules = patch.dict(sys.modules, runtime_modules())
        modules.start()
        self.addCleanup(modules.stop)

    def test_point_cloud_xyz_encoding_topic_and_transform_before_cloud(self):
        from usim.simulation import Ros2Config

        bridge = self.module.Bridge(
            camera_enabled=False,
            lidar_enabled=True,
            lidar_points=True,
            lidar_frame='sim_mid360_link',
            lidar_offset=(0.144, 0.0, 0.275),
            base_link='base_footprint',
            configuration=Ros2Config(points_topic='/livox/lidar'),
        )
        self.assertNotIn('/scan_raw', bridge.publishers)
        self.assertIsNone(bridge.scan_pub)
        self.assertEqual(bridge.points_pub.qos.reliability, 1)
        events = []
        bridge.tf_pub.publish = lambda message: events.append(('tf', message))
        bridge.points_pub.publish = lambda message: events.append(('points', message))
        stamp = {'seconds': 4.0}
        cloud = bridge.publish_points(
            stamp, [[1, 2, 3], [np.inf, 0, 0], [0, np.nan, 0], [-4, 5, -6]]
        )
        self.assertEqual([event[0] for event in events], ['tf', 'points'])
        self.assertIs(events[1][1], cloud)
        self.assertIs(cloud.header.stamp, stamp)
        self.assertEqual(cloud.header.frame_id, 'sim_mid360_link')
        self.assertEqual((cloud.height, cloud.width), (1, 2))
        self.assertEqual((cloud.point_step, cloud.row_step), (12, 24))
        self.assertFalse(cloud.is_bigendian)
        self.assertTrue(cloud.is_dense)
        self.assertEqual(
            [(f.name, f.offset, f.datatype, f.count) for f in cloud.fields],
            [('x', 0, 7, 1), ('y', 4, 7, 1), ('z', 8, 7, 1)],
        )
        self.assertEqual(cloud.data, np.array([[1, 2, 3], [-4, 5, -6]], dtype='<f4').tobytes())
        transform = events[0][1].transforms[0]
        self.assertEqual(transform.header.frame_id, 'base_footprint')
        self.assertIs(transform.header.stamp, stamp)
        self.assertEqual(transform.child_frame_id, 'sim_mid360_link')
        self.assertEqual(vars(transform.transform.translation), dict(x=0.144, y=0.0, z=0.275))
        self.assertEqual(vars(transform.transform.rotation), dict(x=0.0, y=0.0, z=0.0, w=1.0))

    def test_empty_cloud_and_external_sensor_transform(self):
        from usim.simulation import Ros2Config

        bridge = self.module.Bridge(
            camera_enabled=False,
            lidar_enabled=True,
            lidar_points=True,
            configuration=Ros2Config(publish_lidar_tf=False),
        )
        cloud = bridge.publish_points('stamp', np.empty((0, 3)))
        self.assertEqual((cloud.height, cloud.width, cloud.row_step, cloud.data), (1, 0, 0, b''))
        self.assertEqual(bridge.points_pub.messages, [cloud])
        self.assertEqual(bridge.tf_pub.messages, [])

    def test_scan_uses_external_robot_state_publisher_transform(self):
        from usim.simulation import Ros2Config

        bridge = self.module.Bridge(
            camera_enabled=False,
            lidar_enabled=True,
            configuration=Ros2Config(publish_lidar_tf=False),
        )
        bridge.publish_scan(
            {'seconds': 4.0},
            [1.0, 2.0],
            angle_min=-1,
            angle_max=1,
            range_min=0.05,
            range_max=30,
            scan_time=0.1,
        )
        self.assertEqual(len(bridge.scan_pub.messages), 1)
        self.assertEqual(bridge.tf_pub.messages, [])

    def test_physical_scan_metadata_and_matching_body_transform(self):
        from usim.simulation import Ros2Config

        bridge = self.module.Bridge(
            camera_enabled=False,
            lidar_enabled=True,
            lidar_offset=(0.1, -0.2, 0.4),
            base_link='base_footprint',
            configuration=Ros2Config(scan_topic='/scan_raw'),
        )
        stamp = {'seconds': 4.0}
        scan = bridge.publish_scan(
            stamp,
            [1.0, float('inf'), 3.0],
            angle_min=-1,
            angle_max=1,
            range_min=0.05,
            range_max=30,
            scan_time=0.1,
        )
        self.assertIs(bridge.publishers['/scan_raw'].messages[0], scan)
        self.assertEqual(scan.header.frame_id, 'lidar_link')
        self.assertIs(scan.header.stamp, stamp)
        self.assertEqual(scan.angle_increment, 1.0)
        self.assertEqual(scan.time_increment, 0.0)
        self.assertEqual(scan.scan_time, 0.1)
        self.assertEqual(scan.ranges, [1.0, float('inf'), 3.0])
        transform = bridge.tf_pub.messages[0].transforms[0]
        self.assertEqual(transform.header.frame_id, 'base_footprint')
        self.assertIs(transform.header.stamp, stamp)
        self.assertEqual(transform.child_frame_id, 'lidar_link')
        self.assertEqual(vars(transform.transform.translation), dict(x=0.1, y=-0.2, z=0.4))
        self.assertEqual(transform.transform.rotation.w, 1.0)

    def test_topics_qos_and_motor_command(self):
        bridge = self.module.Bridge(camera_enabled=True)
        self.assertEqual(bridge.name, 'usim_ros2_bridge')
        self.assertEqual(bridge.subscriptions[0][1], '/cmd_vel')
        self.assertEqual(bridge.subscriptions[0][3], 10)
        self.assertEqual(bridge.services[0][1], '/motor_power')
        self.assertEqual(
            set(bridge.publishers),
            {
                '/camera/color/image_raw',
                '/camera/depth/image_raw',
                '/clock',
                '/odom',
                '/sim/ground_truth_pose',
                '/tf',
            },
        )
        self.assertEqual(bridge.image_pub.qos.depth, 1)
        self.assertEqual(bridge.image_pub.qos.reliability, 1)
        command = twist()
        command.linear.x, command.angular.z = 2.0, -3.0
        response = bridge.on_motor(NS(data=True), NS())
        with patch.object(self.module.time, 'monotonic', return_value=12.0):
            bridge.on_command(command)
            self.assertEqual(bridge.command, (0.4, -1.0))
        self.assertEqual((bridge.command_count, bridge.last_command), (1, 12.0))
        self.assertTrue(response.success and bridge.motor_on)
        bridge.on_motor(NS(data=False), NS())
        self.assertFalse(bridge.motor_on)
        self.assertEqual(bridge.motor_requests, 2)
        physics = self.module.Bridge(camera_enabled=False)
        self.assertIsNone(physics.image_pub)
        self.assertIsNone(physics.depth_pub)

    def test_state_frames_and_camera_encoding(self):
        bridge = self.module.Bridge(camera_enabled=True)
        stamp = bridge.publish_state(RobotState(1.5, (2, 3, 4), (0, 0, 0, 1), Velocity(0.3, 0.2)))
        odom = bridge.odom_pub.messages[0]
        self.assertEqual(odom.header.stamp, stamp)
        self.assertEqual(odom.header.frame_id, 'odom')
        self.assertEqual(odom.child_frame_id, 'base_link')
        self.assertEqual(odom.pose.pose.orientation.w, 1.0)
        self.assertEqual(odom.pose.pose.orientation.x, 0.0)
        self.assertAlmostEqual(odom.twist.twist.linear.x, 0.3)
        self.assertEqual(odom.twist.twist.angular.z, 0.2)
        self.assertEqual(bridge.clock_pub.messages[0].clock, stamp)
        self.assertIs(bridge.pose_pub.messages[0].pose, odom.pose.pose)
        transform = bridge.tf_pub.messages[0].transforms[0]
        self.assertEqual(transform.child_frame_id, 'base_link')
        self.assertEqual(transform.transform.translation.z, 4.0)
        rgba = np.zeros((480, 640, 4), dtype=np.uint8)
        rgba[:, :, :3] = [1, 2, 3]
        image = bridge.publish_color(stamp, rgba)
        self.assertEqual(
            (image.encoding, image.step, image.header.frame_id), ('rgb8', 1920, 'camera_link')
        )
        self.assertEqual(bytes(image.data), bytes([1, 2, 3]) * (480 * 640))
        self.assertEqual(image.elementwise_bytes, 0)
        depth = np.full((480, 640), 2.5)
        bridge.publish_depth(image.header, depth)
        output = bridge.depth_pub.messages[0]
        self.assertIs(output.header, image.header)
        self.assertEqual((output.encoding, output.step), ('32FC1', 2560))
        self.assertEqual(bytes(output.data), depth.astype('<f4').tobytes())
        self.assertEqual(output.elementwise_bytes, 0)
        tiny = bridge.publish_color(stamp, np.zeros((3, 4, 3), dtype=np.uint8))
        bridge.publish_depth(tiny.header, np.ones((3, 4)))
        depth_output = bridge.depth_pub.messages[-1]
        self.assertEqual((tiny.width, tiny.height, tiny.step), (4, 3, 12))
        self.assertEqual((depth_output.width, depth_output.height, depth_output.step), (4, 3, 16))


class RosLoaderTest(unittest.TestCase):
    def test_sourced_environment_needs_no_isaac(self):
        with (
            patch.dict(sys.modules, {'rclpy': types.ModuleType('rclpy')}),
            patch.dict(ros_runtime.os.environ, {}, clear=True),
        ):
            ros_runtime.load_ros_python()

    def test_bundled_windows_loader_retains_handle(self):
        imported = []
        original = __import__

        def ros_import(name, *args, **kwargs):
            if name == 'rclpy':
                imported.append(name)
                if len(imported) == 1:
                    raise ModuleNotFoundError(name)
                return types.ModuleType(name)
            return original(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / 'exts/isaacsim.ros2.bridge/humble'
            (bundle / 'rclpy').mkdir(parents=True)
            (bundle / 'lib').mkdir()
            handle = object()
            with (
                patch('builtins.__import__', side_effect=ros_import),
                patch.dict(ros_runtime.os.environ, {'ISAAC_PATH': directory, 'PATH': ''}),
                patch.object(ros_runtime.sys, 'path', list(sys.path)),
                patch.object(ros_runtime.sys, 'platform', 'win32'),
                patch.object(
                    ros_runtime.os, 'add_dll_directory', return_value=handle, create=True
                ),
                patch.object(ros_runtime, '_ROS_DLL_DIRECTORY', None),
            ):
                ros_runtime.load_ros_python()
                self.assertIs(ros_runtime._ROS_DLL_DIRECTORY, handle)
                self.assertEqual(sys.path[-1], str(bundle / 'rclpy'))
                self.assertEqual(len(imported), 2)
