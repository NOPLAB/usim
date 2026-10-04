"""Check bridge callbacks without requiring a ROS installation."""

import importlib.util
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from usim import Observation, StepResult


def test_single_batch_episode_publishes_completion_and_resets(monkeypatch):
    modules = {
        name: ModuleType(name)
        for name in (
            'rclpy',
            'rclpy.node',
            'rcl_interfaces',
            'rcl_interfaces.msg',
            'sensor_msgs',
            'sensor_msgs.msg',
            'std_msgs',
            'std_msgs.msg',
            'std_srvs',
            'std_srvs.srv',
            'cv_bridge',
        )
    }
    modules['rclpy.node'].Node = object
    modules['rcl_interfaces.msg'].ParameterDescriptor = MagicMock()
    for name in ('Image', 'JointState'):
        setattr(modules['sensor_msgs.msg'], name, SimpleNamespace)
    for name in ('Bool', 'String', 'Float32MultiArray'):
        setattr(modules['std_msgs.msg'], name, SimpleNamespace)
    for name in ('Trigger', 'SetBool'):
        setattr(modules['std_srvs.srv'], name, SimpleNamespace)
    modules['cv_bridge'].CvBridge = object
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    origin = importlib.util.find_spec('usim.bridges.simulation_ros2').origin
    spec = importlib.util.spec_from_file_location('simulation_ros2_under_test', origin)
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    node = object.__new__(bridge.UsimSimNode)
    result = StepResult(
        Observation(),
        np.array([1.0]),
        np.array([True]),
        np.array([False]),
        [{'success': np.array([True])}],
    )
    node.simulator = SimpleNamespace(step=MagicMock(return_value=result), reset=MagicMock())
    node.latest_action = np.zeros(2, dtype=np.float32)
    node.episode_step = 0
    node.max_episode_steps = 200
    node.auto_reset = True
    logger = MagicMock()
    node.get_logger = lambda: logger
    node.episode_done_pub = MagicMock()
    node.task_info_pub = MagicMock()
    node._execute_step()

    logger.error.assert_not_called()
    assert node.episode_done_pub.publish.call_args.args[0].data is True
    node.simulator.reset.assert_called_once()
    assert node.latest_action is None
    assert node.episode_step == 0
