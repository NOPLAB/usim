import copy
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import pytest

from usim.robot import MobileRobot, render_robot
from usim.simulation import SimulationConfig


@pytest.fixture
def config(tmp_path: Path) -> SimulationConfig:
    robot = tmp_path / 'robot.urdf'
    robot.write_text(render_robot(MobileRobot()), encoding='utf-8')
    world = tmp_path / 'world.sdf'
    world.write_text('<sdf version="1.6"><world name="custom"/></sdf>', encoding='utf-8')
    return SimulationConfig(world, robot)


@pytest.fixture
def paired_config(config: SimulationConfig) -> SimulationConfig:
    robot = ET.parse(config.robot_urdf).getroot()
    for side in ('left', 'right'):
        original = robot.find(f"joint[@name='{side}_wheel_joint']")
        assert original is not None
        origin = original.find('origin')
        assert origin is not None
        xyz = origin.get('xyz', '').split()
        origin.set('xyz', ' '.join(['0.1', *xyz[1:]]))
        extra = copy.deepcopy(original)
        extra.set('name', f'{side}_rear_joint')
        rear_origin, rear_child = extra.find('origin'), extra.find('child')
        assert rear_origin is not None and rear_child is not None
        rear_origin.set('xyz', ' '.join(['-0.1', *xyz[1:]]))
        rear_child.set('link', f'{side}_rear_link')
        robot.append(extra)
        original_link = robot.find(f"link[@name='{side}_wheel_link']")
        assert original_link is not None
        link = copy.deepcopy(original_link)
        link.set('name', f'{side}_rear_link')
        robot.append(link)
    ET.ElementTree(robot).write(config.robot_urdf)
    return replace(
        config,
        left_joints=('left_wheel_joint', 'left_rear_joint'),
        right_joints=('right_wheel_joint', 'right_rear_joint'),
    )
