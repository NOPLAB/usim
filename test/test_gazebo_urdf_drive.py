"""URDF, drive, and camera asset behavior."""

from dataclasses import replace
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from usim.ports.gazebo import GazeboAssetConfig
from usim_gazebo.assets import prepare_assets
from usim.simulation import Ros2Config, SimulationConfig


def prepared(
    config: SimulationConfig, directory: Path
) -> tuple[ET.Element, list[tuple[Path, str]]]:
    mounts = prepare_assets(config, directory, GazeboAssetConfig('/private/run_test/cmd_vel'))
    return ET.parse(directory / 'robot.urdf').getroot(), mounts


def test_custom_drive_geometry_frames_and_topics_are_not_robot_specific(
    config: SimulationConfig, tmp_path: Path
) -> None:
    text = config.robot_urdf.read_text().replace('base_link', 'chassis')
    text = text.replace('left_wheel_joint', 'port_axle').replace('right_wheel_joint', 'starboard')
    config.robot_urdf.write_text(text)
    ros = Ros2Config(
        cmd_vel_topic='/pilot/drive',
        odom_topic='/pilot/pose',
        rgb_topic='/pilot/rgb',
        depth_topic='/pilot/depth',
    )
    config = replace(
        config,
        base_link='chassis',
        left_joint='port_axle',
        right_joint='starboard',
        wheel_radius=0.17,
        wheel_separation=0.61,
        ros=ros,
        camera_offset=(0.31, -0.02, 0.8),
        camera_width=123,
        camera_height=87,
        camera_hz=7.5,
    )
    before = config.robot_urdf.read_bytes(), config.world.read_bytes()
    root, _ = prepared(config, tmp_path / 'prepared')
    drive = root.find("gazebo/plugin[@name='usim_drive']")
    assert drive is not None
    assert drive.findtext('num_wheel_pairs') == '1'
    assert len(drive.findall('left_joint')) == len(drive.findall('right_joint')) == 1
    assert drive.findtext('left_joint') == 'port_axle'
    assert drive.findtext('right_joint') == 'starboard'
    assert float(drive.findtext('wheel_diameter', '')) == pytest.approx(0.34)
    assert float(drive.findtext('wheel_separation', '')) == pytest.approx(0.61)
    assert drive.findtext('robot_base_frame') == 'chassis'
    assert int(drive.findtext('odometry_source', '')) == 1
    assert drive.findtext('odometry_frame') == 'odom'
    assert [e.text for e in drive.findall('ros/remapping')] == [
        'cmd_vel:=/private/run_test/cmd_vel',
        'odom:=/pilot/pose',
    ]
    assert '/pilot/drive' not in ET.tostring(root, encoding='unicode')
    sensor = root.find("gazebo[@reference='chassis']/sensor")
    assert sensor is not None
    assert sensor.attrib['type'] == 'depth'
    assert tuple(map(float, sensor.findtext('pose', '').split())) == (0.31, -0.02, 0.8, 0, 0, 0)
    assert int(sensor.findtext('camera/image/width', '')) == 123
    assert int(sensor.findtext('camera/image/height', '')) == 87
    assert float(sensor.findtext('update_rate', '')) == 7.5
    assert sensor.findtext('camera/image/format') == 'R8G8B8'
    assert [e.text for e in sensor.findall('plugin/ros/remapping')] == [
        'usim_camera/image_raw:=/pilot/rgb',
        'usim_camera/depth/image_raw:=/pilot/depth',
    ]
    assert (config.robot_urdf.read_bytes(), config.world.read_bytes()) == before
    # Humble spawn_entity hands text to lxml, which rejects encoding declarations.
    assert not (tmp_path / 'prepared' / 'robot.urdf').read_bytes().startswith(b'<?xml')


def test_robot_plugins_cannot_bypass_gate_and_camera_can_be_omitted(
    config: SimulationConfig, tmp_path: Path
) -> None:
    root = ET.parse(config.robot_urdf).getroot()
    extension = ET.SubElement(root, 'gazebo')
    ET.SubElement(extension, 'plugin', name='old_drive', filename='libgazebo_ros_diff_drive.so')
    ET.SubElement(extension, 'plugin', name='other_controller', filename='custom_controller.so')
    ET.ElementTree(root).write(config.robot_urdf)
    robot, _ = prepared(replace(config, camera_enabled=False), tmp_path / 'prepared')
    assert [plugin.get('name') for plugin in robot.iter('plugin')] == ['usim_drive']
    assert not list(robot.iter('sensor'))
    assert len(robot.findall('link')) == len(root.findall('link'))


def test_no_ros_keeps_camera_sensor_but_has_no_ros_plugins(
    config: SimulationConfig, tmp_path: Path
) -> None:
    root, _ = prepared(replace(config, ros=None), tmp_path / 'prepared')
    assert not list(root.iter('plugin'))
    assert root.findall('gazebo/sensor')[0].get('type') == 'depth'


def test_paired_drive_plugin_entries(paired_config: SimulationConfig, tmp_path: Path) -> None:
    config = paired_config
    root, _ = prepared(config, tmp_path / 'prepared')
    drive = root.find("gazebo/plugin[@name='usim_drive']")
    assert drive is not None
    assert config.ros is not None
    assert drive.findtext('num_wheel_pairs') == '2'
    for tag, values in (
        ('left_joint', config.left_joints),
        ('right_joint', config.right_joints),
        ('wheel_diameter', (str(config.wheel_radius * 2),) * 2),
        ('wheel_separation', (str(config.wheel_separation),) * 2),
    ):
        assert tuple(element.text for element in drive.findall(tag)) == values
    assert drive.findtext('ros/remapping') == 'cmd_vel:=/private/run_test/cmd_vel'
    assert config.ros.cmd_vel_topic not in ET.tostring(drive, encoding='unicode').replace(
        '/private/run_test/cmd_vel', ''
    )
    with pytest.raises(ValueError, match='wheel joint must exist and rotate'):
        prepared(
            replace(config, left_joints=('left_wheel_joint', 'missing')), tmp_path / 'missing'
        )
