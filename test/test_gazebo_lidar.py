"""Gazebo lidar configuration and generated sensors."""

import math
from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from usim.ports.gazebo import GazeboAssetConfig, GazeboLidarConfig
from usim_gazebo.assets import prepare_assets
from usim.simulation import ConfigurationError, SimulationConfig


def test_lidar_is_injected_after_plugin_stripping_with_configured_ray(
    config: SimulationConfig, tmp_path: Path
) -> None:
    root = ET.parse(config.robot_urdf).getroot()
    ET.SubElement(root, 'link', name='scanner_mount')
    extension = ET.SubElement(root, 'gazebo', reference='scanner_mount')
    sensor = ET.SubElement(extension, 'sensor', name='old_scan', type='ray')
    ET.SubElement(sensor, 'plugin', name='old_scan', filename='libgazebo_ros_ray_sensor.so')
    ET.ElementTree(root).write(config.robot_urdf)
    before = config.robot_urdf.read_bytes(), config.world.read_bytes()
    lidar = GazeboLidarConfig(
        link_name='scanner_mount',
        frame_name='laser_frame',
        topic='/robot/scan',
        update_rate=12.5,
        horizontal_samples=360,
        min_angle=-1.5,
        max_angle=2.0,
        range_min=0.2,
        range_max=25.0,
    )
    directory = tmp_path / 'prepared'
    prepare_assets(
        replace(config, camera_enabled=False),
        directory,
        GazeboAssetConfig('/private/run_test/cmd_vel', lidar),
    )
    robot = ET.parse(directory / 'robot.urdf').getroot()
    assert [plugin.get('name') for plugin in robot.iter('plugin')] == [
        'usim_drive',
        'usim_lidar_ros',
    ]
    assert [sensor.get('name') for sensor in robot.iter('sensor')] == ['old_scan', 'usim_lidar']
    scan = robot.find("gazebo[@reference='scanner_mount']/sensor[@name='usim_lidar']")
    assert scan is not None and scan.get('type') == 'ray'
    assert scan.findtext('always_on') == 'true'
    assert float(scan.findtext('update_rate', '')) == 12.5
    assert int(scan.findtext('ray/scan/horizontal/samples', '')) == 360
    assert float(scan.findtext('ray/scan/horizontal/min_angle', '')) == -1.5
    assert float(scan.findtext('ray/scan/horizontal/max_angle', '')) == 2.0
    assert scan.find('ray/scan/vertical') is None
    assert float(scan.findtext('ray/range/min', '')) == 0.2
    assert float(scan.findtext('ray/range/max', '')) == 25.0
    plugin = scan.find('plugin')
    assert plugin is not None
    assert plugin.get('filename') == 'libgazebo_ros_ray_sensor.so'
    assert plugin.findtext('output_type') == 'sensor_msgs/LaserScan'
    assert plugin.findtext('frame_name') == 'laser_frame'
    assert [e.text for e in plugin.findall('ros/remapping')] == ['~/out:=/robot/scan']
    assert robot.find("gazebo/sensor[@type='depth']") is None
    assert (config.robot_urdf.read_bytes(), config.world.read_bytes()) == before


def test_lidar_without_ros_keeps_ray_sensor_but_no_plugin(
    config: SimulationConfig, tmp_path: Path
) -> None:
    lidar = GazeboLidarConfig('base_link', 'base_scan', '/scan')
    prepare_assets(
        replace(config, ros=None),
        tmp_path / 'bare',
        GazeboAssetConfig('/private/x/cmd_vel', lidar),
    )
    bare = ET.parse(tmp_path / 'bare' / 'robot.urdf').getroot()
    assert not list(bare.iter('plugin'))
    assert [sensor.get('type') for sensor in bare.iter('sensor')] == ['depth', 'ray']


@pytest.mark.parametrize(
    'change',
    [
        {'link_name': ' '},
        {'frame_name': ''},
        {'topic': 'scan'},
        {'topic': '/my scan'},
        {'update_rate': 0.0},
        {'update_rate': math.inf},
        {'horizontal_samples': 0},
        {'horizontal_samples': True},
        {'min_angle': 1.0, 'max_angle': 1.0},
        {'min_angle': -4.0, 'max_angle': 4.0},
        {'max_angle': math.nan},
        {'range_min': 0.0},
        {'range_min': 5.0, 'range_max': 5.0},
    ],
)
def test_invalid_lidar_settings_are_rejected(change: dict[str, float | str]) -> None:
    valid = GazeboLidarConfig('base_link', 'laser', '/scan')
    with pytest.raises(ConfigurationError):
        replace(valid, **change)


def test_lidar_on_missing_link_fails_before_launch(
    config: SimulationConfig, tmp_path: Path
) -> None:
    lidar = GazeboLidarConfig('absent_link', 'laser', '/scan')
    with pytest.raises(ConfigurationError) as caught:
        prepare_assets(
            config, tmp_path / 'prepared', GazeboAssetConfig('/private/run_test/cmd_vel', lidar)
        )
    assert caught.value.parameter == 'lidar.link_name'
    assert not (tmp_path / 'prepared').exists()
