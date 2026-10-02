"""Portable asset regressions; real simulator smoke lives in scripts/smoke_mobile.py."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import threading
import xml.etree.ElementTree as ET

import pytest

from usim.ports.gazebo import GazeboSimulator
from usim.ports.gazebo.assets import prepare_assets
from usim.ports.gazebo.runner import load_configuration
from usim.robot import MobileRobot, render_robot
from usim.simulation import Ros2Config, SimulationConfig


@pytest.fixture
def config(tmp_path: Path) -> SimulationConfig:
    robot = tmp_path / 'robot.urdf'
    robot.write_text(render_robot(MobileRobot()), encoding='utf-8')
    world = tmp_path / 'world.sdf'
    world.write_text('<sdf version="1.6"><world name="custom"/></sdf>', encoding='utf-8')
    return SimulationConfig(world, robot)


def prepared(
    config: SimulationConfig, directory: Path
) -> tuple[ET.Element, list[tuple[Path, str]]]:
    mounts = prepare_assets(config, directory, '/private/run_test/cmd_vel')
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


def test_relative_mesh_and_adjacent_texture_mounts_are_real(
    config: SimulationConfig, tmp_path: Path
) -> None:
    mesh_dir = tmp_path / 'geometry'
    mesh_dir.mkdir()
    mesh = mesh_dir / 'body.stl'
    mesh.write_text('solid body\nendsolid body\n')
    (tmp_path / 'textures').mkdir()
    (tmp_path / 'textures' / 'color.png').write_bytes(b'fixture')
    root = ET.parse(config.robot_urdf).getroot()
    geometry = root.findall('link/visual/geometry')[0]
    geometry.clear()
    ET.SubElement(geometry, 'mesh', filename='geometry/body.stl')
    ET.ElementTree(root).write(config.robot_urdf)
    robot, mounts = prepared(config, tmp_path / 'prepared')
    filename = robot.findall('link/visual/geometry/mesh')[0].get('filename')
    host, target = next((host, target) for host, target in mounts if host == tmp_path)
    assert filename == f'file://{target}/geometry/body.stl'
    assert (host / 'geometry' / '..' / 'textures' / 'color.png').is_file()
    assert (host / 'geometry' / 'body.stl').read_bytes() == mesh.read_bytes()


@pytest.mark.parametrize(
    'change', ['base', 'joint', 'joint_type', 'mesh', 'package', 'xml', 'world']
)
def test_malformed_or_unresolved_assets_fail_before_launch(
    config: SimulationConfig, tmp_path: Path, change: str
) -> None:
    if change == 'base':
        config = replace(config, base_link='absent')
    elif change == 'joint':
        config = replace(config, left_joint='absent')
    elif change == 'joint_type':
        config.robot_urdf.write_text(config.robot_urdf.read_text().replace('continuous', 'fixed'))
    elif change in ('mesh', 'package'):
        root = ET.parse(config.robot_urdf).getroot()
        ET.SubElement(
            root.findall('link/visual/geometry')[0],
            'mesh',
            filename=('missing.stl' if change == 'mesh' else 'package://not_installed/mesh.stl'),
        )
        ET.ElementTree(root).write(config.robot_urdf)
    elif change == 'xml':
        config.robot_urdf.write_text('<robot')
    else:
        config.world.write_text('<sdf><model name="not_a_world"/></sdf>')
    with pytest.raises(ValueError):
        prepared(config, tmp_path / 'prepared')
    assert not (tmp_path / 'prepared').exists()


def test_package_mesh_resolves_from_own_manifest(config: SimulationConfig, tmp_path: Path) -> None:
    (tmp_path / 'package.xml').write_text('<package><name>my_robot</name></package>')
    (tmp_path / 'mesh.stl').write_text('solid x\nendsolid x\n')
    root = ET.parse(config.robot_urdf).getroot()
    ET.SubElement(
        root.findall('link/visual/geometry')[0], 'mesh', filename='package://my_robot/mesh.stl'
    )
    ET.ElementTree(root).write(config.robot_urdf)
    robot, mounts = prepared(config, tmp_path / 'prepared')
    mesh = robot.findall('link/visual/geometry/mesh')[0]
    assert mesh.get('filename') == 'file:///assets/0/mesh.stl'
    assert mounts == [(tmp_path, '/assets/0')]


def test_public_topic_cannot_be_used_as_private_drive(
    config: SimulationConfig, tmp_path: Path
) -> None:
    assert config.ros is not None
    with pytest.raises(ValueError):
        prepare_assets(config, tmp_path / 'prepared', config.ros.cmd_vel_topic)


def test_native_config_roundtrip_and_precancel(config: SimulationConfig, tmp_path: Path) -> None:
    fields = asdict(replace(config, ros=None))
    fields['world'], fields['robot_urdf'] = str(config.world), str(config.robot_urdf)
    path = tmp_path / 'config.json'
    path.write_text(
        json.dumps(
            {
                'configuration': fields,
                'private_topic': '/private/run/cmd_vel',
                'smoke_output': None,
            }
        )
    )
    loaded, topic, output = load_configuration(path)
    assert loaded == replace(config, ros=None)
    assert topic == '/private/run/cmd_vel'
    assert output is None
    stop = threading.Event()
    stop.set()
    GazeboSimulator(image='not-an-image').run(loaded, stop=stop)


@pytest.mark.parametrize('cancelled', [True, False])
def test_unexecuted_smoke_cannot_leave_stale_success(
    config: SimulationConfig, tmp_path: Path, cancelled: bool
) -> None:
    output = tmp_path / 'output'
    output.mkdir()
    report = output / 'smoke.json'
    report.write_text(json.dumps({'passed': True}))
    if cancelled:
        stop = threading.Event()
        stop.set()
        GazeboSimulator().smoke(config, output, stop=stop)
    else:
        with pytest.raises(ValueError):
            GazeboSimulator().smoke(replace(config, robot_urdf=tmp_path / 'absent'), output)
    assert not report.exists()


def test_unrelated_ros_package_directory_cannot_resolve_wrong_package(
    config: SimulationConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / 'mesh.stl').write_text('solid x')
    monkeypatch.setenv('ROS_PACKAGE_PATH', str(tmp_path))
    monkeypatch.delenv('AMENT_PREFIX_PATH', raising=False)
    root = ET.parse(config.robot_urdf).getroot()
    ET.SubElement(
        root.findall('link/visual/geometry')[0],
        'mesh',
        filename='package://wrong_package/mesh.stl',
    )
    ET.ElementTree(root).write(config.robot_urdf)
    with pytest.raises(ValueError):
        prepared(config, tmp_path / 'prepared')
