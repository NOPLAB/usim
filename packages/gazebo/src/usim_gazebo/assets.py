"""Prepare private Gazebo assets without changing the caller's descriptions."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol
from urllib.parse import unquote
from urllib.request import url2pathname

from usim.simulation import ConfigurationError, SimulationConfig


class LidarAssetConfig(Protocol):
    """Lidar fields consumed while writing a Gazebo sensor asset."""

    @property
    def link_name(self) -> str: ...

    @property
    def frame_name(self) -> str: ...

    @property
    def topic(self) -> str: ...

    @property
    def update_rate(self) -> float: ...

    @property
    def horizontal_samples(self) -> int: ...

    @property
    def min_angle(self) -> float: ...

    @property
    def max_angle(self) -> float: ...

    @property
    def range_min(self) -> float: ...

    @property
    def range_max(self) -> float: ...


@dataclass(frozen=True, slots=True)
class GazeboAssetConfig:
    """Private drive topic and optional lidar used while staging Gazebo assets."""

    private_topic: str
    lidar: LidarAssetConfig | None = None


def _parse(path: Path, root: str) -> ET.Element:
    try:
        element = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as error:
        raise ValueError(f'cannot read {root} asset {path}: {error}') from error
    if element.tag != root:
        raise ValueError(f'{path}: expected <{root}>, got <{element.tag}>')
    return element


def _package_path(uri: str, source: Path) -> tuple[Path, Path]:
    package, separator, relative = uri.removeprefix('package://').partition('/')
    if not separator or not package or '..' in PurePosixPath(relative).parts:
        raise ValueError(f'invalid package URI: {uri}')
    candidates = []
    for parent in source.parents:
        manifest = parent / 'package.xml'
        if manifest.is_file() and ET.parse(manifest).findtext('name') == package:
            candidates.append(parent)
    for prefix in os.environ.get('AMENT_PREFIX_PATH', '').split(os.pathsep):
        if prefix:
            candidates.append(Path(prefix) / 'share' / package)
    for prefix in os.environ.get('ROS_PACKAGE_PATH', '').split(os.pathsep):
        if prefix:
            folder = Path(prefix)
            candidates.append(folder / package)
            manifest = folder / 'package.xml'
            if folder.name == package or (
                manifest.is_file() and ET.parse(manifest).findtext('name') == package
            ):
                candidates.append(folder)
    for candidate in candidates:
        path = candidate / relative
        if path.exists():
            return path.resolve(), candidate.resolve()
    raise ValueError(
        f'unresolved {uri} in {source}; supply the package via '
        'AMENT_PREFIX_PATH or ROS_PACKAGE_PATH, or use a relative mesh path'
    )


def prepare_assets(
    configuration: SimulationConfig,
    directory: Path,
    asset_config: GazeboAssetConfig,
) -> list[tuple[Path, str]]:
    """Write temporary URDF/SDF and return explicit read-only resource mounts.

    Supplied robot plugins are removed so only the gated controller can drive;
    an optional lidar is injected afterwards on its configured link.
    Geometry, inertia, friction and other extensions survive. Mesh folders stay
    intact, including adjacent texture resources.
    """
    world_path = configuration.world.resolve()
    robot_path = configuration.robot_urdf.resolve()
    private_topic = asset_config.private_topic
    lidar = asset_config.lidar
    world = _parse(world_path, 'sdf')
    if len(world.findall('world')) != 1:
        raise ValueError(f'{world_path}: expected exactly one SDF world')
    robot = _parse(robot_path, 'robot')
    links = {link.get('name') for link in robot.findall('link')}
    joints = {joint.get('name'): joint for joint in robot.findall('joint')}
    if configuration.base_link not in links:
        raise ValueError(f'base link not found: {configuration.base_link}')
    if lidar is not None and lidar.link_name not in links:
        raise ConfigurationError('lidar.link_name')
    for name in configuration.left_wheel_joints + configuration.right_wheel_joints:
        joint = joints.get(name)
        if (
            joint is None
            or sum(item.get('name') == name for item in robot.findall('joint')) != 1
            or joint.get('type') not in ('continuous', 'revolute')
        ):
            raise ValueError(f'wheel joint must exist and rotate: {name}')
        for end in ('parent', 'child'):
            reference = joint.find(end)
            if reference is None or reference.get('link') not in links:
                raise ValueError(f'wheel joint {name}: missing {end} link')
    if not private_topic.startswith('/') or (
        configuration.ros and private_topic == configuration.ros.cmd_vel_topic
    ):
        raise ValueError('drive requires an absolute private command topic')

    mounts: dict[Path, str] = {}

    def mount(folder: Path) -> str:
        folder = folder.resolve()
        if folder not in mounts:
            mounts[folder] = f'/assets/{len(mounts)}'
        return mounts[folder]

    mount(world_path.parent)

    def resource(value: str, source: Path) -> str:
        value = unquote(value.strip())
        if value.startswith('model://'):
            if source == robot_path:
                raise ValueError(f'URDF mesh must be a file or package URI: {value}')
            model_path = PurePosixPath(value.removeprefix('model://'))
            if (
                not model_path.parts
                or model_path.is_absolute()
                or any(part in {'.', '..'} or ':' in part for part in model_path.parts)
            ):
                raise ValueError(f'invalid model URI in {source}: {value}')
            model_name = model_path.parts[0]
            folder = next(
                (
                    candidate
                    for parent in source.parents
                    for candidate in (parent / 'models', parent)
                    if (candidate / model_name).is_dir()
                    and (candidate / model_name / 'model.config').is_file()
                ),
                None,
            )
            if folder is not None:
                mount(folder)
            return value
        if value.startswith('package://'):
            path, folder = _package_path(value, source)
        else:
            if value.startswith('file://'):
                value = url2pathname(value.removeprefix('file://'))
            if '://' in value:
                raise ValueError(f'unsupported asset URI in {source}: {value}')
            path = Path(value)
            if not path.is_absolute():
                path = (source.parent / path).resolve()
                folder = Path(os.path.commonpath((source.parent, path.parent)))
                if folder == Path(folder.anchor):
                    raise ValueError(f'asset references must not require a root mount: {value}')
            else:
                path = path.resolve()
                folder = path.parent
        if not path.exists() or (source == robot_path and not path.is_file()):
            raise ValueError(f'unresolved asset in {source}: {value}')
        return 'file://' + mount(folder) + '/' + path.relative_to(folder).as_posix()

    for parent in robot.iter():
        for plugin in list(parent.findall('plugin')):
            parent.remove(plugin)
    for element in robot.iter():
        if element.tag in ('mesh', 'texture') and 'filename' in element.attrib:
            element.set('filename', resource(element.attrib['filename'], robot_path))
    for element in world.iter('uri'):
        if element.text:
            element.text = resource(element.text, world_path)

    ros = configuration.ros
    if ros is not None:
        plugin = ET.SubElement(
            ET.SubElement(robot, 'gazebo'),
            'plugin',
            name='usim_drive',
            filename='libgazebo_ros_diff_drive.so',
        )
        ros_element = ET.SubElement(plugin, 'ros')
        ET.SubElement(ros_element, 'namespace').text = '/'
        ET.SubElement(ros_element, 'remapping').text = f'cmd_vel:={private_topic}'
        ET.SubElement(ros_element, 'remapping').text = f'odom:={ros.odom_topic}'
        ET.SubElement(plugin, 'num_wheel_pairs').text = str(len(configuration.left_wheel_joints))
        for left, right in zip(configuration.left_wheel_joints, configuration.right_wheel_joints):
            for name, value in {
                'left_joint': left,
                'right_joint': right,
                'wheel_diameter': 2 * configuration.wheel_radius,
                'wheel_separation': configuration.wheel_separation,
            }.items():
                ET.SubElement(plugin, name).text = str(value)
        for name, value in {
            'max_wheel_torque': 20,
            'max_wheel_acceleration': 10,
            'update_rate': 100,
            'odometry_source': 1,
            'odometry_frame': 'odom',
            'robot_base_frame': configuration.base_link,
            'publish_odom': 'true',
            'publish_odom_tf': 'true',
            'publish_wheel_tf': 'false',
        }.items():
            ET.SubElement(plugin, name).text = str(value)

    if configuration.camera_enabled:
        _camera(robot, configuration)
    if lidar is not None:
        _lidar(robot, lidar, ros_enabled=ros is not None)
    directory.mkdir(parents=True, exist_ok=True)
    for root, filename in ((world, 'world.sdf'), (robot, 'robot.urdf')):
        ET.indent(root, space='  ')
        ET.ElementTree(root).write(directory / filename, encoding='utf-8', xml_declaration=False)
    return list(mounts.items())


def _lidar(robot: ET.Element, lidar: LidarAssetConfig, *, ros_enabled: bool) -> None:
    gazebo = ET.SubElement(robot, 'gazebo', reference=lidar.link_name)
    sensor = ET.SubElement(gazebo, 'sensor', name='usim_lidar', type='ray')
    ET.SubElement(sensor, 'always_on').text = 'true'
    ET.SubElement(sensor, 'visualize').text = 'false'
    ET.SubElement(sensor, 'update_rate').text = str(lidar.update_rate)
    ray = ET.SubElement(sensor, 'ray')
    horizontal = ET.SubElement(ET.SubElement(ray, 'scan'), 'horizontal')
    ET.SubElement(horizontal, 'samples').text = str(lidar.horizontal_samples)
    ET.SubElement(horizontal, 'resolution').text = '1'
    ET.SubElement(horizontal, 'min_angle').text = str(lidar.min_angle)
    ET.SubElement(horizontal, 'max_angle').text = str(lidar.max_angle)
    range_element = ET.SubElement(ray, 'range')
    ET.SubElement(range_element, 'min').text = str(lidar.range_min)
    ET.SubElement(range_element, 'max').text = str(lidar.range_max)
    ET.SubElement(range_element, 'resolution').text = '0.01'
    if ros_enabled:
        plugin = ET.SubElement(
            sensor, 'plugin', name='usim_lidar_ros', filename='libgazebo_ros_ray_sensor.so'
        )
        ros_element = ET.SubElement(plugin, 'ros')
        ET.SubElement(ros_element, 'namespace').text = '/'
        ET.SubElement(ros_element, 'remapping').text = f'~/out:={lidar.topic}'
        ET.SubElement(plugin, 'output_type').text = 'sensor_msgs/LaserScan'
        ET.SubElement(plugin, 'frame_name').text = lidar.frame_name


def _camera(robot: ET.Element, configuration: SimulationConfig) -> None:
    gazebo = ET.SubElement(robot, 'gazebo', reference=configuration.base_link)
    sensor = ET.SubElement(gazebo, 'sensor', name='usim_camera', type='depth')
    ET.SubElement(sensor, 'pose').text = (
        ' '.join(str(value) for value in configuration.camera_offset) + ' 0 0 0'
    )
    ET.SubElement(sensor, 'always_on').text = 'true'
    ET.SubElement(sensor, 'update_rate').text = str(configuration.camera_hz)
    camera = ET.SubElement(sensor, 'camera', name='usim_camera')
    ET.SubElement(camera, 'horizontal_fov').text = '1.0471975512'
    image = ET.SubElement(camera, 'image')
    ET.SubElement(image, 'width').text = str(configuration.camera_width)
    ET.SubElement(image, 'height').text = str(configuration.camera_height)
    ET.SubElement(image, 'format').text = 'R8G8B8'
    clip = ET.SubElement(camera, 'clip')
    ET.SubElement(clip, 'near').text = '0.05'
    ET.SubElement(clip, 'far').text = '10'
    ros = configuration.ros
    if ros is not None:
        plugin = ET.SubElement(
            sensor, 'plugin', name='usim_camera_ros', filename='libgazebo_ros_camera.so'
        )
        ros_element = ET.SubElement(plugin, 'ros')
        ET.SubElement(ros_element, 'namespace').text = '/'
        ET.SubElement(ros_element, 'remapping').text = f'usim_camera/image_raw:={ros.rgb_topic}'
        ET.SubElement(
            ros_element, 'remapping'
        ).text = f'usim_camera/depth/image_raw:={ros.depth_topic}'
        ET.SubElement(plugin, 'camera_name').text = 'usim_camera'
        ET.SubElement(plugin, 'frame_name').text = 'usim_camera_optical_frame'
        ET.SubElement(plugin, 'min_depth').text = '0.05'
        ET.SubElement(plugin, 'max_depth').text = '10'
