"""Native Gazebo sensor attachments and typed image/scan decoding."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from math import asin, atan2
from uuid import uuid4
import xml.etree.ElementTree as ET

from usim.facets import SensorLayout
from usim.model import ModelError
from usim.specs import SensorSpec
from usim_gazebo.assets import _lidar
from usim_gazebo.models import GazeboModel
from usim_gazebo.transport import Json
from usim_gazebo import GazeboLidarConfig


@dataclass(frozen=True, slots=True)
class SensorChannel:
    layout: SensorLayout
    topic: str
    message_type: str


def attach(
    model: GazeboModel,
    specs: tuple[SensorSpec, ...],
) -> tuple[str, tuple[SensorChannel, ...]]:
    """Add only owned ROS sensor plugins after authored plugins have been assessed."""
    root = ET.fromstring(model.xml)
    channels = []
    for spec in specs:
        parent = spec.parent or model.links[0]
        if parent not in model.links:
            raise ModelError(f'Unknown sensor parent link: {parent}')
        name = 'usim_sensor_' + uuid4().hex
        topic = '/' + name
        frame_name = name if spec.kind == 'lidar' else name + '_optical'
        x, y, z, w = spec.pose.orientation_xyzw
        roll = atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
        yaw = atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        pose = ' '.join(str(value) for value in (*spec.pose.position, roll, pitch, yaw))
        # The shared ray template is exercised by the legacy lidar tests.
        if spec.kind == 'lidar':
            temporary = ET.Element('robot')
            _lidar(
                temporary,
                GazeboLidarConfig(
                    parent,
                    frame_name,
                    topic,
                    update_rate=spec.rate,
                    horizontal_samples=spec.width,
                ),
                ros_enabled=True,
            )
            sensor = temporary.find('gazebo/sensor')
            if sensor is None:
                raise ModelError('Lidar template did not produce a sensor')
            sensor.set('name', name)
            ET.SubElement(sensor, 'pose').text = pose
            channels.append(
                SensorChannel(
                    SensorLayout(spec.name, 'lidar', (spec.width,), 'float32', 'm', frame_name),
                    topic,
                    'LaserScan',
                )
            )
        else:
            sensor = ET.Element(
                'sensor', name=name, type='depth' if spec.kind != 'rgb' else 'camera'
            )
            ET.SubElement(sensor, 'pose').text = pose
            ET.SubElement(sensor, 'always_on').text = 'true'
            ET.SubElement(sensor, 'update_rate').text = str(spec.rate)
            camera = ET.SubElement(sensor, 'camera', name=name)
            ET.SubElement(camera, 'horizontal_fov').text = '1.0471975512'
            image = ET.SubElement(camera, 'image')
            ET.SubElement(image, 'width').text = str(spec.width)
            ET.SubElement(image, 'height').text = str(spec.height)
            ET.SubElement(image, 'format').text = 'R8G8B8'
            clip = ET.SubElement(camera, 'clip')
            ET.SubElement(clip, 'near').text = '0.05'
            ET.SubElement(clip, 'far').text = '10'
            plugin = ET.SubElement(
                sensor,
                'plugin',
                name=name + '_ros',
                filename='libgazebo_ros_camera.so',
            )
            ros = ET.SubElement(plugin, 'ros')
            ET.SubElement(ros, 'namespace').text = '/'
            ET.SubElement(plugin, 'camera_name').text = name
            ET.SubElement(plugin, 'frame_name').text = frame_name
            ET.SubElement(ros, 'remapping').text = f'{name}/image_raw:={topic}/rgb'
            ET.SubElement(ros, 'remapping').text = f'{name}/depth/image_raw:={topic}/depth'
            if spec.kind in {'rgb', 'rgbd'}:
                channels.append(
                    SensorChannel(
                        SensorLayout(
                            spec.name if spec.kind == 'rgb' else spec.name + '/rgb',
                            'rgb',
                            (spec.height, spec.width, 3),
                            'uint8',
                            'rgb',
                            frame_name,
                        ),
                        topic + '/rgb',
                        'Image',
                    )
                )
            if spec.kind in {'depth', 'rgbd'}:
                channels.append(
                    SensorChannel(
                        SensorLayout(
                            spec.name if spec.kind == 'depth' else spec.name + '/depth',
                            'depth',
                            (spec.height, spec.width),
                            'float32',
                            'm',
                            frame_name,
                        ),
                        topic + '/depth',
                        'Image',
                    )
                )
        if root.tag == 'robot':
            ET.SubElement(root, 'gazebo', reference=parent).append(sensor)
        else:
            model_element = root.find('model')
            if model_element is None:
                raise ModelError('Sensors require an entity model')
            link = next(
                item for item in model_element.findall('link') if item.get('name') == parent
            )
            link.append(sensor)
    return ET.tostring(root, encoding='unicode'), tuple(channels)


def decode(channel: SensorChannel, response: dict[str, Json]):
    """Decode native bytes, preserving row padding and depth units rather than resizing."""
    import numpy as np

    layout = channel.layout
    if layout.kind == 'lidar':
        ranges = response.get('ranges')
        if not isinstance(ranges, list) or len(ranges) != layout.shape[0]:
            raise ModelError('Native scan dimensions differ from the requested sensor')
        parsed: list[float] = []
        for value in ranges:
            match value:
                case None:
                    parsed.append(float('inf'))
                case bool():
                    raise ModelError('A scan range must be numeric or null')
                case int() | float():
                    parsed.append(float(value))
                case _:
                    raise ModelError('A scan range must be numeric or null')
        return np.asarray(parsed, dtype=np.float32)
    encoding = response.get('encoding')
    formats = {'rgb8': ('u1', 3), 'bgr8': ('u1', 3), '32FC1': ('f4', 1), '16UC1': ('u2', 1)}
    if not isinstance(encoding, str) or encoding not in formats:
        raise ModelError(f'Unsupported native image encoding: {encoding}')
    dtype, components = formats[encoding]
    if (layout.kind == 'rgb') != (components == 3):
        raise ModelError('Native image encoding does not match the requested sensor kind')
    height, width, stride, data = (
        response.get('height'),
        response.get('width'),
        response.get('step'),
        response.get('data'),
    )
    if (
        (height, width) != layout.shape[:2]
        or not isinstance(stride, int)
        or not isinstance(data, str)
    ):
        raise ModelError('Native image dimensions differ from the requested sensor')
    native_dtype = np.dtype(('>' if response.get('is_bigendian') else '<') + dtype)
    row_bytes = layout.shape[1] * components * native_dtype.itemsize
    raw = base64.b64decode(data, validate=True)
    if stride < row_bytes or len(raw) != stride * layout.shape[0]:
        raise ModelError('Native image has invalid row stride or payload length')
    packed = b''.join(raw[offset : offset + row_bytes] for offset in range(0, len(raw), stride))
    array = np.frombuffer(packed, dtype=native_dtype).reshape(layout.shape).copy()
    if encoding == 'bgr8':
        array = array[..., ::-1].copy()
    if encoding == '16UC1':
        array = array.astype(np.float32) * 0.001
    return array
