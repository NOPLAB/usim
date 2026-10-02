"""Create portable differential-drive robot descriptions from primitive geometry."""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass


class RobotGeometryError(ValueError):
    """Identify an invalid robot parameter at the configuration boundary."""

    field: str
    constraint: str

    def __init__(self, field: str, constraint: str) -> None:
        self.field = field
        self.constraint = constraint
        super().__init__(f'{field}: {constraint}')


@dataclass(frozen=True, slots=True)
class MobileRobot:
    """Physical dimensions for a two-wheel robot with a passive spherical caster."""

    name: str = 'mobile_robot'
    wheel_radius: float = 0.08
    wheel_separation: float = 0.32
    wheel_width: float = 0.04
    body_size: tuple[float, float, float] = (0.45, 0.24, 0.10)
    mass: float = 6.0
    camera_offset: tuple[float, float, float] = (0.12, 0.0, 0.28)

    def __post_init__(self) -> None:
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', self.name):
            raise RobotGeometryError('name', 'use letters, digits, underscores or hyphens')
        dimensions = {
            'wheel_radius': self.wheel_radius,
            'wheel_separation': self.wheel_separation,
            'wheel_width': self.wheel_width,
            'mass': self.mass,
        }
        for field, value in dimensions.items():
            if not math.isfinite(value) or value <= 0:
                raise RobotGeometryError(field, 'must be finite and positive')
        if len(self.body_size) != 3 or any(
            not math.isfinite(value) or value <= 0 for value in self.body_size
        ):
            raise RobotGeometryError('body_size', 'requires three finite positive dimensions')
        if len(self.camera_offset) != 3 or any(
            not math.isfinite(value) for value in self.camera_offset
        ):
            raise RobotGeometryError('camera_offset', 'requires three finite coordinates')
        if self.wheel_separation < self.body_size[1] + self.wheel_width:
            raise RobotGeometryError('wheel_separation', 'wheels must clear the chassis width')


def render_robot(configuration: MobileRobot) -> str:
    """Render a mesh-free URDF with proper mass, inertia and wheel collision geometry."""
    robot = ET.Element('robot', name=configuration.name)
    length, width, height = configuration.body_size
    radius = configuration.wheel_radius
    wheel_width = configuration.wheel_width
    body_mass = configuration.mass * 0.8
    wheel_mass = configuration.mass * 0.09
    caster_mass = configuration.mass * 0.02
    caster_radius = min(radius / 3, 0.03)
    body_inertia = tuple(
        body_mass * value / 12
        for value in (width**2 + height**2, length**2 + height**2, length**2 + width**2)
    )
    transverse = wheel_mass * (3 * radius**2 + wheel_width**2) / 12
    wheel_inertia = (transverse, wheel_mass * radius**2 / 2, transverse)
    caster_inertia = (2 * caster_mass * caster_radius**2 / 5,) * 3
    body_center = (0.0, 0.0, radius + height / 2)
    # The body mass sits between axle and caster; above the axle the chassis rocks forward.
    body_mass_center = (-length / 6, 0.0, radius + height / 2)
    specifications = [
        (
            'base_link',
            'box',
            {'size': f'{length} {width} {height}'},
            body_center,
            body_mass_center,
            '0 0 0',
            body_mass,
            body_inertia,
        ),
        (
            'left_wheel_link',
            'cylinder',
            {'radius': str(radius), 'length': str(wheel_width)},
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            f'{math.pi / 2} 0 0',
            wheel_mass,
            wheel_inertia,
        ),
        (
            'right_wheel_link',
            'cylinder',
            {'radius': str(radius), 'length': str(wheel_width)},
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            f'{math.pi / 2} 0 0',
            wheel_mass,
            wheel_inertia,
        ),
        (
            'caster_link',
            'sphere',
            {'radius': str(caster_radius)},
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            '0 0 0',
            caster_mass,
            caster_inertia,
        ),
    ]
    for name, kind, attributes, offset, center, rotation, mass, inertia in specifications:
        link = ET.SubElement(robot, 'link', name=name)
        xyz = ' '.join(str(value) for value in offset)
        inertial = ET.SubElement(link, 'inertial')
        _ = ET.SubElement(
            inertial, 'origin', xyz=' '.join(str(value) for value in center), rpy='0 0 0'
        )
        _ = ET.SubElement(inertial, 'mass', value=str(mass))
        _ = ET.SubElement(
            inertial,
            'inertia',
            ixx=str(inertia[0]),
            iyy=str(inertia[1]),
            izz=str(inertia[2]),
            ixy='0',
            ixz='0',
            iyz='0',
        )
        for role in ('visual', 'collision'):
            element = ET.SubElement(link, role)
            if role == 'collision' and kind == 'cylinder':
                # Isaac's imported cylinder colliders sink and hop on the ground plane; a
                # radius-matched sphere rolls identically in both engines.
                _ = ET.SubElement(element, 'origin', xyz=xyz, rpy='0 0 0')
                geometry = ET.SubElement(element, 'geometry')
                _ = ET.SubElement(geometry, 'sphere', radius=attributes['radius'])
                continue
            _ = ET.SubElement(element, 'origin', xyz=xyz, rpy=rotation)
            _ = ET.SubElement(ET.SubElement(element, 'geometry'), kind, attributes)
            if role == 'visual':
                material = ET.SubElement(element, 'material', name=name + '_color')
                _ = ET.SubElement(material, 'color', rgba='0.15 0.35 0.65 1')
    for side, sign in (('left', 1), ('right', -1)):
        joint = ET.SubElement(robot, 'joint', name=f'{side}_wheel_joint', type='continuous')
        _ = ET.SubElement(joint, 'parent', link='base_link')
        _ = ET.SubElement(joint, 'child', link=f'{side}_wheel_link')
        _ = ET.SubElement(
            joint,
            'origin',
            xyz=f'0 {sign * configuration.wheel_separation / 2} {radius}',
            rpy='0 0 0',
        )
        _ = ET.SubElement(joint, 'axis', xyz='0 1 0')
        _ = ET.SubElement(joint, 'limit', effort='100', velocity='100')
        _ = ET.SubElement(joint, 'dynamics', damping='0.0', friction='0.0')
    caster = ET.SubElement(robot, 'joint', name='caster_joint', type='fixed')
    _ = ET.SubElement(caster, 'parent', link='base_link')
    _ = ET.SubElement(caster, 'child', link='caster_link')
    _ = ET.SubElement(caster, 'origin', xyz=f'{-length / 3} 0 {caster_radius}', rpy='0 0 0')
    _ = ET.SubElement(robot, 'link', name='camera_link')
    camera = ET.SubElement(robot, 'joint', name='camera_joint', type='fixed')
    _ = ET.SubElement(camera, 'parent', link='base_link')
    _ = ET.SubElement(camera, 'child', link='camera_link')
    _ = ET.SubElement(
        camera,
        'origin',
        xyz=' '.join(str(value) for value in configuration.camera_offset),
        rpy='0 0 0',
    )
    ET.indent(robot, space='  ')
    return '<?xml version="1.0"?>\n' + ET.tostring(robot, encoding='unicode') + '\n'
