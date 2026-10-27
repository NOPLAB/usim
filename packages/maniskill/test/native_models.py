"""In-memory native model builders, including static/dynamic component distinctions."""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import numpy as np


@dataclass(frozen=True, slots=True)
class NativeFailure(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


class NativePose:
    def __init__(self, p=(0, 0, 0), q=(1, 0, 0, 0)):
        self.raw_pose = np.asarray([tuple(p) + tuple(q)], dtype=np.float32)


class Joint:
    def __init__(self, name):
        self.name = name
        self.type = 'revolute'
        self.friction = 0

    def set_friction(self, value):
        self.friction = value


class Robot:
    def __init__(self, name, names, links, count, fixed, pose):
        self.name = name
        self.joints = [Joint(name) for name in names]
        self.links_map = {name: SimpleNamespace(name=name) for name in links}
        self.qpos = np.zeros((count, len(names)), dtype=np.float32)
        self.qvel = self.qpos.copy()
        self.qf = self.qpos.copy()
        self.fixed_root_link = np.full(count, fixed)
        self.pose = pose
        self.targets = {}

    def get_active_joints(self):
        return self.joints

    def get_links(self):
        return list(self.links_map.values())

    def set_pose(self, pose):
        self.pose = pose

    def get_qpos(self):
        return self.qpos

    def get_qvel(self):
        return self.qvel

    def set_qpos(self, value):
        self.qpos[:] = value

    def set_qvel(self, value):
        self.qvel[:] = value

    def set_qf(self, value):
        self.qf[:] = value


class Builder:
    def __init__(self, scene, names, links):
        self.scene = scene
        self.names = names
        self.links = links
        self.initial_pose = None
        self.name = ''

    def set_name(self, name):
        self.name = name

    def build(self, fix_root_link=None, name=None) -> 'Robot | Actor':
        if self.scene.fail_build:
            raise NativeFailure('native build failed')
        result = Robot(
            name or self.name,
            self.names,
            self.links,
            self.scene.count,
            fix_root_link,
            self.initial_pose,
        )
        self.scene.built.append(result)
        return result


class StaticComponent:
    __slots__ = ()


class DynamicComponent:
    __slots__ = ('mass', 'inertia')

    def __init__(self):
        self.mass = 0.0
        self.inertia = (0.0, 0.0, 0.0)


class Actor:
    def __init__(self, name, pose, component):
        self.name = name
        self.pose = pose
        self.component = component

    def set_pose(self, pose):
        self.pose = pose


class ActorBuilder(Builder):
    def __init__(self, scene, links, mass):
        super().__init__(scene, [], links)
        self._mass = mass
        self._auto_inertial = mass is None
        self.physx_body_type = 'dynamic'

    def build_dynamic(self, name):
        self.physx_body_type = 'dynamic'
        return self.build(name=name)

    def build_static(self, name):
        self.physx_body_type = 'static'
        return self.build(name=name)

    def build(self, fix_root_link=None, name=None) -> Actor:
        component = StaticComponent() if self.physx_body_type == 'static' else DynamicComponent()
        if not self._auto_inertial:
            # Model the pinned native bug: a static component cannot accept mass.
            setattr(component, 'mass', self._mass)
            setattr(component, 'inertia', (1.0, 1.0, 1.0))
        result = Actor(name or self.name, self.initial_pose, component)
        self.scene.built.append(result)
        return result


class Loader:
    def __init__(self, scene, format):
        self.scene = scene
        self.format = format
        self.name = None
        self.fix_root_link = None

    def parse(self, path, **kwargs):
        self.scene.parsed.append((self.format, Path(path), self.name, self.fix_root_link, kwargs))
        root = ET.parse(path).getroot()
        if self.format == 'urdf':
            names = [
                joint.attrib['name']
                for joint in root.findall('joint')
                if joint.get('type') != 'fixed'
            ]
            links = [link.attrib['name'] for link in root.findall('link')]
        else:
            names = [joint.attrib['name'] for joint in root.findall('.//joint')]
            links = [body.attrib['name'] for body in root.findall('.//body')]
        if not names and len(links) == 1:
            mass = root.find('link/inertial/mass')
            return {
                'articulation_builders': [],
                'actor_builders': [
                    ActorBuilder(
                        self.scene,
                        links,
                        float(mass.attrib['value']) if mass is not None else None,
                    )
                ],
                'cameras': [],
            }
        return {
            'articulation_builders': [Builder(self.scene, names, links)],
            'actor_builders': [],
            'cameras': [],
        }


class NativeScene:
    def __init__(self, count, fail_build=False):
        self.count = count
        self.fail_build = fail_build
        self.parsed = []
        self.built = []
        self.sync = []
        self.px = SimpleNamespace(
            gpu_update_articulation_kinematics=lambda: self.sync.append('kinematics')
        )

    def create_urdf_loader(self):
        return Loader(self, 'urdf')

    def create_mjcf_loader(self):
        return Loader(self, 'mjcf')

    def _gpu_apply_all(self):
        self.sync.append('apply')

    def _gpu_fetch_all(self):
        self.sync.append('fetch')


@dataclass(frozen=True, slots=True)
class ControllerConfig:
    names: list[str]

    def __init__(self, names, **kwargs):
        object.__setattr__(self, 'names', names)


class SubController:
    def __init__(self, robot, mode, config):
        self.robot = robot
        self.mode = mode
        self.config = config

    def set_action(self, action):
        self.robot.targets[self.mode] = np.asarray(action).copy()
