"""Attached RGB-D cameras and body-frame planar PhysX ray scans."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import TYPE_CHECKING, Mapping

import numpy as np
from numpy.typing import NDArray

from usim.facets import SensorLayout
from usim.model import ModelError
from usim.specs import SensorSpec
from usim_isaacsim.sim import _lidar_ranges

if TYPE_CHECKING:
    from isaacsim.core.prims import SingleXFormPrim
    from isaacsim.sensors.experimental.rtx import CameraSensor
    from usim_isaacsim.contracts import SessionView as Session


@dataclass(frozen=True, slots=True)
class LidarSettings:
    """SensorSpec.width is planar ray count; range defaults match the continuous runner."""

    lidar_samples: int
    lidar_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lidar_angle_min: float = -pi
    lidar_angle_max: float = pi
    lidar_min_range: float = 0.05
    lidar_max_range: float = 30.0


@dataclass(frozen=True, slots=True)
class AttachedSensor:
    spec: SensorSpec
    frame: str
    camera: CameraSensor | None = None
    transform: SingleXFormPrim | None = None


@dataclass(frozen=True, slots=True)
class SensorObservation:
    frames: Mapping[str, NDArray[np.generic]]
    timestamps: Mapping[str, float]
    rgb: NDArray[np.generic] | None = None
    depth: NDArray[np.generic] | None = None


class Sensors:
    """Mutable rate-limited sensor snapshots; reads render only with zero physics delta."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.attached: dict[str, tuple[AttachedSensor, ...]] = {}
        self._frames: dict[tuple[str, str], tuple[float, dict[str, NDArray[np.generic]]]] = {}

    def build(self) -> None:
        from isaacsim.core.prims import SingleXFormPrim
        from pxr import Usd, UsdGeom

        stage = self.session._native.stage
        assert stage is not None
        camera_created = False
        for name, agent in self.session._agents.items():
            entry = self.session._native.entities[name]
            attached = []
            root = stage.GetPrimAtPath(entry.prim_path)
            for index, spec in enumerate(agent.sensors):
                parent = entry.prim_path
                if spec.parent is not None:
                    matches = [
                        str(prim.GetPath())
                        for prim in Usd.PrimRange(root)
                        if prim.GetName() == spec.parent
                        or str(prim.GetPath()) == entry.prim_path + '/' + spec.parent
                    ]
                    if len(matches) != 1:
                        raise ModelError(f'Sensor parent is missing or ambiguous: {spec.parent}')
                    parent = matches[0]
                    if not stage.GetPrimAtPath(parent).IsA(UsdGeom.Xformable):
                        raise ModelError(
                            'Sensor parents must be transformable link or geometry prims'
                        )
                path = f'{parent}/UsimSensor_{index}'
                q = spec.pose.orientation_xyzw
                orientation = np.asarray([[q[3], *q[:3]]])
                if spec.kind == 'lidar':
                    UsdGeom.Xform.Define(stage, path)
                    transform = SingleXFormPrim(prim_path=path, name=f'{name}:{spec.name}')
                    transform.set_local_pose(
                        translation=np.asarray(spec.pose.position), orientation=orientation[0]
                    )
                    attached.append(AttachedSensor(spec, path, transform=transform))
                else:
                    from isaacsim.sensors.experimental.rtx import CameraSensor

                    UsdGeom.Camera.Define(stage, path)
                    annotators = []
                    if spec.kind in {'rgb', 'rgbd'}:
                        annotators.append('rgb')
                    if spec.kind in {'depth', 'rgbd'}:
                        annotators.append('distance_to_image_plane')
                    camera = CameraSensor(
                        path, resolution=(spec.height, spec.width), annotators=annotators
                    )
                    camera.authoring_object.set_local_poses(
                        translations=np.asarray([spec.pose.position]), orientations=orientation
                    )
                    attached.append(AttachedSensor(spec, path, camera=camera))
                    camera_created = True
            self.attached[name] = tuple(attached)
        if camera_created:
            import isaacsim.core.experimental.utils.app as app_utils

            app_utils.play(commit=True)
            world = self.session._native.world
            assert world is not None
            world.pause()

    def layouts(self, name: str) -> tuple[SensorLayout, ...]:
        layouts = []
        for sensor in self.attached.get(name, ()):
            spec = sensor.spec
            for kind, shape, dtype, unit in (
                ('rgb', (spec.height, spec.width, 3), 'uint8', 'color'),
                ('depth', (spec.height, spec.width), 'float32', 'm'),
                ('lidar', (spec.width,), 'float32', 'm'),
            ):
                if spec.kind == kind or (spec.kind == 'rgbd' and kind != 'lidar'):
                    layouts.append(
                        SensorLayout(f'{spec.name}/{kind}', kind, shape, dtype, unit, sensor.frame)
                    )
        return tuple(layouts)

    def observe(self, name: str) -> SensorObservation:
        world = self.session._native.world
        assert world is not None
        frames = {}
        stamps = {}
        first_rgb = first_depth = None
        for sensor in self.attached.get(name, ()):
            spec = sensor.spec
            key = (name, spec.name)
            now = float(world.current_time)
            cached = self._frames.get(key)
            if cached is None or now < cached[0] or now - cached[0] >= 1 / spec.rate:
                images = {}
                if sensor.camera is not None:
                    import omni.replicator.core as rep

                    rep.orchestrator.step(rt_subframes=2, pause_timeline=True, delta_time=0.0)
                    if spec.kind in {'rgb', 'rgbd'}:
                        color, _ = sensor.camera.get_data('rgb')
                        color = np.asarray(color)
                        if color.shape not in {
                            (spec.height, spec.width, 3),
                            (spec.height, spec.width, 4),
                        }:
                            raise ModelError('Isaac camera returned the wrong RGB dimensions')
                        images['rgb'] = color[..., :3].astype(np.uint8, copy=True)
                    if spec.kind in {'depth', 'rgbd'}:
                        depth, _ = sensor.camera.get_data('distance_to_image_plane')
                        images['depth'] = (
                            np.asarray(depth, dtype=np.float32)
                            .reshape(spec.height, spec.width)
                            .copy()
                        )
                else:
                    from omni.physx import get_physx_scene_query_interface

                    assert sensor.transform is not None
                    position, quaternion = sensor.transform.get_world_pose()
                    images['lidar'] = np.asarray(
                        _lidar_ranges(
                            get_physx_scene_query_interface(),
                            position,
                            quaternion,
                            LidarSettings(spec.width),
                            self.session._native.entities[name].prim_path,
                        ),
                        dtype=np.float32,
                    )
                self._frames[key] = (now, images)
            stamp, images = self._frames[key]
            for kind, image in images.items():
                frame = f'{spec.name}/{kind}'
                frames[frame] = image.copy()
                stamps[frame] = stamp
                if kind == 'rgb' and first_rgb is None:
                    first_rgb = image.copy()
                if kind == 'depth' and first_depth is None:
                    first_depth = image.copy()
        return SensorObservation(frames, stamps, first_rgb, first_depth)

    def invalidate(self) -> None:
        self._frames.clear()

    def clear(self) -> None:
        self.invalidate()
        self.attached.clear()
