"""Native raycast sensors: 360-degree azimuth, native 60-degree elevation.

A single elevation row is planar. Results use sensor-local XYZ and meters,
with image-like (environment, elevation, azimuth) indexing. Values come from
the native sensor cache at the last completed physics update, never an implicit step.
"""

from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np
from numpy.typing import NDArray

from usim.facets import SensorLayout
from usim.model import ModelError
from usim.specs import SensorSpec

from .native import numpy


@dataclass(frozen=True, slots=True)
class LidarFrame:
    points: NDArray[np.generic]
    distances: NDArray[np.generic]
    frame: str
    step: int


class BoundLidar:
    """Sample the native raycast cache at the declared sensor rate."""

    def __init__(self, native: Any, spec: SensorSpec, owner: str):
        self.native = native
        self.spec = spec
        self.frame = f'{owner}/{spec.name}'
        self._sample = -1
        self._cached: LidarFrame | None = None

    @property
    def layouts(self) -> tuple[SensorLayout, ...]:
        shape = (self.spec.height, self.spec.width)
        return (
            SensorLayout(
                f'{self.spec.name}.points',
                'lidar_points',
                shape + (3,),
                'float32',
                'm',
                self.frame,
            ),
            SensorLayout(
                f'{self.spec.name}.distances', 'lidar_ranges', shape, 'float32', 'm', self.frame
            ),
        )

    def invalidate(self) -> None:
        self._sample = -1
        self._cached = None

    def read(self, clock: tuple[int, float], n_envs: int) -> LidarFrame:
        step, rate = clock
        sample = int(step * self.spec.rate / rate)
        if self._cached is None or sample != self._sample:
            values = self.native.read()
            shape = (n_envs, self.spec.width, self.spec.height)
            self._cached = LidarFrame(
                numpy(values.points)
                .astype(np.float32)
                .reshape(shape + (3,))
                .transpose(0, 2, 1, 3),
                numpy(values.distances).astype(np.float32).reshape(shape).transpose(0, 2, 1),
                self.frame,
                step,
            )
            self._sample = sample
        return self._cached


def add_lidar(scene: Any, entity: Any, declaration: tuple[str, SensorSpec]) -> BoundLidar:
    owner, spec = declaration
    gs = import_module('genesis')
    link = entity.get_link(spec.parent) if spec.parent not in {None, 'world'} else entity.base_link
    if link is None:
        raise ModelError(f'Unknown lidar attachment link: {spec.parent}')
    x, y, z, w = spec.pose.orientation_xyzw
    euler = gs.utils.geom.quat_to_xyz(np.asarray((w, x, y, z)), rpy=True, degrees=True)
    native = scene.add_sensor(
        gs.sensors.Lidar(
            entity_idx=-1 if spec.parent == 'world' else entity.idx,
            link_idx_local=link.idx_local,
            pos_offset=spec.pose.position,
            euler_offset=tuple(euler),
            return_world_frame=False,
            pattern=gs.sensors.SphericalPattern(
                n_points=(spec.width, spec.height),
                fov=(360.0, 0.0 if spec.height == 1 else 60.0),
            ),
        )
    )
    return BoundLidar(native, spec, owner)
