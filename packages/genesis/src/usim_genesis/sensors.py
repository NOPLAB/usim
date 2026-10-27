"""Declared cameras sampled on demand at episode-clock safe points.

Sensor poses use Genesis camera coordinates: X right, Y up, and -Z forward.
An omitted parent attaches to the agent's root; ``world`` uses a fixed world pose.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from usim.facets import SensorLayout
from usim.model import ModelError
from usim.specs import SensorSpec

from .camera import render_images


CAMERA_KINDS = frozenset({'camera', 'rgb', 'rgbd', 'depth'})


@dataclass(frozen=True, slots=True)
class CameraFrame:
    rgb: NDArray[np.uint8] | None
    depth: NDArray[np.float32] | None
    frame: str


class BoundCamera:
    """Own a mutable native camera and its episode-clock sampling cache."""

    def __init__(self, camera: Any, spec: SensorSpec, owner: str):
        self.native = camera
        self.spec = spec
        self.frame = f'{owner}/{spec.name}'
        self._sample = -1
        self._cached: CameraFrame | None = None

    @property
    def layouts(self) -> tuple[SensorLayout, ...]:
        layouts = []
        if self.spec.kind != 'depth':
            layouts.append(
                SensorLayout(
                    f'{self.spec.name}.rgb',
                    'rgb',
                    (self.spec.height, self.spec.width, 3),
                    'uint8',
                    'color',
                    self.frame,
                )
            )
        if self.spec.kind in {'rgbd', 'depth'}:
            layouts.append(
                SensorLayout(
                    f'{self.spec.name}.depth',
                    'depth',
                    (self.spec.height, self.spec.width),
                    'float32',
                    'm',
                    self.frame,
                )
            )
        return tuple(layouts)

    def attach(self, entity: Any) -> None:
        x, y, z, w = self.spec.pose.orientation_xyzw
        transform = np.eye(4)
        transform[:3, :3] = (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
        transform[:3, 3] = self.spec.pose.position
        if self.spec.parent == 'world':
            self.native.set_pose(transform=transform)
        else:
            link = entity.get_link(self.spec.parent) if self.spec.parent else entity.base_link
            if link is None:
                raise ModelError(f'Unknown sensor attachment link: {self.spec.parent}')
            self.native.attach(link, transform)

    def invalidate(self) -> None:
        self._sample = -1
        self._cached = None

    def read(self, clock: tuple[int, float], n_envs: int) -> CameraFrame:
        step, sim_rate = clock
        sample = int(step * self.spec.rate / sim_rate)
        if self._cached is None or sample != self._sample:
            if self.spec.parent != 'world':
                self.native.move_to_attach()
            rgb, depth = render_images(self.native, n_envs, self.spec.kind in {'rgbd', 'depth'})
            self._cached = CameraFrame(
                None if self.spec.kind == 'depth' else rgb, depth, self.frame
            )
            self._sample = sample
        return self._cached
