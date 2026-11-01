"""Instantaneous rotary RTX scans, with native imports deferred until construction."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from usim.simulation import SimulationConfig

if TYPE_CHECKING:
    from isaacsim.core.api import World


@final
class RtxLidar:
    """Own a native sensor and track consecutive missing scans.

    Rotary samples cover [angle_min, angle_max), unlike the PhysX implementation's
    duplicated endpoint. Construct only after SimulationApp has initialized.
    """

    def __init__(
        self,
        configuration: SimulationConfig,
        world: World,
        prim_path: str = '/World/UsimRtxLidar',
    ) -> None:
        if not math.isclose(
            configuration.lidar_angle_max - configuration.lidar_angle_min,
            math.tau,
            rel_tol=0,
            abs_tol=1e-9,
        ):
            message = 'RTX lidar supports only a full 360-degree rotary scan'
            raise ValueError(message)
        scan_rate = configuration.lidar_hz
        pattern_rate = scan_rate * configuration.lidar_samples
        if not float(scan_rate).is_integer() or not float(pattern_rate).is_integer():
            message = 'RTX lidar requires integral SDK scan and pattern firing rates in Hz'
            raise ValueError(message)

        import carb
        import omni.usd
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension('isaacsim.sensors.experimental.rtx')
        from isaacsim.sensors.experimental.rtx import (
            Lidar,
            LidarSensor,
            parse_generic_model_output_data,
        )
        from pxr import Vt

        self._stage = omni.usd.get_context().get_stage()
        if self._stage.GetPrimAtPath(prim_path).IsValid():
            message = f'RTX lidar requires an unoccupied prim path: {prim_path}'
            raise ValueError(message)
        self._configuration = configuration
        self._world = world
        self._prim_path = prim_path
        self._lidar = None
        self._sensor = None
        self._parse = parse_generic_model_output_data
        self._previous_stamp: int | None = None
        self._invalid_captures = 0
        self._closed = False
        elevations = configuration.lidar_vertical_angles or (0.0,)
        emitters = len(elevations)
        self._expected_elements = configuration.lidar_samples * emitters
        prefix = 'omni:sensor:Core:'
        initialized = False
        try:
            carb.settings.get_settings().set_bool('/app/sensors/nv/lidar/outputBufferOnGPU', True)
            self._lidar = Lidar(
                prim_path,
                tick_rate=int(scan_rate),
                accumulate_outputs=False,
                aux_output_level='BASIC',
                attributes={
                    prefix + 'scanType': 'ROTARY',
                    prefix + 'numberOfEmitters': emitters,
                    prefix + 'numberOfChannels': emitters,
                    prefix + 'patternFiringRateHz': int(pattern_rate),
                    prefix + 'scanRateBaseHz': int(scan_rate),
                    prefix + 'instantLidar': True,
                    prefix + 'rotationDirection': 'CCW',
                    prefix + 'startAzimuthOffsetDeg': math.degrees(configuration.lidar_angle_min),
                    prefix + 'nearRangeM': configuration.lidar_min_range,
                    prefix + 'farRangeM': configuration.lidar_max_range,
                    prefix + 'maxReturns': 1,
                    prefix + 'rangeAccuracyM': 0,
                    prefix + 'azimuthErrorStd': 0,
                    prefix + 'elevationErrorStd': 0,
                    prefix + 'elementsCoordsType': 'CARTESIAN',
                    prefix + 'skipDroppingInvalidPoints': True,
                },
            )
            # Replicator converts lists incorrectly; author typed arrays directly after creation.
            for name, values in {
                'azimuthDeg': Vt.FloatArray([0.0] * emitters),
                'elevationDeg': Vt.FloatArray([math.degrees(angle) for angle in elevations]),
                'fireTimeNs': Vt.UIntArray([0] * emitters),
                'channelId': Vt.UIntArray(list(range(1, emitters + 1))),
            }.items():
                self._lidar.prims[0].GetAttribute(prefix + 'emitterState:s001:' + name).Set(values)
            self._sensor = LidarSensor(self._lidar, annotators=['generic-model-output'])
            initialized = True
        finally:
            if not initialized:
                self.close()

    def capture(
        self, position_w: ArrayLike, orientation_wxyz: ArrayLike
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]] | None:
        """Render once at the body-fixed sensor pose and return fresh sensor-local hits."""
        if self._closed or self._lidar is None or self._sensor is None:
            message = 'RTX lidar is closed'
            raise RuntimeError(message)
        w, x, y, z = np.asarray(orientation_wxyz)
        rotation = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        origin = np.asarray(position_w) + rotation @ self._configuration.lidar_offset
        self._lidar.set_world_poses(
            positions=origin.reshape(1, 3),
            orientations=np.asarray(orientation_wxyz).reshape(1, 4),
        )
        self._world.render()
        # LidarSensor.get_data requests CUDA; the official parser performs the host copy once.
        data, _ = self._sensor.get_data('generic-model-output')
        if data is None or data.size == 0:
            return self._missing_scan()
        gmo = self._parse(data)
        stamp = int(gmo.timestampNs)
        if gmo.numElements == 0 or (
            self._previous_stamp is not None and stamp <= self._previous_stamp
        ):
            return self._missing_scan()
        if gmo.numElements != self._expected_elements:
            message = (
                f'RTX lidar emitted {gmo.numElements} rays; expected {self._expected_elements}'
            )
            raise RuntimeError(message)
        self._previous_stamp = stamp
        points = np.stack((gmo.x, gmo.y, gmo.z), axis=1).astype(np.float32)
        intensities = np.array(gmo.scalar, dtype=np.float32, copy=True)
        if points.shape != (self._expected_elements, 3) or intensities.shape != (
            self._expected_elements,
        ):
            message = 'RTX lidar Cartesian or intensity payload does not match its ray count'
            raise RuntimeError(message)
        distances = np.linalg.norm(points, axis=1)
        valid = (
            np.isfinite(points).all(axis=1)
            & np.isfinite(intensities)
            & (distances >= self._configuration.lidar_min_range)
            & (distances <= self._configuration.lidar_max_range)
        )
        if not valid.any():
            return self._missing_scan()
        self._invalid_captures = 0
        return points[valid], intensities[valid]

    def _missing_scan(self) -> None:
        """Bound warm-up, stale buffers, and scans without meaningful hits by render count."""
        self._invalid_captures += 1
        if self._invalid_captures >= 30:
            message = 'RTX lidar produced no fresh valid scan for 30 consecutive captures'
            raise RuntimeError(message)
        return None

    def close(self) -> None:
        """Detach annotators, release the runtime, and remove only this owned sensor prim."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._sensor is not None:
                self._sensor.detach_annotators(self._sensor.annotators)
        finally:
            # Releasing LidarSensor runs its native destructor, destroying its render product.
            self._sensor = None
            self._lidar = None
            if self._stage.GetPrimAtPath(self._prim_path).IsValid():
                self._stage.RemovePrim(self._prim_path)
