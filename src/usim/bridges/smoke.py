"""Event-driven ROS smoke probe, reusable from either engine's native interpreter."""

from __future__ import annotations

import array
import hashlib
import json
import math
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, TYPE_CHECKING

from usim.simulation import RobotState, SimulationConfig, Velocity

if TYPE_CHECKING:
    from sensor_msgs.msg import Image


def _yaw(state: RobotState) -> float:
    x, y, z, w = state.orientation_xyzw
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _angle(start: RobotState, end: RobotState) -> float:
    difference = _yaw(end) - _yaw(start)
    return math.atan2(math.sin(difference), math.cos(difference))


def _distance(start: RobotState, end: RobotState) -> float:
    return math.hypot(end.position[0] - start.position[0], end.position[1] - start.position[1])


def probe(config: SimulationConfig, out_dir: Path) -> dict[str, object]:
    """Subscribe before all actions, then await measured state, not guessed delays."""
    from usim.bridges.ros2 import Ros2Client
    from usim.bridges.ros_runtime import load_ros_python

    load_ros_python()
    import rclpy
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    if config.ros is None:
        raise ValueError('smoke needs ROS')
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'smoke.json').unlink(missing_ok=True)
    condition = threading.Condition()
    images: dict[str, Image] = {}
    context = Context()
    rclpy.init(context=context)
    node = Node('usim_smoke_images', context=context)

    def image_callback(kind, message):
        with condition:
            images[kind] = message
            condition.notify_all()

    for kind, topic in (('rgb', config.ros.rgb_topic), ('depth', config.ros.depth_topic)):
        node.create_subscription(
            Image, topic, lambda m, k=kind: image_callback(k, m), qos_profile_sensor_data
        )
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, name='usim-smoke-images')
    thread.start()
    client = None
    trace = []
    result: dict[str, object] = {}
    try:
        client = Ros2Client(config.ros)
        state = client.wait_ready(timeout=300)
        first_time = state.sim_time

        def advance(
            phase: str,
            predicate: Callable[[RobotState], bool],
            velocity: Velocity | None = None,
            timeout: float = 30,
        ) -> RobotState:
            nonlocal state
            deadline = time.monotonic() + timeout
            while True:
                if velocity is not None:
                    client.command(velocity)
                state = client.observe(
                    after=state.sim_time, timeout=max(0, deadline - time.monotonic())
                )
                trace.append({'phase': phase, **asdict(state)})
                if predicate(state):
                    return state
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'{phase}: expected physical state was not observed')

        def settled(phase: str, after_wall: float = 0.0) -> RobotState:
            stable_since: float | None = None

            def stable(current: RobotState) -> bool:
                nonlocal stable_since
                if (
                    time.monotonic() < after_wall
                    or abs(current.velocity.linear) > 0.015
                    or abs(current.velocity.angular) > 0.04
                ):
                    stable_since = None
                    return False
                if stable_since is None:
                    stable_since = current.sim_time
                return current.sim_time - stable_since >= 0.25

            return advance(phase, stable)

        def immobile(phase: str, velocity: Velocity | None) -> None:
            start = state
            maximum = [0.0, 0.0]

            def check(current: RobotState) -> bool:
                maximum[0] = max(maximum[0], _distance(start, current))
                maximum[1] = max(maximum[1], abs(_angle(start, current)))
                return current.sim_time - start.sim_time >= 0.6

            advance(phase, check, velocity)
            if maximum[0] > 0.02 or maximum[1] > 0.04:
                raise AssertionError(f'{phase}: moved while disabled/stopped: {maximum}')
            result[phase] = {'distance_m': maximum[0], 'yaw_rad': maximum[1]}

        settled('initial_settle')
        immobile('motor_disabled', Velocity(0.3, 0.5))
        client.set_motor(True)
        start = state
        advance('forward', lambda current: _distance(start, current) >= 0.15, Velocity(0.25, 0.0))
        projection = (state.position[0] - start.position[0]) * math.cos(_yaw(start)) + (
            state.position[1] - start.position[1]
        ) * math.sin(_yaw(start))
        if projection < 0.12:
            raise AssertionError('forward command did not move along the body forward axis')
        result['forward_m'] = projection
        start = state
        advance('turn', lambda current: _angle(start, current) >= 0.35, Velocity(0.0, 0.7))
        result['turn_rad'] = _angle(start, state)
        client.set_motor(False)
        settled('disable_settle')
        immobile('motor_disabled_again', Velocity(0.3, 0.5))
        client.set_motor(True)
        advance(
            'watchdog_drive', lambda current: current.velocity.linear > 0.12, Velocity(0.25, 0.0)
        )
        client.command(Velocity(0.25, 0.0))
        last_command = time.monotonic()
        settled('watchdog_settle', after_wall=last_command + 0.5)
        result['watchdog_wall_seconds'] = time.monotonic() - last_command
        immobile('watchdog_stopped', None)
        result['sim_time_start'] = first_time
        result['sim_time_end'] = state.sim_time
        with condition:
            if not condition.wait_for(lambda: len(images) == 2, timeout=30):
                raise TimeoutError('RGB and depth payloads were not both received')
            captured = dict(images)
        result['images'] = _save_images(captured, config, out_dir)
        result['passed'] = True
        (out_dir / 'smoke.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result, indent=2), flush=True)
        return result
    finally:
        (out_dir / 'trajectory.json').write_text(json.dumps(trace), encoding='utf-8')
        try:
            if client is not None:
                client.close()
        finally:
            executor.shutdown(timeout_sec=5)
            thread.join(timeout=5)
            node.destroy_node()
            context.shutdown()
            if thread.is_alive():
                raise RuntimeError('image executor failed to stop')


def _save_images(
    images: dict[str, Image], config: SimulationConfig, out_dir: Path
) -> dict[str, object]:
    result: dict[str, object] = {}
    for kind, encoding, channels in (('rgb', 'rgb8', 3), ('depth', '32FC1', 4)):
        image = images[kind]
        payload = bytes(image.data)
        if (
            image.encoding != encoding
            or image.width != config.camera_width
            or image.height != config.camera_height
            or image.step < image.width * channels
            or len(payload) != image.step * image.height
        ):
            raise AssertionError(f'{kind}: invalid image metadata or payload size')
        # Preserve payload equality while stripping any ROS row padding for artifacts.
        packed = b''.join(
            payload[row * image.step : row * image.step + image.width * channels]
            for row in range(image.height)
        )
        if kind == 'rgb':
            if len(set(packed)) < 8:
                raise AssertionError('camera returned a constant/empty image')
            header = chr(10).join(('P6', f'{image.width} {image.height}', '255', '')).encode()
            (out_dir / 'rgb.ppm').write_bytes(header + packed)
        else:
            values = array.array('f')
            values.frombytes(packed)
            if bool(image.is_bigendian) != (sys.byteorder == 'big'):
                values.byteswap()
            finite = [value for value in values if math.isfinite(value) and 0.05 < value < 10]
            if len(finite) < image.width:
                raise AssertionError('depth contains no real in-range scene geometry')
            result['depth_range_m'] = [min(finite), max(finite)]
            # Artifact format is always little-endian float32, independently of ROS endian.
            if sys.byteorder == 'big':
                values.byteswap()
            (out_dir / 'depth.f32').write_bytes(values.tobytes())
        result[kind] = {
            'encoding': encoding,
            'width': image.width,
            'height': image.height,
            'bytes': len(payload),
            'sha256': hashlib.sha256(payload).hexdigest(),
            'sim_time': image.header.stamp.sec + image.header.stamp.nanosec * 1e-9,
        }
    return result
