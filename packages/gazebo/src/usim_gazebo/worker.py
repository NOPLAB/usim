"""Standalone Humble worker: JSON framing around bounded native ROS service calls.

This file runs directly in the container, without importing the host usim wheel.
Gazebo advances continuously; no request fabricates an episode or a physics step.
"""

from __future__ import annotations

import importlib
import base64
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from typing import TypeAlias
from .worker_runtime import SERVICE_TYPES, prepare_world, start_display

Json: TypeAlias = None | bool | int | float | str | list['Json'] | dict[str, 'Json']


class NativeWorker:
    """Own the ROS node, native process, service clients and a single request deadline."""

    def __init__(self, node, ros, directory: Path) -> None:
        self.node = node
        self.ros = ros
        self.directory = directory
        self.server = None
        self.clients = {}
        self.deadline: float = 0.0
        self.display: subprocess.Popen[bytes] | None = None
        self.subscriptions = {}
        self.sensor_waiters = {}
        self.closing = False

    def _watch_server(self, server: subprocess.Popen[bytes]) -> None:
        status = server.wait()
        if not self.closing:
            os._exit(status or 1)

    def _start_display(self) -> None:
        start_display(self)

    def _sensor_message(self, topic: str, message) -> None:
        future = self.sensor_waiters.get(topic)
        if future is None or future.done():
            return
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        if hasattr(message, 'ranges'):
            result = {
                'ranges': [value if math.isfinite(value) else None for value in message.ranges],
                'stamp': stamp,
                'frame': message.header.frame_id,
                'angle_min': message.angle_min,
                'angle_increment': message.angle_increment,
                'range_min': message.range_min,
                'range_max': message.range_max,
            }
        else:
            result = {
                'data': base64.b64encode(bytes(message.data)).decode('ascii'),
                'height': message.height,
                'width': message.width,
                'encoding': message.encoding,
                'step': message.step,
                'is_bigendian': bool(message.is_bigendian),
                'stamp': stamp,
                'frame': message.header.frame_id,
            }
        future.set_result(result)

    def call(self, service: str, fields: dict[str, Json]) -> dict[str, Json]:
        from rosidl_runtime_py.convert import message_to_ordereddict
        from rosidl_runtime_py.set_message import set_message_fields

        if self.server is not None and self.server.poll() is not None:
            raise RuntimeError(f'gzserver exited with {self.server.returncode}')
        module, name = SERVICE_TYPES[service]
        message_type = getattr(importlib.import_module(module), name)
        if service not in self.clients:
            self.clients[service] = self.node.create_client(message_type, service)
        client = self.clients[service]
        remaining = self.deadline - time.monotonic()
        if remaining <= 0 or not client.wait_for_service(timeout_sec=remaining):
            raise TimeoutError(f'Gazebo service unavailable: {service}')
        request = message_type.Request()
        set_message_fields(request, fields)
        future = client.call_async(request)
        self.ros.spin_until_future_complete(
            self.node,
            future,
            timeout_sec=max(0.0, self.deadline - time.monotonic()),
        )
        if not future.done():
            future.cancel()
            raise TimeoutError(f'Gazebo service acknowledgement timed out: {service}')
        response = future.result()
        if hasattr(response, 'success') and not response.success:
            raise RuntimeError(f'{service}: {getattr(response, "status_message", "rejected")}')
        return dict(message_to_ordereddict(response))

    def dispatch(
        self,
        operation: str,
        payload: dict[str, Json],
        timeout: float,
    ) -> dict[str, Json]:
        self.deadline = time.monotonic() + timeout
        match operation:
            case 'initialize':
                if self.server is not None:
                    raise RuntimeError('Gazebo world is already initialized')
                if payload.get('rendering'):
                    self._start_display()
                xml = payload.get('world')
                if not isinstance(xml, str):
                    raise ValueError('World XML must be a string')
                world_path = prepare_world(self.directory, xml)
                self.server = subprocess.Popen(
                    [
                        'gzserver',
                        '--verbose',
                        '--pause',
                        str(world_path),
                        '-s',
                        'libgazebo_ros_init.so',
                        '-s',
                        'libgazebo_ros_factory.so',
                    ],
                    stdout=sys.stderr,
                    stderr=sys.stderr,
                )
                # EOF must reach the host even while this worker is blocked reading stdin.
                # The container init process reaps the native server's children.
                threading.Thread(
                    target=self._watch_server,
                    args=(self.server,),
                    name='gazebo-exit',
                    daemon=True,
                ).start()
                self.call('/pause_physics', {})
                # A real service acknowledgement confirms the server and plugins.
                for service in (
                    '/spawn_entity',
                    '/delete_entity',
                    '/get_entity_state',
                    '/set_entity_state',
                    '/get_model_properties',
                    '/get_joint_properties',
                    '/set_joint_properties',
                ):
                    module, name = SERVICE_TYPES[service]
                    message_type = getattr(importlib.import_module(module), name)
                    client = self.node.create_client(message_type, service)
                    self.clients[service] = client
                    remaining = self.deadline - time.monotonic()
                    if remaining <= 0 or not client.wait_for_service(timeout_sec=remaining):
                        raise TimeoutError(f'Gazebo initialization missing {service}')
                available = {name for name, _ in self.node.get_service_names_and_types()}
                return {'services': sorted(available & SERVICE_TYPES.keys())}
            case 'subscribe_sensors':
                from rclpy.qos import qos_profile_sensor_data

                channels = payload.get('channels')
                if not isinstance(channels, list):
                    raise ValueError('Sensor channels must be a list')
                for channel in channels:
                    if not isinstance(channel, dict):
                        raise ValueError('Invalid sensor channel')
                    topic, message_type = channel.get('topic'), channel.get('message_type')
                    if not isinstance(topic, str) or message_type not in {'Image', 'LaserScan'}:
                        raise ValueError('Invalid sensor topic or message type')
                    native_type = getattr(importlib.import_module('sensor_msgs.msg'), message_type)
                    self.subscriptions[topic] = self.node.create_subscription(
                        native_type,
                        topic,
                        lambda message, topic=topic: self._sensor_message(topic, message),
                        qos_profile_sensor_data,
                    )
                return {}
            case 'read_sensor':
                from rclpy.task import Future

                topic = payload.get('topic')
                if not isinstance(topic, str) or topic not in self.subscriptions:
                    raise ValueError('Sensor is not subscribed')
                future = Future()
                self.sensor_waiters[topic] = future
                try:
                    self.ros.spin_until_future_complete(
                        self.node,
                        future,
                        timeout_sec=max(0.0, self.deadline - time.monotonic()),
                    )
                    if not future.done():
                        future.cancel()
                        raise TimeoutError(f'No native sensor frame received from {topic}')
                    return future.result()
                finally:
                    self.sensor_waiters.pop(topic)
            case 'service':
                service, fields = payload.get('service'), payload.get('fields', {})
                if not isinstance(service, str) or not isinstance(fields, dict):
                    raise ValueError('A service request needs its name and message fields')
                return self.call(service, fields)
            case _:
                raise ValueError(f'Unknown worker operation: {operation}')

    def close(self) -> None:
        self.closing = True
        if self.server is not None:
            self.server.terminate()
            try:
                self.server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server.kill()
                self.server.wait(timeout=5)
        if self.display is not None:
            self.display.terminate()
            try:
                self.display.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.display.kill()
                self.display.wait(timeout=5)
        self.node.destroy_node()


def serve(worker, incoming, outgoing) -> None:
    """Frame every request, including failures, with its original correlation ID."""
    for line in incoming:
        identifier = None
        try:
            request = json.loads(line)
            identifier = request['id']
            if not isinstance(identifier, str):
                raise ValueError('Request ID must be a string')
            result = worker.dispatch(request['operation'], request['payload'], request['timeout'])
            response = {'id': identifier, 'ok': True, 'result': result}
        except (KeyError, TypeError, ValueError, RuntimeError, OSError, TimeoutError) as error:
            response = {'id': identifier, 'ok': False, 'error': str(error)}
        outgoing.write('USIM_RESPONSE ' + json.dumps(response, allow_nan=False) + '\n')
        outgoing.flush()


def main() -> None:
    import rclpy

    rclpy.init()
    with tempfile.TemporaryDirectory(prefix='usim-native-') as directory:
        worker = NativeWorker(rclpy.create_node('usim_grouped'), rclpy, Path(directory))
        try:
            serve(worker, sys.stdin, sys.stdout)
        finally:
            worker.close()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
