"""Native Python 3.10/Humble lifecycle. Invoked only inside the Gazebo container."""

from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from usim.simulation import CommandGate, Ros2Config, SimulationConfig, Velocity


def load_configuration(path: Path) -> tuple[SimulationConfig, str, Path | None]:
    data = json.loads(path.read_text(encoding='utf-8'))
    fields = data['configuration']
    fields['world'] = Path(fields['world'])
    fields['robot_urdf'] = Path(fields['robot_urdf'])
    fields['camera_offset'] = tuple(fields['camera_offset'])
    if fields['ros'] is not None:
        fields['ros'] = Ros2Config(**fields['ros'])
    return (
        SimulationConfig(**fields),
        data['private_topic'],
        Path(data['smoke_output']) if data.get('smoke_output') else None,
    )


def _terminate(process: subprocess.Popen[bytes]) -> None:
    # Kill the session even if the leader already exited, so ros2 children cannot leak.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run(
    config: SimulationConfig,
    private_topic: str,
    config_path: Path,
    smoke_output: Path | None,
    stop: threading.Event,
) -> None:
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.signals import SignalHandlerOptions
    from rosgraph_msgs.msg import Clock
    from std_srvs.srv import SetBool

    processes = []

    def launch(command: list[str]) -> subprocess.Popen[bytes]:
        process = subprocess.Popen(command, start_new_session=True)
        processes.append(process)
        return process

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    ros = config.ros
    node = rclpy.create_node(ros.node_name if ros else 'usim_gazebo_lifecycle')
    gate = CommandGate()
    sim_time = [0.0]
    odometry = [False]
    publisher = None

    def clock(message):
        sim_time[0] = message.clock.sec + message.clock.nanosec * 1e-9

    node.create_subscription(Clock, '/clock', clock, qos_profile_sensor_data)
    if ros is not None:
        publisher = node.create_publisher(Twist, private_topic, 10)

        def publish():
            velocity = gate.sample()
            message = Twist()
            message.linear.x, message.angular.z = velocity.linear, velocity.angular
            publisher.publish(message)

        def command(message):
            try:
                gate.command(Velocity(float(message.linear.x), float(message.angular.z)))
            except ValueError as error:
                gate.command(Velocity())
                node.get_logger().warning(f'rejected cmd_vel: {error}')

        def motor(request, response):
            gate.set_motor(bool(request.data))
            publish()
            response.success = True
            response.message = 'enabled' if gate.enabled else 'disabled'
            return response

        def odom(message):
            odometry[0] = True

        node.create_subscription(Twist, ros.cmd_vel_topic, command, 10)
        node.create_subscription(Odometry, ros.odom_topic, odom, 10)
        node.create_service(SetBool, ros.motor_service, motor)
        node.create_timer(0.02, publish)

    try:
        display_process = None
        if config.headless and config.camera_enabled:
            read_fd, write_fd = os.pipe()
            try:
                display_process = subprocess.Popen(
                    [
                        'Xvfb',
                        '-displayfd',
                        str(write_fd),
                        '-screen',
                        '0',
                        '1280x720x24',
                        '-nolisten',
                        'tcp',
                        '-ac',
                    ],
                    pass_fds=(write_fd,),
                    start_new_session=True,
                )
                processes.append(display_process)
                os.close(write_fd)
                write_fd = -1
                if not select.select([read_fd], [], [], 20)[0]:
                    raise TimeoutError('Xvfb did not report a ready display')
                display = os.read(read_fd, 64).decode().strip()
                if not display.isdigit():
                    raise RuntimeError('Xvfb exited before display readiness')
                os.environ['DISPLAY'] = ':' + display
                os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', '1')
            finally:
                os.close(read_fd)
                if write_fd != -1:
                    os.close(write_fd)
        server = launch(
            [
                'gzserver',
                '--verbose',
                str(config.world),
                '-s',
                'libgazebo_ros_init.so',
                '-s',
                'libgazebo_ros_factory.so',
            ]
        )
        critical = [server] + ([display_process] if display_process else [])
        if not config.headless:
            critical.append(launch(['gzclient', '--verbose']))
        spawn = launch(
            [
                'ros2',
                'run',
                'gazebo_ros',
                'spawn_entity.py',
                '-entity',
                config.robot_name,
                '-file',
                str(config.robot_urdf),
                '-timeout',
                '60',
            ]
        )
        started = time.monotonic()
        spawned_at = None
        probe = None
        while not stop.is_set():
            rclpy.spin_once(node, timeout_sec=0.05)
            for process in critical:
                if process.poll() is not None:
                    raise RuntimeError(f'{process.args!r} exited with {process.returncode}')
            if spawned_at is None:
                if spawn.poll() is not None:
                    if spawn.returncode:
                        raise RuntimeError(f'robot spawn failed with {spawn.returncode}')
                    spawned_at = time.monotonic()
                    print('USIM_GAZEBO_SPAWNED', flush=True)
                    if smoke_output is not None:
                        probe = launch(
                            [
                                sys.executable,
                                '-m',
                                'usim_gazebo.smoke',
                                str(config_path),
                                str(smoke_output),
                            ]
                        )
                elif time.monotonic() - started > 90:
                    raise TimeoutError('Gazebo robot spawn timed out')
            elif ros is not None and not odometry[0] and time.monotonic() - started > 90:
                raise TimeoutError('Gazebo drive did not publish odometry')
            if probe is not None and probe.poll() is not None:
                if probe.returncode:
                    raise RuntimeError(f'Gazebo smoke probe failed with {probe.returncode}')
                return
            if (
                spawned_at is not None
                and config.max_seconds
                and time.monotonic() - spawned_at >= config.max_seconds
            ):
                if probe is not None:
                    raise TimeoutError('simulation deadline reached before smoke completion')
                return
    finally:
        gate.set_motor(False)
        if publisher is not None:
            publisher.publish(Twist())
        for process in reversed(processes):
            _terminate(process)
        node.destroy_node()
        rclpy.shutdown()


def main() -> None:
    config_path = Path(sys.argv[1])
    config, private_topic, output = load_configuration(config_path)
    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stop.set())
    run(config, private_topic, config_path, output, stop)


if __name__ == '__main__':
    main()
