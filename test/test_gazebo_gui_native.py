"""Native ROS lifecycle checks for ordering GUI startup after robot spawning."""

from dataclasses import replace
from pathlib import Path
import threading

import pytest

from usim_gazebo import runner
from usim.simulation import SimulationConfig

rclpy = pytest.importorskip('rclpy', reason='requires the native ROS Python runtime')


def test_gui_waits_for_successful_spawn(
    config: SimulationConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: process completion is controlled by an exact spawn event, not a delay.
    spawned = threading.Event()
    spawn_started = threading.Event()
    gui_started = threading.Event()
    stop = threading.Event()
    errors: list[BaseException] = []

    class Process:
        def __init__(self, command, **kwargs):
            self.args = command
            self.returncode = None
            self.spawner = command[:3] == ['ros2', 'run', 'gazebo_ros']
            if self.spawner:
                spawn_started.set()
            elif command[0] == 'gzclient':
                gui_started.set()
                if spawned.is_set():
                    stop.set()

        def poll(self):
            if self.spawner and spawned.is_set():
                self.returncode = 0
            return self.returncode

    def spin_once(*args, **kwargs):
        assert spawned.wait(timeout=5)

    def execute():
        try:
            runner.run(
                replace(config, ros=None, headless=False, camera_enabled=False),
                '/private/cmd_vel',
                tmp_path / 'config.json',
                None,
                stop,
            )
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(runner.subprocess, 'Popen', Process)
    monkeypatch.setattr(runner, '_terminate', lambda _: None)
    monkeypatch.setattr(rclpy, 'spin_once', spin_once)
    worker = threading.Thread(target=execute, daemon=True)
    worker.start()
    try:
        assert spawn_started.wait(timeout=5)
        assert not gui_started.is_set()
        # When: the native spawner reports actual successful completion.
        spawned.set()
        assert gui_started.wait(timeout=5)
    finally:
        spawned.set()
        stop.set()
        worker.join(timeout=5)
    # Then: GUI startup never overlaps world/robot factory initialization.
    assert not worker.is_alive()
    assert not errors
