"""Deterministic supervisor receipts, including a native zero exit without results."""

import io
import json
from pathlib import Path
import subprocess
import threading
from unittest.mock import patch

import pytest

from usim import SimulationConfig
from usim.ports.isaac import IsaacSimulator
from usim_isaacsim.runner import main


class Process:
    def __init__(self, output):
        self.stdout = io.StringIO(output)

    def wait(self, timeout):
        return 0

    def poll(self):
        return 0


@pytest.mark.parametrize(
    'output,passes',
    [
        ('{"status":"finished","physics_steps":1,"camera_frames":1}\n', True),
        ('', False),
        ('{"status":"error","error":"fixture"}\n', False),
    ],
)
def test_worker_result_required_and_owned_files_removed(tmp_path, output, passes):
    world, robot = tmp_path / 'world.usd', tmp_path / 'robot.urdf'
    world.write_bytes(b'fixture')
    robot.write_bytes(b'fixture')
    process = Process(output)
    configurations = []

    def launch(command, **kwargs):
        configurations.append(Path(command[-1]))
        data = json.loads(configurations[-1].read_text())
        assert data['configuration']['world'] == str(world.resolve())
        assert Path(data['stop_file']).parent == configurations[-1].parent
        assert kwargs['stdin'] is subprocess.DEVNULL
        return process

    with patch('usim_isaacsim.runner.subprocess.Popen', side_effect=launch):
        if passes:
            IsaacSimulator().run(SimulationConfig(world, robot))
        else:
            with pytest.raises(RuntimeError, match='Isaac worker failed'):
                IsaacSimulator().run(SimulationConfig(world, robot))
    assert len(configurations) == 1
    assert not configurations[0].parent.exists()
    assert process.stdout.closed


def test_shutdown_timeout_kills_worker_and_removes_assets(tmp_path):
    world, robot = tmp_path / 'world.usd', tmp_path / 'robot.urdf'
    world.write_bytes(b'fixture')
    robot.write_bytes(b'fixture')
    parsed, released = threading.Event(), threading.Event()
    clock = [0.0]

    class Output:
        closed = False

        def __iter__(self):
            yield '{"status":"finished","physics_steps":1,"camera_frames":1}\n'
            # Re-entered only after the reader has recorded the yielded result.
            parsed.set()
            assert released.wait(timeout=5)

        def close(self):
            self.closed = True

    class Hung:
        stdout = Output()
        calls = 0
        killed = False

        def wait(self, timeout):
            if self.killed:
                return -9
            assert parsed.wait(timeout=5)
            self.calls += 1
            if self.calls == 2:
                clock[0] = 31.0
            raise subprocess.TimeoutExpired('fixture', timeout)

        def poll(self):
            return -9 if self.killed else None

        def kill(self):
            self.killed = True
            released.set()

    process = Hung()
    directories = []

    def launch(command, **kwargs):
        directories.append(Path(command[-1]).parent)
        return process

    with (
        patch('usim_isaacsim.runner.subprocess.Popen', side_effect=launch),
        patch('usim_isaacsim.runner.time.monotonic', side_effect=lambda: clock[0]),
        pytest.raises(TimeoutError, match='shutdown exceeded'),
    ):
        IsaacSimulator().run(SimulationConfig(world, robot))
    assert process.killed and process.stdout.closed
    assert not directories[0].exists()


def test_precancel_does_not_start_sdk(tmp_path):
    stop = threading.Event()
    stop.set()
    with patch('usim_isaacsim.runner.subprocess.Popen') as launch:
        IsaacSimulator().run(
            SimulationConfig(tmp_path / 'missing', tmp_path / 'missing2'), stop=stop
        )
    launch.assert_not_called()


def test_worker_deserializes_wheel_groups_as_tuples(tmp_path):
    from dataclasses import asdict

    config = SimulationConfig(
        tmp_path / 'world.usd',
        tmp_path / 'robot.urdf',
        left_joints=('lf', 'lr'),
        right_joints=('rf', 'rr'),
        ros=None,
    )
    fields = asdict(config)
    fields.update(world=str(config.world), robot_urdf=str(config.robot_urdf))
    path = tmp_path / 'configuration.json'
    path.write_text(
        json.dumps(
            {
                'configuration': fields,
                'port': {'contact_out': None},
                'stop_file': str(tmp_path / 'stop'),
                'asset_directory': str(tmp_path),
            }
        )
    )
    with (
        patch('sys.argv', ['worker', str(path)]),
        patch('usim_isaacsim.runner.threading.Thread'),
        patch('usim_isaacsim.sim._run') as execute,
    ):
        main()
    restored = execute.call_args.args[0]
    assert restored == config
    assert isinstance(restored.left_joints, tuple)
    assert isinstance(restored.right_joints, tuple)
