"""Continuous CLI supervision and backend configuration without native engines."""

import io
import queue
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import Mock

import pytest

from usim.cli import build_parser, main
from usim.runtime_cli import configured
from usim_gazebo import GazeboLidarConfig, GazeboSimulator


def arguments(*options):
    return build_parser().parse_args(
        ['simulate', '--world', 'room.sdf', '--robot-urdf', 'robot.urdf', *options]
    )


@pytest.fixture
def runner(monkeypatch):
    runner = Mock()
    factory = Mock(return_value=runner)
    monkeypatch.setattr('usim.factory.create_runner', factory)
    return runner, factory


def test_gazebo_defaults_remain_backend_owned(runner):
    execute, factory = runner
    args = arguments('--backend', 'gazebo')
    args.handler(args)
    factory.assert_called_once_with('gazebo')
    execute.run.assert_called_once_with(configured(args))


def test_gazebo_options_and_lidar_forwarding(runner):
    execute, factory = runner
    args = arguments(
        '--backend',
        'gazebo',
        '--engine',
        'podman',
        '--image',
        'custom:latest',
        '--network',
        'bridge',
        '--fastdds-profile',
        'dds.xml',
        '--lidar-link',
        'sensor_link',
        '--lidar-frame',
        'laser',
        '--lidar-topic',
        '/scan',
        '--lidar-update-rate',
        '15',
        '--lidar-horizontal-samples',
        '360',
        '--lidar-min-angle',
        '-1.5',
        '--lidar-max-angle',
        '1.5',
        '--lidar-range-min',
        '0.2',
        '--lidar-range-max',
        '20',
    )
    args.handler(args)
    factory.assert_called_once_with(
        'gazebo',
        engine='podman',
        image='custom:latest',
        network='bridge',
        fastdds_profile=Path('dds.xml'),
        lidar=GazeboLidarConfig('sensor_link', 'laser', '/scan', 15, 360, -1.5, 1.5, 0.2, 20),
    )
    execute.run.assert_called_once_with(configured(args))


def test_lidar_defaults_remain_dataclass_owned(runner):
    _, factory = runner
    args = arguments(
        '--backend',
        'gazebo',
        '--lidar-link',
        'base_link',
        '--lidar-frame',
        'laser',
        '--lidar-topic',
        '/scan',
    )
    args.handler(args)
    lidar = factory.call_args.kwargs['lidar']
    assert lidar == GazeboLidarConfig('base_link', 'laser', '/scan')
    assert GazeboSimulator().lidar is None


@pytest.fixture
def stdin_control(monkeypatch):
    lines = queue.Queue()
    reading = threading.Event()
    threads = []
    original_thread = threading.Thread

    class Input:
        def readline(self):
            reading.set()
            return lines.get(timeout=5)

    def thread(*args, **kwargs):
        result = original_thread(*args, **kwargs)
        threads.append(result)
        return result

    monkeypatch.setattr(sys, 'stdin', Input())
    monkeypatch.setattr('usim.runtime_cli.threading.Thread', thread)
    yield lines, reading
    lines.put('')
    for result in threads:
        result.join(timeout=5)
        assert not result.is_alive()


@pytest.mark.parametrize('stop_line', ['stop\n', ''])
@pytest.mark.parametrize('backend', ['gazebo', 'isaacsim'])
def test_stdin_stop_and_eof_cancel_runner(runner, stdin_control, stop_line, backend):
    execute, _ = runner
    lines, reading = stdin_control

    def run(config, *, stop):
        assert isinstance(stop, threading.Event)
        assert reading.wait(timeout=5)
        assert not stop.is_set()
        lines.put('ignored\n')
        lines.put(stop_line)
        assert stop.wait(timeout=5)

    execute.run.side_effect = run
    args = arguments('--backend', backend, '--stop-on-stdin')
    assert args.handler(args) is None


@pytest.mark.parametrize('max_seconds', ['0', '5'])
def test_supervised_return_requires_stop_only_for_indefinite_runs(
    runner, stdin_control, max_seconds
):
    execute, _ = runner
    _, reading = stdin_control

    def run(config, *, stop):
        assert reading.wait(timeout=5)
        assert not stop.is_set()

    execute.run.side_effect = run
    args = arguments('--backend', 'gazebo', '--stop-on-stdin', '--max-seconds', max_seconds)
    if max_seconds == '0':
        with pytest.raises(RuntimeError, match='exited before a stop request'):
            args.handler(args)
    else:
        assert args.handler(args) is None


@pytest.mark.parametrize(
    'options',
    [
        ['--engine', 'invalid'],
        ['--backend', 'gazebo', '--lidar-link', 'base_link'],
        ['--backend', 'gazebo', '--lidar-link', 'base_link', '--lidar-frame', 'laser'],
        ['--backend', 'gazebo', '--lidar-link', 'base_link', '--lidar-topic', '/scan'],
        ['--backend', 'gazebo', '--lidar-frame', 'laser'],
        ['--engine', 'docker'],
        [
            '--backend',
            'gazebo',
            '--lidar-link',
            'base_link',
            '--lidar-frame',
            'laser',
            '--lidar-topic',
            'relative',
        ],
        [
            '--backend',
            'gazebo',
            '--lidar-link',
            'base_link',
            '--lidar-frame',
            'laser',
            '--lidar-topic',
            '/scan',
            '--lidar-horizontal-samples',
            '0',
        ],
    ],
)
def test_invalid_options_fail_at_cli_boundary(monkeypatch, runner, options):
    monkeypatch.setattr(
        sys,
        'argv',
        ['usim', 'simulate', '--world', 'room.sdf', '--robot-urdf', 'robot.urdf', *options],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    runner[1].assert_not_called()


def test_cli_requested_stop_exits_successfully(monkeypatch, runner, capsys):
    monkeypatch.setattr(sys, 'stdin', io.StringIO('stop\n'))
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'usim',
            'simulate',
            '--backend',
            'gazebo',
            '--world',
            'room.sdf',
            '--robot-urdf',
            'robot.urdf',
            '--stop-on-stdin',
        ],
    )

    def run(config, *, stop):
        assert stop.wait(timeout=5)
        print('native log')
        print('native error log', file=sys.stderr)

    runner[0].run.side_effect = run
    assert main() is None
    captured = capsys.readouterr()
    assert captured.out == 'native log\n'
    assert captured.err == 'native error log\n'


@pytest.mark.parametrize('mode', ['stop', 'eof', 'unexpected'])
def test_module_cli_exit_status(mode):
    code = """
import runpy
import sys
import threading
from unittest.mock import patch

mode = sys.argv[1]
reading = threading.Event()
release = threading.Event()

class Input:
    def readline(self):
        reading.set()
        assert release.wait(timeout=10)
        return 'stop\\n' if mode == 'stop' else ''

class Runner:
    def run(self, config, *, stop):
        assert reading.wait(timeout=5)
        if mode != 'unexpected':
            release.set()
            assert stop.wait(timeout=5)

sys.stdin = Input()
sys.argv = [
    'usim', 'simulate', '--backend', 'gazebo', '--world', 'room.sdf',
    '--robot-urdf', 'robot.urdf', '--stop-on-stdin',
]
with patch('usim.factory.create_runner', return_value=Runner()):
    runpy.run_module('usim.cli', run_name='__main__')
"""
    child = subprocess.run(
        [sys.executable, '-B', '-c', code, mode], capture_output=True, text=True, timeout=30
    )
    if mode == 'unexpected':
        assert child.returncode != 0
        assert 'simulation runner exited before a stop request' in child.stderr
    else:
        assert child.returncode == 0, child.stdout + child.stderr
