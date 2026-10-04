"""Gazebo runtime configuration and smoke behavior."""

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import runpy
import subprocess
import threading
from typing import Literal

import pytest

from usim.ports.gazebo import GazeboSimulator
from usim_gazebo.runner import load_configuration
from usim.robot import MobileRobot, render_robot
from usim.simulation import SimulationConfig


@pytest.mark.parametrize('paired', [False, True])
def test_native_config_roundtrip_and_precancel(
    config: SimulationConfig, tmp_path: Path, paired: bool
) -> None:
    if paired:
        config = replace(config, left_joints=('lf', 'lr'), right_joints=('rf', 'rr'))
    fields = asdict(replace(config, ros=None))
    fields['world'], fields['robot_urdf'] = str(config.world), str(config.robot_urdf)
    path = tmp_path / 'config.json'
    path.write_text(
        json.dumps(
            {
                'configuration': fields,
                'private_topic': '/private/run/cmd_vel',
                'smoke_output': None,
            }
        )
    )
    loaded, topic, output = load_configuration(path)
    assert loaded == replace(config, ros=None)
    assert topic == '/private/run/cmd_vel'
    assert output is None
    stop = threading.Event()
    stop.set()
    GazeboSimulator(image='not-an-image').run(loaded, stop=stop)


@pytest.mark.skipif(
    not os.environ.get('USIM_GAZEBO_WHEEL_SMOKE'),
    reason='requires native Gazebo container runtime',
)
@pytest.mark.parametrize('paired', [False, True])
def test_native_wheel_smoke(
    config: SimulationConfig,
    paired_config: SimulationConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    paired: bool,
) -> None:
    match os.environ.get('USIM_GAZEBO_ENGINE', 'docker'):
        case 'docker':
            engine: Literal['docker', 'podman'] = 'docker'
        case 'podman':
            engine = 'podman'
        case unsupported:
            pytest.fail(f'unsupported Gazebo engine: {unsupported}')

    # The probe is in the same container, so isolate its Gazebo master and ROS topics
    # from other sessions sharing the host network.
    native_run = subprocess.run

    def isolated_run(command, **kwargs):
        if command[:2] == [engine, 'create']:
            command = list(command)
            command[command.index('--network') + 1] = 'none'
        return native_run(command, **kwargs)

    monkeypatch.setattr(subprocess, 'run', isolated_run)
    world = runpy.run_path(str(Path(__file__).parents[1] / 'scripts' / 'smoke_mobile.py'))['WORLD']
    config.world.write_text(world, encoding='utf-8')
    if not paired:
        config.robot_urdf.write_text(render_robot(MobileRobot()), encoding='utf-8')
    config = replace(
        paired_config if paired else config,
        headless=True,
        camera_width=320,
        camera_height=240,
    )
    output = tmp_path / 'native-smoke'
    GazeboSimulator(image=os.environ['USIM_GAZEBO_WHEEL_SMOKE'], engine=engine).smoke(
        config, output
    )
    assert json.loads((output / 'smoke.json').read_text())['passed']


@pytest.mark.parametrize('cancelled', [True, False])
def test_unexecuted_smoke_cannot_leave_stale_success(
    config: SimulationConfig, tmp_path: Path, cancelled: bool
) -> None:
    output = tmp_path / 'output'
    output.mkdir()
    report = output / 'smoke.json'
    report.write_text(json.dumps({'passed': True}))
    if cancelled:
        stop = threading.Event()
        stop.set()
        GazeboSimulator().smoke(config, output, stop=stop)
    else:
        with pytest.raises(ValueError):
            GazeboSimulator().smoke(replace(config, robot_urdf=tmp_path / 'absent'), output)
    assert not report.exists()
