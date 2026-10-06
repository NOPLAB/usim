"""Gazebo container lifecycle and engine selection."""

from dataclasses import dataclass, field, replace
from pathlib import Path
import os
import subprocess
import sys
from typing import Literal
import xml.etree.ElementTree as ET

import pytest

from usim.ports.gazebo import GazeboLidarConfig, GazeboSimulator
from usim.simulation import SimulationConfig


@dataclass(frozen=True, slots=True)
class ContainerCli:
    """Records fake container CLI calls in its append-only lists."""

    commands: list[list[str]] = field(default_factory=list)
    robots: list[ET.Element] = field(default_factory=list)
    profiles: list[bytes] = field(default_factory=list)


@pytest.fixture
def container_cli(monkeypatch: pytest.MonkeyPatch) -> ContainerCli:
    cli = ContainerCli()
    native_run = subprocess.run
    native_popen = subprocess.Popen

    def run(command, **kwargs):
        cli.commands.append(list(command))
        if command[1] == 'create':
            staging = next(
                argument.split(',')[1].removeprefix('source=')
                for argument in command
                if argument.endswith('target=/run/usim,readonly')
            )
            runtime = Path(staging) / 'runtime'
            assert (runtime / 'usim' / 'simulation.py').is_file()
            assert (runtime / 'usim' / 'robot.py').is_file()
            assert (runtime / 'usim' / 'bridges' / 'smoke.py').is_file()
            assert (runtime / 'usim_gazebo' / 'runner.py').is_file()
            assert (runtime / 'usim_gazebo' / 'smoke.py').is_file()
            assert not (runtime / 'usim' / 'ports' / 'gazebo').exists()
            assert 'python3 -m usim_gazebo.runner' in command[-1]
            imported = native_run(
                [
                    sys.executable,
                    '-c',
                    'import usim; import usim_gazebo.runner; import usim_gazebo.smoke',
                ],
                cwd=runtime,
                env={**os.environ, 'PYTHONPATH': str(runtime)},
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert imported.returncode == 0, imported.stdout + imported.stderr
            cli.robots.append(ET.parse(Path(staging) / 'robot.urdf').getroot())
            if 'FASTRTPS_DEFAULT_PROFILES_FILE=/run/usim/fastdds_profile.xml' in command:
                cli.profiles.append((Path(staging) / 'fastdds_profile.xml').read_bytes())
        return subprocess.CompletedProcess(command, 0, stdout='owned\n')

    class FinishedProcess:
        def wait(self, timeout=None):
            return 0

    def popen(command, **kwargs):
        if command[0] == sys.executable:
            return native_popen(command, **kwargs)
        cli.commands.append(list(command))
        return FinishedProcess()

    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr(subprocess, 'Popen', popen)
    return cli


def test_container_model_path_keeps_builtin_gazebo_models(
    config: SimulationConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = []

    def run(command, **kwargs):
        commands.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout='')

    class FinishedProcess:
        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr(subprocess, 'Popen', lambda command: FinishedProcess())
    GazeboSimulator().run(replace(config, headless=True))
    create = next(command for command in commands if command[:2] == ['docker', 'create'])
    model_path = next(argument for argument in create if argument.startswith('GAZEBO_MODEL_PATH='))
    assert model_path.endswith(':/usr/share/gazebo-11/models')


def test_owned_container_is_removed_when_container_enumeration_times_out(
    config: SimulationConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: creation succeeds, but the engine's global container listing hangs.
    created = []
    removed = []

    def run(command, **kwargs):
        if command[1] == 'create':
            created.append(command[command.index('--name') + 1])
        if command[1] == 'ps':
            raise subprocess.TimeoutExpired(command, kwargs['timeout'])
        if command[1] == 'rm':
            removed.append(command[-1])
        return subprocess.CompletedProcess(command, 0, stdout='')

    class FinishedProcess:
        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr(subprocess, 'Popen', lambda command: FinishedProcess())
    # When: the owned simulator finishes.
    GazeboSimulator(engine='podman').run(replace(config, headless=True))
    # Then: cleanup removes that container without depending on global enumeration.
    assert removed == created and len(removed) == 1


@pytest.mark.parametrize('engine', ['docker', 'podman'])
def test_engine_selects_every_container_cli_call(
    config: SimulationConfig,
    container_cli: ContainerCli,
    engine: Literal['docker', 'podman'],
) -> None:
    GazeboSimulator(engine=engine).run(replace(config, headless=True))
    commands = container_cli.commands
    assert [command[:2] for command in commands] == [
        [engine, 'create'],
        [engine, 'start'],
        [engine, 'rm'],
    ]
    assert commands[2][-1] == commands[1][-1]


@pytest.mark.parametrize(('engine', 'prefix'), [('docker', '^/'), ('podman', '^')])
def test_uncertain_creation_removes_only_its_exact_owned_name(
    config: SimulationConfig,
    monkeypatch: pytest.MonkeyPatch,
    engine: Literal['docker', 'podman'],
    prefix: str,
) -> None:
    # Given: the create command fails after partially creating its named container.
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if command[1] == 'create':
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0, stdout='owned\n')

    monkeypatch.setattr(subprocess, 'run', run)
    # When: the backend propagates the engine failure.
    with pytest.raises(subprocess.CalledProcessError):
        GazeboSimulator(engine=engine).run(replace(config, headless=True))
    # Then: partial cleanup remains limited to the exact UUID it tried to create.
    name = commands[0][commands[0].index('--name') + 1]
    assert commands[1] == [
        engine,
        'ps',
        '--all',
        '--quiet',
        '--filter',
        f'name={prefix}{name}$',
    ]
    assert commands[2] == [engine, 'rm', '--force', name]


def test_host_dds_settings_are_forwarded_only_when_set(
    config: SimulationConfig, container_cli: ContainerCli, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('FASTDDS_BUILTIN_TRANSPORTS', 'UDPv4')
    monkeypatch.setenv('ROS_DOMAIN_ID', '42')
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    monkeypatch.delenv('RMW_IMPLEMENTATION', raising=False)
    GazeboSimulator(engine='podman').run(replace(config, headless=True))
    create = container_cli.commands[0]
    forwarded = {
        create[i + 1]
        for i, item in enumerate(create)
        if item == '--env'
        and create[i + 1].split('=')[0]
        in (
            'FASTDDS_BUILTIN_TRANSPORTS',
            'ROS_DOMAIN_ID',
            'RMW_IMPLEMENTATION',
            'ROS_LOCALHOST_ONLY',
        )
    }
    assert forwarded == {
        'FASTDDS_BUILTIN_TRANSPORTS=UDPv4',
        'ROS_DOMAIN_ID=42',
        'ROS_LOCALHOST_ONLY=1',
    }


def test_explicit_fastdds_profile_is_staged_without_changing_the_source(
    config: SimulationConfig, container_cli: ContainerCli, tmp_path: Path
) -> None:
    # Given: a caller-owned middleware profile outside the container.
    profile = tmp_path / 'udp profile.xml'
    xml = b'<profiles><transport_descriptors/></profiles>'
    profile.write_bytes(xml)
    # When: the backend launches with that explicit profile.
    GazeboSimulator(fastdds_profile=profile).run(replace(config, headless=True))
    # Then: native ROS sees the exact file on the private read-only staging mount.
    assert container_cli.profiles == [xml]
    assert profile.read_bytes() == xml
    assert str(profile) not in container_cli.commands[0]


def test_missing_fastdds_profile_fails_before_any_engine_call(
    config: SimulationConfig, container_cli: ContainerCli, tmp_path: Path
) -> None:
    # Given: an explicit profile which cannot be read.
    profile = tmp_path / 'missing.xml'
    # When: the backend prepares the run.
    with pytest.raises(FileNotFoundError):
        GazeboSimulator(fastdds_profile=profile).run(replace(config, headless=True))
    # Then: no container is created for an invalid input.
    assert container_cli.commands == []


def test_default_run_does_not_set_a_fastdds_profile(
    config: SimulationConfig, container_cli: ContainerCli
) -> None:
    # Given / When: a run without middleware-specific configuration.
    GazeboSimulator().run(replace(config, headless=True))
    # Then: the adapter does not impose a profile or depend on its files.
    assert container_cli.profiles == []
    assert not any(
        item.startswith('FASTRTPS_DEFAULT_PROFILES_FILE=') for item in container_cli.commands[0]
    )


def test_simulator_lidar_reaches_staged_robot(
    config: SimulationConfig, container_cli: ContainerCli
) -> None:
    lidar = GazeboLidarConfig('base_link', 'laser', '/robot/scan')
    GazeboSimulator(lidar=lidar).run(replace(config, headless=True))
    remapping = container_cli.robots[0].findtext(
        "gazebo/sensor[@name='usim_lidar']/plugin/ros/remapping"
    )
    assert remapping == '~/out:=/robot/scan'
