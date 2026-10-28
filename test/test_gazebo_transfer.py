"""Container-local assets must not become daemon-host bind paths."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import tarfile
from typing import BinaryIO, Literal

import pytest

from usim.simulation import SimulationConfig
from usim_gazebo import GazeboSimulator


@pytest.mark.parametrize('engine', ['docker', 'podman'])
@pytest.mark.parametrize('selection', ['explicit', 'environment'])
def test_local_assets_are_transferred_before_worker_starts(
    config: SimulationConfig,
    monkeypatch: pytest.MonkeyPatch,
    engine: Literal['docker', 'podman'],
    selection: str,
) -> None:
    # Given: the daemon cannot bind paths from the caller's filesystem.
    commands = []
    transferred = {}
    texture = config.world.parent / 'texture.dat'
    texture.write_bytes(b'caller-owned resource')

    def run(command, *, stdin: BinaryIO | None = None, **kwargs):
        commands.append(list(command))
        if command[1] == 'create' and '--mount' in command:
            raise subprocess.CalledProcessError(1, command)
        if command[1] == 'cp':
            assert stdin is not None
            with tarfile.open(fileobj=stdin, mode='r:') as archive:
                for member in archive:
                    if member.isfile():
                        content = archive.extractfile(member)
                        assert content is not None
                        with content:
                            transferred[member.name] = content.read()
        return subprocess.CompletedProcess(command, 0, stdout='owned\n')

    class FinishedProcess:
        def wait(self, timeout=None):
            return 0

    def popen(command):
        assert 'run/usim/config.json' in transferred
        commands.append(list(command))
        return FinishedProcess()

    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr(subprocess, 'Popen', popen)
    if selection == 'environment':
        monkeypatch.setenv('USIM_GAZEBO_ASSET_TRANSFER', 'copy')
        simulator = GazeboSimulator(engine=engine)
    else:
        simulator = GazeboSimulator(engine=engine, asset_transfer='copy')
    # When: the worker is launched using copied inputs.
    simulator.run(replace(config, headless=True))
    # Then: runtime, rewritten descriptions and adjacent resources reach it.
    assert [command[1] for command in commands] == ['create', 'cp', 'start', 'rm']
    assert transferred['assets/0/texture.dat'] == b'caller-owned resource'
    assert transferred['assets/0/world.sdf'] == config.world.read_bytes()
    assert 'run/usim/runtime/usim_gazebo/runner.py' in transferred
    payload = json.loads(transferred['run/usim/config.json'])
    assert payload['configuration']['world'] == '/run/usim/world.sdf'
    assert texture.read_bytes() == b'caller-owned resource'


def test_failed_asset_transfer_removes_its_owned_container(
    config: SimulationConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: creation works but copying the prepared archive fails.
    commands = []

    def run(command, **kwargs):
        commands.append(list(command))
        if command[1] == 'cp':
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0, stdout='owned\n')

    monkeypatch.setattr(subprocess, 'run', run)
    # When: transfer fails before the worker can start.
    with pytest.raises(subprocess.CalledProcessError):
        GazeboSimulator(asset_transfer='copy').run(replace(config, headless=True))
    # Then: the container is removed without a start or global listing.
    assert [command[1] for command in commands] == ['create', 'cp', 'rm']
    assert commands[-1][-1] == commands[0][commands[0].index('--name') + 1]


def test_copy_mode_recovers_smoke_outputs_before_removing_worker(
    config: SimulationConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Given: smoke evidence is written inside the remote worker.
    commands = []
    output = tmp_path / 'evidence'

    def run(command, **kwargs):
        commands.append(list(command))
        if command[1] == 'cp' and command[2].endswith(':/output/.'):
            (Path(command[3]) / 'smoke.json').write_text('{"status":"passed"}')
        return subprocess.CompletedProcess(command, 0, stdout='owned\n')

    class FinishedProcess:
        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr(subprocess, 'Popen', lambda command: FinishedProcess())
    # When: the caller requests persisted smoke evidence.
    GazeboSimulator(asset_transfer='copy').smoke(replace(config, headless=True), output)
    # Then: output transfer precedes container removal and reaches the caller.
    assert [command[1] for command in commands] == ['create', 'cp', 'cp', 'rm']
    assert json.loads((output / 'smoke.json').read_text()) == {'status': 'passed'}
