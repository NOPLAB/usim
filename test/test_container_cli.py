"""All-in-one Docker launcher keeps SDK interpreter environments separate."""

import sys
from unittest.mock import Mock

import pytest

from usim.container_cli import main


@pytest.mark.parametrize(
    ('arguments', 'executable'),
    [
        (['run', '--engine', 'genesis'], '/opt/usim/genesis/bin/python'),
        (['run', '--engine', 'maniskill'], '/opt/usim/maniskill/bin/python'),
        (['run', '--engine', 'bullet'], '/opt/usim/bullet/bin/python'),
        (['run', '--engine', 'mujoco'], '/opt/usim/mujoco/bin/python'),
        (['run', '--engine', 'mujoco-warp'], '/opt/usim/mujoco-warp/bin/python'),
        (['run', '--engine', 'isaacsim'], '/isaac-sim/python.sh'),
        (
            ['simulate', '--backend', 'isaac', '--world', 'room.usd', '--robot-urdf', 'r.urdf'],
            '/isaac-sim/python.sh',
        ),
        (
            ['simulate', '--backend', 'gazebo', '--world', 'room.sdf', '--robot-urdf', 'r.urdf'],
            None,
        ),
        (['convert-world', '--world', 'room.sdf', '--out', 'room.usd'], '/isaac-sim/python.sh'),
        (['backends'], None),
        (['create-robot', '--out', 'r.urdf'], None),
    ],
)
def test_command_executes_only_selected_interpreter(monkeypatch, arguments, executable):
    # Given a parsed CLI command and independently installed SDK interpreters.
    monkeypatch.setenv('ACCEPT_EULA', 'Y')
    monkeypatch.setattr(sys, 'argv', ['usim', *arguments])
    execute = Mock()
    monkeypatch.setattr('usim.container_cli.os.execv', execute)

    # When the container launcher routes the command.
    main()

    # Then process replacement preserves every original CLI argument.
    expected = sys.executable if executable is None else executable
    execute.assert_called_once_with(expected, [expected, '-m', 'usim.cli', *arguments])


def test_isaac_execution_requires_existing_eula_environment(monkeypatch):
    # Given Isaac selection without the existing EULA acceptance environment.
    monkeypatch.delenv('ACCEPT_EULA', raising=False)
    monkeypatch.setattr(sys, 'argv', ['usim', 'run', '--engine', 'isaacsim'])
    execute = Mock()
    monkeypatch.setattr('usim.container_cli.os.execv', execute)

    # When selecting the Isaac SDK.
    with pytest.raises(SystemExit) as error:
        main()

    # Then no runtime process is started.
    assert error.value.code == 2
    execute.assert_not_called()


def test_discovery_does_not_require_isaac_acceptance(monkeypatch):
    # Given a discovery command without Isaac acceptance.
    monkeypatch.delenv('ACCEPT_EULA', raising=False)
    monkeypatch.setattr(sys, 'argv', ['usim', 'backends'])
    execute = Mock()
    monkeypatch.setattr('usim.container_cli.os.execv', execute)

    # When dispatching discovery.
    main()

    # Then only the core interpreter is used.
    execute.assert_called_once_with(sys.executable, [sys.executable, '-m', 'usim.cli', 'backends'])
