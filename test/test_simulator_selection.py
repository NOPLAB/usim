"""CLI simulator selection follows installed execution providers."""

from pathlib import Path
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from usim import Observation, StepResult
from usim.cli import build_parser
from usim.runtime_cli import configured


@pytest.mark.parametrize('name', ['gazebo', 'isaacsim', 'isaac', 'external'])
def test_continuous_selection_dispatches_selected_provider(monkeypatch, name):
    # Given installed runners, including a third-party provider.
    monkeypatch.setattr('usim.list_runners', lambda: ['gazebo', 'isaacsim', 'external'])
    runner = Mock()
    factory = Mock(return_value=runner)
    monkeypatch.setattr('usim.factory.create_runner', factory)
    args = build_parser().parse_args(
        [
            'simulate',
            '--backend',
            name,
            '--world',
            'room.sdf',
            '--robot-urdf',
            'robot.urdf',
            '--no-ros',
        ]
    )

    # When the command executes.
    args.handler(args)

    # Then the selected public runner receives the shared configuration.
    expected = 'isaacsim' if name == 'isaac' else name
    assert factory.call_args.args == (expected,)
    if expected != 'isaacsim':
        assert factory.call_args.kwargs == {}
    runner.run.assert_called_once_with(configured(args))


@pytest.mark.parametrize('name', ['genesis', 'maniskill', 'isaacsim'])
def test_episode_selection_dispatches_native_configuration(monkeypatch, name):
    # Given an engine boundary with a real observation and registered robot.
    observation = Observation(qpos=np.array([[0.2]]))
    simulator = Mock()
    simulator.reset.return_value = (observation, {})
    simulator.step.return_value = StepResult(observation, 0.0, False, False, {})
    factory = Mock(return_value=simulator)
    monkeypatch.setattr('usim.create_simulator', factory)
    args = build_parser().parse_args(
        [
            'run',
            '--engine',
            name,
            '--robot-uid',
            'external',
            '--joint-names',
            'hinge',
            '--joint-stiffness',
            '100',
            '--joint-damping',
            '10',
            '--steps',
            '1',
            '--action',
            '0.3',
            '--seed',
            '7',
        ]
    )

    # When the command executes through the public episode API.
    result = args.handler(args)

    # Then model configuration and native actions reach only that engine.
    expected = name
    selected, config = factory.call_args.args
    assert selected == expected
    assert config.robot_path is None
    assert config.robot_uid == 'external'
    assert config.joint_names == ('hinge',)
    assert config.joint_stiffness == (100.0,)
    assert config.joint_damping == (10.0,)
    simulator.reset.assert_called_once_with(seed=7)
    np.testing.assert_allclose(simulator.step.call_args.args[0], [0.3])
    simulator.close.assert_called_once_with()
    assert result == {'engine': expected, 'steps': 1, 'qpos_shape': [1, 1], 'rgb_shape': []}


@pytest.mark.parametrize('name', ['genesis', 'maniskill', 'isaacsim'])
def test_external_model_uses_grouped_loading_and_named_targets(monkeypatch, tmp_path, name):
    # Given a model-first backend boundary and a real finite observation.
    robot = tmp_path / 'robot.urdf'
    robot.write_text('<robot name="external"/>', encoding='utf-8')
    provider = MagicMock()
    session = provider.open.return_value.__enter__.return_value
    observation = Observation(qpos=np.array([[0.2, 0.4]]))
    session.execution.reset.return_value = (observation, {})
    session.state.for_entity.return_value.get_observation.return_value = observation
    factory = Mock(return_value=provider)
    monkeypatch.setattr('usim.create_backend', factory)
    args = build_parser().parse_args(
        [
            'run',
            '--engine',
            name,
            '--robot',
            str(robot),
            '--joint-names',
            'hinge',
            'wheel',
            '--velocity-joints',
            'wheel',
            '--steps',
            '1',
            '--action',
            '0.3',
            '0.5',
            '--seed',
            '7',
        ]
    )

    # When executing an external model.
    result = args.handler(args)

    # Then the backend owns loading, physical control and session cleanup.
    factory.assert_called_once_with(name)
    scene, execution = provider.open.call_args.args
    agent = scene.agents[0]
    assert agent.model.path == robot.resolve()
    assert agent.control.joint_names == ('hinge', 'wheel')
    assert agent.control.position_joints == ('hinge',)
    assert agent.control.velocity_joints == ('wheel',)
    assert agent.requirements.base == 'fixed'
    assert execution.backend == 'cpu'
    session.execution.reset.assert_called_once_with(seed=7)
    target = session.control.for_entity.return_value.apply.call_args.args[0]
    assert target.positions == {'hinge': 0.3}
    assert target.velocities == {'wheel': 0.5}
    session.execution.step.assert_called_once_with()
    provider.open.return_value.__exit__.assert_called_once()
    assert result['qpos_shape'] == [1, 2]


@pytest.mark.parametrize(
    'arguments',
    [
        [
            'simulate',
            '--backend',
            'genesis',
            '--world',
            'room.sdf',
            '--robot-urdf',
            'robot.urdf',
        ],
        ['run', '--engine', 'gazebo'],
        ['run', '--engine', 'missing'],
    ],
)
def test_incompatible_execution_selection_is_rejected(arguments):
    # Given a simulator without the requested execution capability.
    # When parsing that command.
    # Then it fails before any native engine is loaded.
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(arguments)
    assert error.value.code == 2


def test_gazebo_container_engine_is_distinct_from_backend(monkeypatch):
    # Given an explicit Gazebo container configuration.
    factory = Mock()
    monkeypatch.setattr('usim.factory.create_runner', factory)
    args = build_parser().parse_args(
        [
            'simulate',
            '--backend',
            'gazebo',
            '--world',
            'room.sdf',
            '--robot-urdf',
            'robot.urdf',
            '--engine',
            'docker',
            '--image',
            'custom:gazebo',
            '--fastdds-profile',
            'dds.xml',
        ]
    )

    # When launching the runner.
    args.handler(args)

    # Then the container engine remains distinct from the simulator choice.
    factory.assert_called_once_with(
        'gazebo', engine='docker', image='custom:gazebo', fastdds_profile=Path('dds.xml')
    )
