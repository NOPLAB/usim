"""External model CLI validation and grouped session ownership."""

from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from usim import Observation
from usim.cli import build_parser


@pytest.mark.parametrize(
    'options',
    [
        ['--action', '0.1', '0.2'],
        ['--action', 'nan'],
        ['--action', '0.1', '--robot-uid', 'registered'],
        ['--action', '0.1', '--velocity-joints', 'unknown'],
    ],
)
def test_invalid_model_controls_fail_before_opening_backend(monkeypatch, tmp_path, options):
    # Given an external model and invalid control inputs.
    robot = tmp_path / 'robot.urdf'
    robot.write_text('<robot name="external"/>', encoding='utf-8')
    factory = Mock()
    monkeypatch.setattr('usim.create_backend', factory)
    args = build_parser().parse_args(
        [
            'run',
            '--engine',
            'maniskill',
            '--robot',
            str(robot),
            '--joint-names',
            'hinge',
            '--steps',
            '1',
            *options,
        ]
    )

    # When executing the model command.
    with pytest.raises(ValueError):
        args.handler(args)

    # Then validation fails before acquiring native resources.
    factory.assert_not_called()


def test_grouped_step_failure_releases_session(monkeypatch, tmp_path):
    # Given an opened grouped session whose native step fails.
    robot = tmp_path / 'robot.urdf'
    robot.write_text('<robot name="external"/>', encoding='utf-8')
    provider = MagicMock()
    session = provider.open.return_value.__enter__.return_value
    session.execution.reset.return_value = (Observation(qpos=np.zeros((1, 1))), {})
    session.execution.step.side_effect = RuntimeError('native step failed')
    monkeypatch.setattr('usim.create_backend', Mock(return_value=provider))
    args = build_parser().parse_args(
        [
            'run',
            '--engine',
            'maniskill',
            '--robot',
            str(robot),
            '--joint-names',
            'hinge',
            '--steps',
            '1',
            '--action',
            '0.2',
        ]
    )

    # When stepping propagates a native failure.
    with pytest.raises(RuntimeError, match='native step failed'):
        args.handler(args)

    # Then context management receives that error and owns cleanup.
    assert provider.open.return_value.__exit__.call_args.args[0] is RuntimeError
