"""Separate episode and continuous native execution drivers."""

from __future__ import annotations

from collections.abc import Mapping
from threading import Event
from typing import TYPE_CHECKING

from usim.capability import Phase
from usim.model import ModelError
from usim.types import Observation, StepResult

if TYPE_CHECKING:
    from usim_isaacsim.contracts import SessionView as Session


class EpisodeExecution:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.steps = 0

    def reset(self, seed: int | None = None) -> tuple[Observation, Mapping[str, float]]:
        """Reset authored poses and joint initial states; no randomization is implicit."""
        self.session.require_ready()
        self.session._native.initialize()
        self.session._sensors.invalidate()
        self.steps = 0
        self.session._phase = Phase.PAUSED
        return self.session.state.get_observation(), {}

    def step(self) -> StepResult:
        """Advance exactly one native physics step using already applied targets."""
        self.session.require_ready()
        native = self.session._native
        assert native.world is not None
        self.session._phase = Phase.RUNNING
        succeeded = False
        try:
            native.world.play()
            native.world.step(render=self.session._execution.render_mode == 'human')
            native.world.pause()
            succeeded = True
        finally:
            self.session._phase = Phase.PAUSED if succeeded else Phase.FAILED
        self.steps += 1
        return StepResult(
            self.session.state.get_observation(),
            0.0,
            False,
            self.steps >= self.session._execution.max_episode_steps,
            {'task_reward': False},
        )


class ContinuousExecution:
    """Own the blocking native loop, without an episode step facade."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self._stop = Event()

    def run(self) -> None:
        self.session.require_ready()
        native = self.session._native
        assert native.world is not None
        self._stop.clear()
        self.session._phase = Phase.RUNNING
        succeeded = False
        try:
            native.world.play()
            app = self.session._runtime.app
            if app is None:
                raise ModelError('Isaac application has closed')
            while app.is_running() and not self._stop.is_set():
                native.world.step(render=self.session._execution.render_mode == 'human')
            native.world.pause()
            succeeded = True
        finally:
            self.session._phase = Phase.PAUSED if succeeded else Phase.FAILED

    def stop(self) -> None:
        self._stop.set()
