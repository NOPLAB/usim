"""Native episode execution independent of target submission."""

from typing import TYPE_CHECKING

from usim.capability import Phase
from usim.model import ModelError
from usim.specs import RegisteredAgentRef
from usim.types import Observation, StepResult
from usim_maniskill.facets import numpy_copy, observation
from usim_maniskill.native import native_pose

if TYPE_CHECKING:
    from usim_maniskill.session import Session


class Episode:
    """Advance native episodes while keeping target submission independent of stepping."""

    def __init__(self, session: 'Session'):
        self._session = session
        self._steps = 0

    def reset(self, seed: int | None = None):
        session = self._session
        env = session._require()
        succeeded = False
        try:
            _, info = env.reset(seed=seed)
            if session.spec.agents:
                spec = session.spec.agents[0]
                if isinstance(spec.model, RegisteredAgentRef):
                    agent = env.unwrapped.agent
                    agent.robot.set_pose(native_pose(spec.pose))
                    if spec.control.initial_qpos is not None:
                        agent.reset(spec.control.initial_qpos)
                    session._sync()
                    agent.controller.reset()
            session._targets.clear()
            env.unwrapped.usim_action = None
            self._steps = 0
            session._phase = Phase.READY
            result = self._observation(), info
            succeeded = True
            return result
        finally:
            if not succeeded:
                session._fail()

    def _observation(self) -> Observation:
        session = self._session
        name = (
            session.spec.agents[0].name
            if session.spec.agents
            else next(iter(session._entities), '')
        )
        return observation(session, name)

    def step(self) -> StepResult:
        import numpy as np

        session = self._session
        env = session._require()
        if session.phase is Phase.RUNNING:
            raise ModelError('A native episode step is already running')
        if session.options.native_task != 'JointControl' and env.unwrapped.usim_action is None:
            raise ModelError('Submit a native controller action before stepping the native task')
        session._phase = Phase.RUNNING
        succeeded = False
        try:
            _, reward, terminated, truncated, info = env.step(None)
            self._steps += 1
            truncated = numpy_copy(truncated) | (self._steps >= session.options.max_episode_steps)
            result = StepResult(
                self._observation(),
                numpy_copy(reward),
                numpy_copy(terminated),
                np.asarray(truncated),
                info,
            )
            session._phase = Phase.PAUSED
            succeeded = True
            return result
        finally:
            if not succeeded:
                session._fail()
