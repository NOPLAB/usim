"""Explicit episode advancement, with native reset and command clearing."""

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from usim.capability import Phase
from usim.types import Observation, StepResult

from .native import numpy, resolve_joint_name

if TYPE_CHECKING:
    from .contracts import SessionView as Session


class Episode:
    def __init__(self, session: 'Session'):
        self._session = session

    def _observation(self) -> Observation:
        if not self._session.entities:
            empty = np.empty((self._session.options.n_envs, 0))
            return Observation(qpos=empty, qvel=empty.copy())
        primary_available = len(self._session.controllers) == 1 or len(self._session.entities) == 1
        observation = self._session.state.get_observation() if primary_available else Observation()
        observation.extra['entities'] = {
            name: self._session.state.for_entity(name).get_observation()
            for name in self._session.entities
        }
        return observation

    def reset(self, seed: int | None = None) -> tuple[Observation, Mapping[str, float]]:
        """Restore authored native state; resetting never advances physics."""
        self._session.require_usable()
        ids = np.arange(self._session.options.n_envs)
        with self._session.mutation():
            self._session.native.reset(envs_idx=ids)
            self.clear_commands()
            self._session.invalidate_sensors()
            self._session.steps[:] = 0
            self._session._phase = Phase.READY
            observation = self._observation()
        # The grouped scene has no randomized task reset. A seed changes no physics state.
        return observation, {}

    def clear_commands(self) -> None:
        """Clear every native controller, then initialize the declared agent channels."""
        count = self._session.options.n_envs
        ids = np.arange(count)
        for bound in self._session.entities.values():
            entity = bound.native
            if entity.n_dofs:
                entity.control_dofs_force(np.zeros((count, entity.n_dofs)))
        for name, control in self._session.controllers.items():
            bound = self._session.entities[name]
            entity = bound.native
            indices = [
                entity.get_joint(resolve_joint_name(bound, joint)).dofs_idx_local[0]
                for joint in control.joint_names
            ]
            if control.initial_qpos is not None:
                entity.set_dofs_position(
                    np.tile(control.initial_qpos, (count, 1)), indices, envs_idx=ids
                )
                entity.set_dofs_velocity(np.zeros((count, len(indices))), indices, envs_idx=ids)
            for mode, names in (
                ('position', control.position_joints),
                ('velocity', control.velocity_joints),
                ('force', control.effort_joints),
            ):
                if names:
                    indices = [
                        entity.get_joint(resolve_joint_name(bound, joint)).dofs_idx_local[0]
                        for joint in names
                    ]
                    values = (
                        numpy(entity.get_dofs_position(indices))
                        if mode == 'position'
                        else np.zeros((count, len(indices)))
                    )
                    getattr(entity, f'control_dofs_{mode}')(values, indices)

    def step(self) -> StepResult:
        """Advance exactly one native step using previously applied controls."""
        self._session.require_usable()
        with self._session.mutation():
            self._session._phase = Phase.RUNNING
            self._session.native.step()
            self._session.steps[:] += 1
            self._session._phase = Phase.PAUSED
            count = self._session.options.n_envs
            return StepResult(
                observation=self._observation(),
                reward=np.zeros(count),
                terminated=np.zeros(count, dtype=bool),
                truncated=self._session.steps >= self._session.options.max_episode_steps,
                info=[{} for _ in range(count)],
            )
