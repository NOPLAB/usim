"""Topology, failure lifecycle, and registered-task reward contracts."""

from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest

from usim.capability import Phase, SupportStatus, UnsupportedOperation
from usim.model import ModelError
from usim.scene import (
    EditArticulation,
    RemoveEntity,
    ReplaceModel,
    SetJointState,
    SetParameter,
    SetPose,
    SpawnEntity,
)
from usim.specs import (
    AgentSpec,
    ControlSpec,
    EntitySpec,
    ExecutionSpec,
    Pose,
    RegisteredAgentRef,
    SceneSpec,
)
from usim_maniskill.backend import Backend
from packages.maniskill.test.native_env import direct


def test_state_parameter_edits_and_topology_rebind(native, sources):
    with Backend().open(SceneSpec(agents=(direct(sources[0]),))) as session:
        state = session.state
        control = session.control
        handle = session.scene.handle('robot')
        session.scene.apply(session.scene.plan(SetJointState('robot', (('wheel', 0.8),))))
        np.testing.assert_allclose(np.asarray(state.get_qpos(), dtype=float), [[0, 0.8]])
        session.scene.apply(session.scene.plan(SetParameter('robot', 'joint.wheel.friction', 0.2)))
        assert session._require().agent.robot.joints[1].friction == 0.2
        session.scene.apply(session.scene.plan(SetPose('robot', Pose((1, 0, 0)))))
        assert session.scene.resolve(handle) == 'robot'
        denied = session.scene.plan(SpawnEntity(EntitySpec('object', sources[0])))
        assert denied.support.status is SupportStatus.UNSUPPORTED
        with pytest.raises(UnsupportedOperation):
            session.scene.apply(denied)
        old = session._require()
        session.scene.apply(
            session.scene.plan(
                SpawnEntity(EntitySpec('object', sources[0])),
                require_same_episode=False,
            )
        )
        assert old.closed == 1
        assert session._require() is not old
        assert session.state_for('object').schema.entity == 'object'
        for read in (state.get_qpos, lambda: control.set_positions({'wheel': 1})):
            with pytest.raises(ModelError, match='old topology'):
                read()
        with pytest.raises(ModelError):
            session.scene.resolve(handle)
        session.scene.apply(
            session.scene.plan(
                RemoveEntity('object'),
                require_same_episode=False,
            )
        )
        assert 'object' not in session._entities


@pytest.mark.parametrize('change_type', [ReplaceModel, EditArticulation])
def test_model_replacement_reconstructs_native_objects(native, sources, change_type):
    with Backend().open(SceneSpec(agents=(direct(sources[0]),))) as session:
        old = session._require()
        session.scene.apply(
            session.scene.plan(
                change_type('robot', sources[0]),
                require_same_episode=False,
            )
        )
        assert old.closed == 1
        assert session._require().agent.robot is not old.agent.robot
        session.control.set_positions({'shoulder': 0.5})


def test_failed_native_build_and_step_release_resources(native, sources):
    native[1].fail_build = True
    with pytest.raises(RuntimeError, match='build failed'):
        Backend().open(SceneSpec(agents=(direct(sources[0]),)))
    assert native[0][0].closed == 1
    native[1].fail_build = False
    session = Backend().open(SceneSpec(agents=(direct(sources[0]),)))
    handle = session.scene.handle('robot')
    native[1].fail_step = True
    with pytest.raises(RuntimeError, match='step failed'):
        session.execution.step()
    assert session.phase is Phase.FAILED
    assert native[0][-1].closed == 1
    with pytest.raises(ModelError):
        session.scene.resolve(handle)


def test_failed_reconstruction_invalidates_handles_and_closes_old_world(native, sources):
    session = Backend().open(SceneSpec(agents=(direct(sources[0]),)))
    old = session._require()
    state = session.state
    plan = session.scene.plan(
        SpawnEntity(EntitySpec('new', sources[0])),
        require_same_episode=False,
    )
    native[1].fail_build = True
    with pytest.raises(RuntimeError, match='build failed'):
        session.scene.apply(plan)
    assert old.closed == 1
    assert native[0][-1].closed == 1
    assert session.phase is Phase.FAILED
    assert not session._entities
    with pytest.raises(ModelError):
        state.get_qpos()


def test_native_task_rewards_receive_submitted_action_without_reapplying(native, monkeypatch):
    class Task(native[1].BaseEnv):
        def _load_scene(self, options):
            self._sensor_configs = {
                'built_in': SimpleNamespace(uid='built_in', width=3, height=2, mount='base')
            }

        def _initialize_episode(self, env_idx, options):
            self.agent.reset()

        def _get_obs_agent(self):
            return self.agent.get_proprioception()

        @property
        def _default_sensor_configs(self):
            return []

        def _step_action(self, action):
            assert action is None
            return action

        def step(self, action):
            applied = self._step_action(action)
            assert applied is not None
            return (
                self.get_obs(),
                np.sum(applied, axis=1),
                np.array([False]),
                np.array([False]),
                {},
            )

    gym = ModuleType('gymnasium')
    gym.__dict__['make'] = lambda task, **kwargs: Task(**kwargs)
    monkeypatch.setitem(sys.modules, 'gymnasium', gym)
    spec = AgentSpec(
        'robot',
        RegisteredAgentRef('maniskill', 'registered'),
        control=ControlSpec(native_mode='native_delta'),
    )
    with Backend().open(
        SceneSpec(agents=(spec,)), ExecutionSpec(native_task='Native-v0', obs_mode='rgbd')
    ) as session:
        with pytest.raises(ModelError, match='Submit'):
            session.execution.step()
        session.control.set_native_action(np.array([0.3]))
        np.testing.assert_allclose(np.asarray(session.execution.step().reward, dtype=float), [0.3])
        assert len(session._require().agent.native_actions) == 1
        assert session.state.schema.sensors[0].name == 'built_in.rgb'
        assert session.state.schema.sensors[0].frame == 'base'
