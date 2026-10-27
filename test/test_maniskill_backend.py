"""Grouped ManiSkill contracts exercised through an in-memory native API fixture."""

import numpy as np
import pytest

from usim.capability import CapabilityRequest, Phase, SupportStatus, UnsupportedOperation
from usim.facets import ControlTargets
from usim.model import ImportRequirements, ModelError, ModelSource
from usim.runtime import RuntimeSession
from usim.scene import (
    ReplaceModel,
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
    SensorSpec,
)
from usim_maniskill.backend import Backend


from packages.maniskill.test.native_env import direct, native as native, sources as sources
from packages.maniskill.test.native_models import DynamicComponent, StaticComponent
from packages.maniskill.test.scene_contracts import (
    test_state_parameter_edits_and_topology_rebind as test_state_parameter_edits_and_topology_rebind,
    test_model_replacement_reconstructs_native_objects as test_model_replacement_reconstructs_native_objects,
    test_failed_native_build_and_step_release_resources as test_failed_native_build_and_step_release_resources,
    test_failed_reconstruction_invalidates_handles_and_closes_old_world as test_failed_reconstruction_invalidates_handles_and_closes_old_world,
    test_native_task_rewards_receive_submitted_action_without_reapplying as test_native_task_rewards_receive_submitted_action_without_reapplying,
)


from packages.maniskill.test.model_contracts import (
    test_mjcf_defaults_and_native_top_level_includes_are_inspected as test_mjcf_defaults_and_native_top_level_includes_are_inspected,
    test_fidelity_and_unsupported_features_reject_before_native_mutation as test_fidelity_and_unsupported_features_reject_before_native_mutation,
    test_import_and_backend_discovery_are_lazy as test_import_and_backend_discovery_are_lazy,
)


@pytest.mark.parametrize('index', [0, 1])
def test_unregistered_models_use_real_native_factory(native, sources, index):
    source = sources[index]
    spec = direct(
        source, control=ControlSpec(position_joints=('shoulder',), velocity_joints=('wheel',))
    )
    with Backend().open(SceneSpec(agents=(spec,))) as session:
        env = native[0][0]
        assert isinstance(env.robot_uids, type)
        assert env.scene.parsed[0][0] == source.format.value
        assert env.agent.robot.fixed_root_link[0] == (index == 0)
        assert session.agents.describe('robot') == spec
        assert session.state.schema.entity == 'robot'
        assert session.state.schema.batched
        assert [joint.name for joint in session.state.schema.joints] == ['shoulder', 'wheel']
        session.control.apply(ControlTargets(positions={'shoulder': 0.4}, velocities={'wheel': 2}))
        assert env.steps == 0
        np.testing.assert_allclose(env.agent.robot.targets['positions'], [[0.4]])
        np.testing.assert_allclose(env.agent.robot.targets['velocities'], [[2]])
        result = session.execution.step()
        assert result.observation.qpos is not None
        assert result.observation.qpos.shape == (1, 2)
        assert session.phase is Phase.PAUSED
    assert env.closed == 1
    session.close()
    assert env.closed == 1


def test_multiple_entities_free_base_and_per_open_configuration(native, sources):
    robot = direct(sources[0])
    robot = AgentSpec(
        robot.name,
        robot.model,
        requirements=ImportRequirements(base='floating'),
        control=ControlSpec(initial_qpos=(0.2, 0.3)),
    )
    scene = SceneSpec(
        entities=(EntitySpec('object', sources[0], pose=Pose((1, 2, 3))),),
        agents=(robot,),
    )
    with (
        Backend().open(scene) as first,
        Backend().open(SceneSpec(agents=(direct(sources[0]),))) as second,
    ):
        assert first._require().robot_uids is not second._require().robot_uids
        assert not first._require().agent.robot.fixed_root_link[0]
        assert second._require().agent.robot.fixed_root_link[0]
        assert first.state.schema.entity == 'robot'
        np.testing.assert_allclose(np.asarray(first.state.get_qpos(), dtype=float), [[0.2, 0.3]])
        np.testing.assert_allclose(
            first.state_for('object').get_observation().extra['pose'][0, :3], [1, 2, 3]
        )
        assert len(first._require().scene.built) == 2


def test_one_link_static_reconstruction_avoids_dynamic_inertia_setters(native, sources, tmp_path):
    path = tmp_path / 'box.urdf'
    path.write_text(
        '<robot name="box"><link name="box"><inertial><mass value="2"/>'
        '<inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>'
        '<collision><geometry><box size=".1 .1 .1"/></geometry></collision></link></robot>'
    )
    source = ModelSource.from_path(path)
    fixed = EntitySpec(
        'fixed', source, requirements=ImportRequirements(role='rigid_body', base='fixed')
    )
    free = EntitySpec(
        'free', source, requirements=ImportRequirements(role='rigid_body', base='floating')
    )
    with Backend().open(SceneSpec(entities=(free,), agents=(direct(sources[0]),))) as session:
        dynamic = session._entities['free']
        assert isinstance(dynamic.component, DynamicComponent)
        assert dynamic.component.mass == 2.0
        old_handle = session.scene.handle('robot')
        session.scene.apply(session.scene.plan(SpawnEntity(fixed), require_same_episode=False))
        assert isinstance(session._entities['fixed'].component, StaticComponent)
        assert session._entities['free'].component.mass == 2.0
        assert session.state_for('fixed').get_qpos().shape == (1, 0)
        with pytest.raises(ModelError, match='old topology'):
            session.scene.resolve(old_handle)


def test_registered_controller_actions_keep_native_semantics(native):
    spec = AgentSpec(
        'robot',
        RegisteredAgentRef('maniskill', 'registered'),
        control=ControlSpec(native_mode='native_delta'),
    )
    with Backend().open(SceneSpec(agents=(spec,))) as session:
        action = np.array([[0.7]])
        session.control.set_native_action(action)
        assert session._require().agent.native_actions[0] is action
        with pytest.raises(UnsupportedOperation):
            session.control.set_positions({'shoulder': 0.2})
        assert session._require().steps == 0
        session.execution.step()
        assert session._require().steps == 1


def test_typed_cameras_and_batch_targets(native, sources):
    spec = direct(
        sources[0], sensors=(SensorSpec('eye', 'rgbd', parent='arm', width=4, height=2, rate=30),)
    )
    with Backend().open(
        SceneSpec(agents=(spec,)), ExecutionSpec(backend='gpu', n_envs=2, obs_mode='rgbd')
    ) as session:
        session.control.set_positions({'shoulder': (0.2, 0.4)})
        np.testing.assert_allclose(
            session._require().agent.robot.targets['positions'][:, 0], [0.2, 0.4]
        )
        obs = session.state.get_observation()
        assert obs.rgb_image is not None
        assert obs.depth_image is not None
        assert obs.rgb_image.shape == (2, 2, 4, 3)
        assert obs.depth_image.shape == (2, 2, 4, 1)
        np.testing.assert_allclose(np.asarray(obs.depth_image, dtype=float), 1.5)
        assert session.state.schema.sensors[1].unit == 'm'
        assert session._require().cameras[0].mount.name == 'arm'
        assert session._require().scene.sync[-3:] == ['apply', 'kinematics', 'fetch']
        with pytest.raises(ModelError, match='batch'):
            session.control.set_positions({'shoulder': (1, 2, 3)})


def test_all_controls_are_validated_before_writing(native, sources):
    spec = direct(
        sources[0],
        control=ControlSpec(
            position_joints=('shoulder',),
            effort_joints=('wheel',),
        ),
    )
    with Backend().open(SceneSpec(agents=(spec,))) as session:
        robot = session._require().agent.robot
        with pytest.raises(ModelError):
            session.control.apply(
                ControlTargets(
                    positions={'shoulder': 1},
                    velocities={'wheel': 2},
                )
            )
        assert robot.targets == {}
        session.control.set_efforts({'wheel': 3})
        np.testing.assert_allclose(robot.qf, [[0, 3]])
        assert session._require().steps == 0


def test_invalid_replacement_controls_reject_before_reconstruction(native, sources, tmp_path):
    path = tmp_path / 'other.urdf'
    path.write_text('<robot name="other"><link name="root"/></robot>')
    with Backend().open(
        SceneSpec(
            agents=(
                direct(
                    sources[0],
                    control=ControlSpec(position_joints=('shoulder',)),
                ),
            )
        )
    ) as session:
        old = session._require()
        plan = session.scene.plan(
            ReplaceModel('robot', ModelSource.from_path(path)),
            require_same_episode=False,
        )
        assert plan.support.status is SupportStatus.UNSUPPORTED
        with pytest.raises(UnsupportedOperation):
            session.scene.apply(plan)
        assert old.closed == 0
        assert session._require() is old


def test_episode_reset_limits_and_closed_readers(native, sources):
    session = Backend().open(
        SceneSpec(agents=(direct(sources[0]),)), ExecutionSpec(max_episode_steps=1)
    )
    state = session.state
    assert np.asarray(session.execution.step().truncated).all()
    session.execution.reset(seed=7)
    assert session._require().seed == 7
    assert session.phase is Phase.READY
    with pytest.raises(UnsupportedOperation):
        session.execution.run()
    session.close()
    with pytest.raises(ModelError):
        state.get_qpos()


def test_multiple_passive_entities_require_explicit_state_selection(native, sources):
    scene = SceneSpec(entities=(EntitySpec('first', sources[0]), EntitySpec('second', sources[0])))
    with Backend().open(scene) as session:
        with pytest.raises(ModelError, match='for_entity'):
            session.state.get_qpos()
        assert session.state.for_entity('second').schema.entity == 'second'


def test_collections_and_assessments_use_actual_session_context(native, sources, monkeypatch):
    backend = Backend()
    scene = SceneSpec(
        entities=(EntitySpec('object', sources[0]),),
        agents=(direct(sources[0]),),
    )
    with backend.open(scene, ExecutionSpec(backend='gpu', n_envs=2)) as session:
        assert isinstance(session, RuntimeSession)
        assert session.state.for_entity('object').schema.entity == 'object'
        assert session.state_for('object').schema == session.state.for_entity('object').schema
        session.control.for_entity('robot').apply(
            ControlTargets(positions={'shoulder': (0.2, 0.3)})
        )
        assert session._require().steps == 0
        with pytest.raises(UnsupportedOperation):
            session.control.for_entity('object')
        captured = []
        assess = backend.assess

        def capture(request):
            captured.append(request)
            return assess(request)

        monkeypatch.setattr(backend, 'assess', capture)
        assert session.assess(
            CapabilityRequest('state.read', phase=Phase.BUILDING, n_envs=99)
        ).supported
        assert captured[-1].phase is Phase.READY
        assert captured[-1].n_envs == 2
        assert not session.assess(CapabilityRequest('scene.spawn')).supported
        assert not session.assess(CapabilityRequest('execution.continuous')).supported
        assert not session.assess(CapabilityRequest('control.velocity')).supported
    assert not session.assess(CapabilityRequest('state.read')).supported
