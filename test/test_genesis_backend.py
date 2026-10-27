"""Deterministic grouped Genesis contracts against a stateful native boundary."""

import sys
from dataclasses import replace
from types import SimpleNamespace
from xml.etree import ElementTree

import numpy as np
import pytest

from usim.capability import CapabilityRequest, Phase, SupportStatus
from usim.facets import ControlTargets
from usim.model import ModelError, ModelFormat, ModelSource
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
    SensorSpec,
)
from usim_genesis import Backend
from test.test_genesis import Entity, Scene


class LoadedEntity(Entity):
    """Native fixture preserves independent free-base and scalar-joint coordinates."""

    def __init__(self, morph):
        self.root = morph.file
        self.scalar_names = [joint.get('name') for joint in self.root.iter('joint')]
        super().__init__([f'base{i}' for i in range(6)] + self.scalar_names, morph.pos)
        self.n_dofs = len(self.names)
        self.joints = [
            SimpleNamespace(
                name='root',
                n_qs=7,
                type=SimpleNamespace(name='FREE'),
                qs_idx_local=list(range(7)),
                dofs_idx_local=list(range(6)),
            )
        ] + [
            SimpleNamespace(
                name=name,
                n_qs=1,
                type=SimpleNamespace(name='REVOLUTE'),
                qs_idx_local=[index + 7],
                dofs_idx_local=[index + 6],
            )
            for index, name in enumerate(self.scalar_names)
        ]
        self.links = [SimpleNamespace(is_fixed=False)]
        self.base_link = SimpleNamespace(get_pos=lambda: self.pos, idx_local=0)
        self.initial_quat = morph.quat

    def build(self, count):
        super().build(count)
        self.qpos[:, 6:] = 0.25
        self.quat[:] = self.initial_quat

    def get_joint(self, name):
        return next((joint for joint in self.joints if joint.name == name), None)

    def get_link(self, name):
        return self.base_link if name == 'root' else None

    def get_qpos(self):
        return np.column_stack((self.pos, self.quat, self.qpos[:, 6:]))

    def get_dofs_velocity(self, indices=None):
        return self.qvel.copy() if indices is None else super().get_dofs_velocity(indices)

    def get_quat(self):
        return self.quat.copy()

    def control_dofs_force(self, values, indices=None, envs_idx=None):
        super().control_dofs_force(
            values, np.arange(self.n_dofs) if indices is None else indices, envs_idx
        )

    def set_friction(self, value):
        self.friction = value

    def set_mass(self, value):
        self.mass = value


class LoadedScene(Scene):
    def add_sensor(self, options):
        def read():
            width, height = options.pattern.n_points
            distances = np.tile(np.arange(1, width * height + 1), (self.n_envs, 1))
            return SimpleNamespace(
                distances=distances,
                points=np.stack((distances, distances * 0, distances * 0), axis=-1),
            )

        self.sensor_options = options
        return SimpleNamespace(read=read)

    def add_camera(self, **options):
        camera = SimpleNamespace(samples=0)

        def attach(link, transform):
            camera.link, camera.transform = link, transform

        def move():
            camera.pos = camera.link.get_pos() + camera.transform[:3, 3]

        def render(rgb, depth):
            camera.samples += 1
            width, height = options['res']
            image = np.zeros((self.n_envs, height, width, 3), dtype=np.uint8)
            image[:, :, :, 0] = camera.pos[:, 0, None, None]
            distances = np.full((self.n_envs, height, width), 2.5) if depth else None
            return image, distances, None, None

        camera.attach, camera.move_to_attach, camera.render = attach, move, render
        self.camera = camera
        return camera

    def add_entity(self, morph):
        entity = LoadedEntity(morph)
        entity.idx = len(self.entities)
        self.entities.append(entity)
        self.native.morphs_added.append(morph)
        return entity

    def reset(self, envs_idx):
        super().reset(envs_idx)
        for entity in self.entities:
            entity.qpos[envs_idx, 6:] = 0.25
            entity.pos[envs_idx] = entity.position
            entity.quat[envs_idx] = entity.initial_quat


@pytest.fixture
def grouped_native(monkeypatch):
    module = SimpleNamespace(
        _initialized=False, cpu='cpu', cuda='cuda', backend=None, scenes=[], morphs_added=[]
    )

    def init(backend):
        module._initialized, module.backend = True, backend

    module.init = init
    module.options = SimpleNamespace(SimOptions=SimpleNamespace, VisOptions=SimpleNamespace)
    module.morphs = SimpleNamespace(URDF=SimpleNamespace, MJCF=SimpleNamespace)
    module.sensors = SimpleNamespace(Lidar=SimpleNamespace, SphericalPattern=SimpleNamespace)
    module.utils = SimpleNamespace(
        geom=SimpleNamespace(quat_to_xyz=lambda quaternion, **options: np.zeros(3))
    )
    module.Scene = lambda **options: LoadedScene(module, **options)
    monkeypatch.setitem(sys.modules, 'genesis', module)
    return module


@pytest.fixture
def spec(tmp_path):
    path = tmp_path / 'mobile.xml'
    path.write_text(
        '<robot name="mobile"><joint name="wheel" type="continuous"/>'
        '<joint name="arm" type="revolute"/><joint name="finger" type="prismatic"/>'
        '</robot>'
    )
    source = ModelSource.from_path(path)
    return SceneSpec(
        entities=(EntitySpec('crate', source, Pose((2, 3, 4))),),
        agents=(
            AgentSpec(
                'mobile',
                source,
                ControlSpec(
                    joint_names=('wheel', 'arm', 'finger'),
                    position_joints=('arm',),
                    velocity_joints=('wheel',),
                    effort_joints=('finger',),
                ),
            ),
        ),
    )


def test_two_entities_expose_native_loaded_state_and_free_base_schema(grouped_native, spec):
    # Given two models placed independently; when the grouped scene is built.
    with Backend().open(spec, ExecutionSpec(n_envs=2)) as session:
        # Then state is native authored state, not a fabricated zero observation.
        np.testing.assert_allclose(session.state.get_qpos()[:, -3:], 0.25)
        np.testing.assert_allclose(
            session.state.for_entity('crate').get_qpos()[:, :3], [[2, 3, 4], [2, 3, 4]]
        )
        assert session.state.schema.joints[0].qpos_range == (0, 7)
        assert session.state.schema.joints[0].qvel_range == (0, 6)
        assert session.state.schema.quaternion_order == 'xyzw'
        assert len(grouped_native.scenes[0].entities) == 2
        assert grouped_native.scenes[0].steps == 0


def test_grouped_controller_uses_caller_values_in_declared_order(grouped_native, spec):
    # Given gains and limits ordered differently from native joint indices.
    agent = spec.agents[0]
    control = ControlSpec(
        joint_names=('finger', 'wheel', 'arm'),
        position_joints=('finger', 'arm'),
        velocity_joints=('wheel',),
        joint_stiffness=(200.0, 0.0, 1000.0),
        joint_damping=(20.0, 10.0, 100.0),
        joint_force_limits=(2.0, 5.0, 10.0),
    )
    # When opening the caller-owned model through the grouped API.
    with Backend().open(replace(spec, agents=(replace(agent, control=control),))):
        # Then numeric values and DOF indices retain their explicit correspondence.
        entity = grouped_native.scenes[0].entities[1]
        np.testing.assert_array_equal(entity.kp, [200, 0, 1000])
        np.testing.assert_array_equal(entity.kv, [20, 10, 100])
        lower, upper, indices = entity.force_ranges[0]
        np.testing.assert_array_equal(lower, [-2, -5, -10])
        np.testing.assert_array_equal(upper, [2, 5, 10])
        assert indices == [8, 6, 7]


def test_mixed_controls_apply_without_advancement_then_one_episode_step(grouped_native, spec):
    # Given a mobile arm with three independent control channels.
    with Backend().open(spec, ExecutionSpec(max_episode_steps=1)) as session:
        session.control.apply(
            ControlTargets(
                positions={'arm': 0.8}, velocities={'wheel': 0.4}, efforts={'finger': 0.1}
            )
        )
        assert grouped_native.scenes[0].steps == 0
        # When the explicit episode driver advances.
        result = session.execution.step()
        # Then all three channels act in one articulation and one physics step.
        np.testing.assert_allclose(result.observation.qpos[:, -3:], [[0.65, 0.8, 0.35]])
        assert result.truncated.tolist() == [True]
        assert session.phase is Phase.PAUSED


def test_invalid_mixed_batch_does_not_write_any_targets(grouped_native, spec):
    # Given existing native targets and a valid arm target with an invalid wheel batch.
    with Backend().open(spec) as session:
        entity = grouped_native.scenes[0].entities[1]
        previous = entity.target.copy()
        # When all channels are preflighted.
        with pytest.raises(ModelError):
            session.control.apply(
                ControlTargets(positions={'arm': 1}, velocities={'wheel': [1, 2]})
            )
        # Then neither controller nor physics changed.
        np.testing.assert_array_equal(entity.target, previous)
        assert session.phase is Phase.READY


def test_state_and_parameter_edits_are_native_without_advancement(grouped_native, spec):
    # Given a built scene; when editing native state and a physical parameter.
    with Backend().open(spec) as session:
        for change in (
            SetPose('crate', Pose((7, 8, 9), (0, 0, 1, 0))),
            SetJointState('mobile', (('arm', 0.6),), (('wheel', 0.3),)),
            SetParameter('crate', 'friction', 0.7),
        ):
            session.scene.apply(session.scene.plan(change))
        # Then reads reflect the edits while the physics clock remains unchanged.
        np.testing.assert_allclose(session.state.get_qpos()[:, -2], [0.6])
        np.testing.assert_allclose(session.state.get_qvel()[:, -3], [0.3])
        np.testing.assert_allclose(
            session.state.for_entity('crate').get_qpos()[:, :7], [[7, 8, 9, 0, 0, 1, 0]]
        )
        assert grouped_native.scenes[0].entities[0].friction == 0.7
        assert grouped_native.scenes[0].steps == 0


@pytest.mark.parametrize('kind', ['spawn', 'remove', 'replace', 'articulation'])
def test_rebuild_requires_new_episode_and_invalidates_bound_access(grouped_native, spec, kind):
    # Given persistent velocity targets and a revision-scoped reader.
    with Backend().open(spec) as session:
        previous = session.state.for_entity('mobile')
        session.control.set_velocities({'wheel': 2})
        changes = {
            'spawn': (SpawnEntity(replace(spec.entities[0], name='second_crate')), 3),
            'remove': (RemoveEntity('crate'), 1),
            'replace': (ReplaceModel('crate', spec.entities[0].source), 2),
            'articulation': (EditArticulation('mobile', spec.agents[0].model), 2),
        }
        change, count = changes[kind]
        assert not session.scene.plan(change).support.supported
        # When explicitly allowing a new episode.
        session.scene.apply(session.scene.plan(change, require_same_episode=False))
        # Then a real scene was rebuilt, old handles fail, and commands are reset.
        assert grouped_native.scenes[0].destroyed == 1
        assert len(grouped_native.scenes[1].entities) == count
        with pytest.raises(ModelError):
            previous.get_qpos()
        np.testing.assert_allclose(session.execution.step().observation.qpos[:, -3:], 0.25)


def test_bad_replacement_is_rejected_before_native_mutation(grouped_native, spec, tmp_path):
    # Given an existing scene and malformed replacement content.
    source = tmp_path / 'bad.urdf'
    source.write_text('<robot')
    with Backend().open(spec) as session:
        # When the edit is planned.
        plan = session.scene.plan(
            ReplaceModel('crate', ModelSource(source, ModelFormat.URDF)),
            require_same_episode=False,
        )
        # Then no native resource was allocated or replaced.
        assert plan.support.status is SupportStatus.UNSUPPORTED
        assert len(grouped_native.scenes) == 1
        assert session.phase is Phase.READY


def test_native_edit_failure_marks_session_failed(grouped_native, spec, monkeypatch):
    # Given a native setter that fails after an edit has begun.
    with Backend().open(spec) as session:

        def fail(*args, **kwargs):
            raise RuntimeError('native failure')

        monkeypatch.setattr(grouped_native.scenes[0].entities[0], 'set_quat', fail)
        # When the native pose edit fails.
        with pytest.raises(RuntimeError):
            session.scene.apply(session.scene.plan(SetPose('crate', Pose((1, 2, 3)))))
        # Then subsequent access cannot pretend the scene was rolled back.
        assert session.phase is Phase.FAILED
        with pytest.raises(ModelError):
            session.state.get_qpos()


def test_format_selection_parses_contents_and_native_receives_parsed_asset(
    grouped_native, tmp_path
):
    # Given an MJCF file whose extension is not used for explicit format selection.
    path = tmp_path / 'model.data'
    path.write_text('<mujoco><worldbody><body><joint name="hinge"/></body></worldbody></mujoco>')
    source = ModelSource(path, ModelFormat.MJCF)
    backend = Backend()
    # When loading and building through the native MJCF route.
    with backend.models.load(backend.models.plan(source)) as imported:
        assert imported.value.root.tag == 'mujoco'
    with backend.open(SceneSpec(entities=(EntitySpec('object', source),))):
        # Then the native builder consumes the parsed asset, not an identity result.
        assert isinstance(grouped_native.morphs_added[0].file, ElementTree.Element)
        assert grouped_native.morphs_added[0].file.tag == 'mujoco'


def test_foreign_agent_and_continuous_execution_rejected(grouped_native):
    # Given unsupported registry and execution requests; when opening a session.
    with pytest.raises(ModelError):
        Backend().open(
            SceneSpec(agents=(AgentSpec('x', RegisteredAgentRef('maniskill', 'Panda')),))
        )
    with pytest.raises(ModelError):
        Backend().open(SceneSpec(), ExecutionSpec(mode='continuous'))
    # Then requests are rejected before a native scene exists.
    assert grouped_native.scenes == []
    assert not Backend().assess(CapabilityRequest('execution.continuous')).supported


def test_reset_and_close_clear_commands_and_invalidate_access(grouped_native, spec):
    # Given a live scene with a velocity command.
    session = Backend().open(spec)
    session.control.set_velocities({'wheel': 3})
    session.execution.step()
    # When resetting explicitly and then closing the session.
    observation, _ = session.execution.reset(seed=4)
    np.testing.assert_allclose(observation.qpos[:, -3:], 0.25)
    assert grouped_native.scenes[0].steps == 1
    session.close()
    session.close()
    # Then native resources are released exactly once and state is inaccessible.
    assert session.phase is Phase.CLOSED
    assert grouped_native.scenes[0].destroyed == 1
    with pytest.raises(ModelError):
        session.state.get_qpos()


def test_attached_rgbd_camera_returns_batched_frames_after_native_pose_edit(grouped_native, spec):
    # Given a declared link-mounted camera in two environments.
    sensor = SensorSpec('eye', 'rgbd', parent='root', width=4, height=3, rate=10)
    scene = replace(spec, agents=(replace(spec.agents[0], sensors=(sensor,)),))
    with Backend().open(scene, ExecutionSpec(n_envs=2, obs_mode='rgbd')) as session:
        session.state.get_observation()
        # When its parent is moved without advancing physics.
        session.scene.apply(session.scene.plan(SetPose('mobile', Pose((4, 0, 0)))))
        observation = session.state.get_observation()
        # Then the camera follows the actual link, and the state schema names its frame.
        assert observation.rgb_image.shape == (2, 3, 4, 3)
        np.testing.assert_array_equal(observation.rgb_image[..., 0], 4)
        np.testing.assert_allclose(observation.depth_image, 2.5)
        assert session.state.schema.sensors[0].frame == 'mobile/eye'
        assert grouped_native.scenes[0].steps == 0


def test_lidar_reads_native_cache_in_declared_sensor_frame(grouped_native, spec):
    # Given a named root-mounted lidar and a native width-major raycast cache.
    sensor = SensorSpec('scan', 'lidar', parent='root', width=3, height=2)
    scene = replace(spec, agents=(replace(spec.agents[0], sensors=(sensor,)),))
    with Backend().open(scene, ExecutionSpec(n_envs=2)) as session:
        # When reading state without physics advancement.
        frame = session.state.get_observation().extra['sensors']['scan']
        # Then native hits are exposed in explicit elevation/azimuth order and local coordinates.
        np.testing.assert_array_equal(frame.distances[0], [[1, 3, 5], [2, 4, 6]])
        assert frame.points.shape == (2, 2, 3, 3)
        assert frame.frame == 'mobile/scan'
        assert grouped_native.scenes[0].sensor_options.return_world_frame is False
        assert grouped_native.scenes[0].steps == 0
