"""Grouped Isaac boundary fixtures; no Kit launch or physics accuracy claim."""

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test import test_isaac_adapter
from usim.capability import CapabilityRequest, Phase, SupportStatus, UnsupportedOperation
from usim.facets import ControlTargets
from usim.model import ImportRequirements, ModelError, ModelFormat, ModelSource
from usim.scene import (
    EditArticulation,
    RemoveEntity,
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
from usim_isaacsim.backend import Backend

legacy_native = test_isaac_adapter.native


@pytest.fixture
def grouped(legacy_native, monkeypatch, tmp_path):
    state = legacy_native
    state.assets = {}
    state.unresolved = []
    state.names = ['wheel', 'unused', 'shoulder', 'wrist']
    physics = SimpleNamespace(
        ArticulationRootAPI='root',
        RigidBodyAPI='body',
        CollisionAPI='collision',
        RevoluteJoint='revolute',
        PrismaticJoint='prismatic',
        FixedJoint='fixed',
    )

    class Prim:
        def __init__(self, stage, path, *, kind='xform', apis=()):
            self.stage, self.path, self.kind, self.apis = stage, path, kind, apis

        def GetPath(self):
            return self.path

        def GetName(self):
            return self.path.rsplit('/', 1)[-1]

        def IsActive(self):
            return True

        def IsA(self, schema):
            return (
                schema == self.kind
                or (schema == 'xformable' and self.kind == 'xform')
                or (schema is physics.Joint and self.kind in {'revolute', 'prismatic', 'fixed'})
            )

        def HasAPI(self, api):
            return api in self.apis

        def GetVariantSet(self, name):
            return SimpleNamespace(
                GetVariantNames=lambda: [],
                SetVariantSelection=lambda value: state.events.append(value),
            )

        def GetReferences(self):
            def add(path, root):
                asset = state.assets[path]
                for old in tuple(asset.prims.values()):
                    if old.path == root or old.path.startswith(root + '/'):
                        target = self.path + old.path[len(root) :]
                        self.stage.prims[target] = Prim(
                            self.stage, target, kind=old.kind, apis=old.apis
                        )
                return True

            return SimpleNamespace(AddReference=add)

    class Stage:
        def __init__(self):
            self.prims = {}

        def GetDefaultPrim(self):
            return self.prims.get('/Asset')

        def GetPrimAtPath(self, path):
            return self.prims.get(path)

        def Load(self):
            state.events.append('load')

        def RemovePrim(self, path):
            self.prims = {
                name: prim
                for name, prim in self.prims.items()
                if name != path and not name.startswith(path + '/')
            }
            state.events.append('remove')

    def define(stage, path):
        prim = Prim(stage, path)
        stage.prims[path] = prim
        return SimpleNamespace(GetPrim=lambda: prim)

    def asset(role='articulation', *, path=None):
        path = path or tmp_path / f'{role}-{len(state.assets)}.usd'
        path.write_text('#usda 1.0\n', encoding='utf-8')
        stage = Stage()
        apis = {'articulation': ('root',), 'rigid_body': ('body',), 'visual': ()}[role]
        stage.prims['/Asset'] = Prim(stage, '/Asset', apis=apis)
        if role == 'articulation':
            stage.prims['/Asset/tool'] = Prim(stage, '/Asset/tool', apis=('body',))
            for name in state.names:
                stage.prims[f'/Asset/{name}'] = Prim(stage, f'/Asset/{name}', kind='revolute')
        state.assets[str(path.resolve())] = stage
        return ModelSource(path, ModelFormat.USD)

    stage = Stage()
    monkeypatch.setattr(
        sys.modules['omni.usd'], 'get_context', lambda: SimpleNamespace(get_stage=lambda: stage)
    )
    physics.Joint = lambda prim: SimpleNamespace(
        GetBody0Rel=lambda: SimpleNamespace(GetTargets=lambda: []),
        GetBody1Rel=lambda: SimpleNamespace(GetTargets=lambda: []),
    )
    pxr = sys.modules['pxr']
    monkeypatch.setattr(pxr, 'UsdPhysics', physics)
    monkeypatch.setattr(
        pxr,
        'Usd',
        SimpleNamespace(
            Stage=SimpleNamespace(Open=lambda path: state.assets.get(path)),
            PrimRange=lambda root: [
                prim
                for path, prim in root.stage.prims.items()
                if path == root.path or path.startswith(root.path + '/')
            ],
        ),
    )
    monkeypatch.setattr(
        pxr,
        'UsdUtils',
        SimpleNamespace(ComputeAllDependencies=lambda path: ([], [], state.unresolved)),
        raising=False,
    )
    monkeypatch.setattr(pxr.UsdGeom, 'Xform', SimpleNamespace(Define=define), raising=False)
    monkeypatch.setattr(pxr.UsdGeom, 'Xformable', 'xformable', raising=False)
    prims = sys.modules['isaacsim.core.prims']
    old_robot, old_body = prims.SingleArticulation, prims.SingleRigidPrim

    class Body(old_body):
        def set_mass(self, mass):
            self.mass = mass

        def set_local_pose(self, *, translation, orientation):
            self.set_world_pose(position=translation, orientation=orientation)

    class Robot(old_robot, Body):
        def __init__(self, prim_path, name):
            old_robot.__init__(self, prim_path, name)
            Body.__init__(self, prim_path, name)
            self.effort_targets = self.qpos.copy()

        def apply_action(self, action):
            super().apply_action(action)
            if hasattr(action, 'joint_efforts'):
                self.effort_targets[action.joint_indices] = action.joint_efforts

    monkeypatch.setattr(prims, 'SingleArticulation', Robot)
    monkeypatch.setattr(prims, 'SingleRigidPrim', Body)
    monkeypatch.setattr(prims, 'SingleXFormPrim', Body, raising=False)
    old_world = sys.modules['isaacsim.core.api'].World

    class World(old_world):
        @property
        def current_time(self):
            return self.steps * self.settings['physics_dt']

        def play(self):
            state.events.append('play')

        def pause(self):
            state.events.append('pause')

        def clear(self):
            state.events.append('clear_scene')

    monkeypatch.setattr(sys.modules['isaacsim.core.api'], 'World', World)
    old_camera = sys.modules['isaacsim.sensors.experimental.rtx'].CameraSensor

    class Camera(old_camera):
        def __init__(self, path, *, resolution, annotators):
            super().__init__(path, resolution=resolution, annotators=annotators)
            self.resolution = resolution
            self.local_pose = None
            self.authoring_object.set_local_poses = self.set_local

        def set_local(self, **kwargs):
            self.local_pose = kwargs

        def get_data(self, name):
            if name == 'rgb':
                return np.full((*self.resolution, 4), 23, dtype=np.uint8), {}
            return np.full(self.resolution, 2.0, dtype=np.float32), {}

    monkeypatch.setattr(sys.modules['isaacsim.sensors.experimental.rtx'], 'CameraSensor', Camera)
    for kind in ('urdf', 'mjcf'):
        module = sys.modules[f'isaacsim.asset.importer.{kind}']

        class Importer:
            def __init__(self, config):
                self.config = config

            def convert(self):
                assert state.events[0] == 'app'
                state.imports.append(self.config)
                path = Path(self.config.usd_path) / 'robot.usd'
                asset(path=path)
                return str(path)

            import_mjcf = import_urdf = convert

        monkeypatch.setattr(module, f'{kind.upper()}Importer', Importer)
    state.asset = asset
    return state


def robot_scene(grouped):
    source = grouped.asset()
    return SceneSpec(
        agents=(
            AgentSpec(
                'mobile_arm',
                source,
                ControlSpec(
                    joint_names=('shoulder', 'wheel', 'wrist'),
                    position_joints=('shoulder',),
                    velocity_joints=('wheel',),
                    effort_joints=('wrist',),
                    initial_qpos=(0.1, 0.0, 0.0),
                ),
            ),
        ),
        entities=(
            EntitySpec(
                'box',
                grouped.asset('rigid_body'),
                requirements=ImportRequirements(role='rigid_body'),
            ),
        ),
    )


def test_named_facets_apply_without_step_and_report_native_order(grouped):
    with Backend().open(robot_scene(grouped)) as session:
        session.control.apply(
            ControlTargets(
                positions={'shoulder': 0.4}, velocities={'wheel': [2.0]}, efforts={'wrist': 0.7}
            )
        )
        robot = grouped.robots[0]
        assert grouped.worlds[0].steps == 0
        np.testing.assert_allclose(robot.targets[[2, 0, 3]], [0.4, 0.0, 0.0])
        np.testing.assert_allclose(robot.velocity_targets, [2.0, 0, 0, 0])
        np.testing.assert_allclose(robot.effort_targets, [0, 0, 0, 0.7])
        assert [joint.name for joint in session.state.schema.joints] == [
            'shoulder',
            'wheel',
            'wrist',
        ]
        assert not session.state.schema.batched
        np.testing.assert_allclose(session.state.get_qpos(), [0.1, 0, 0])
        assert len(grouped.robots) == 1


def test_episode_step_advances_once_and_observation_is_owned(grouped):
    with Backend().open(robot_scene(grouped), ExecutionSpec(max_episode_steps=1)) as session:
        session.control.set_positions({'shoulder': 0.5})
        result = session.execution.step()
        assert grouped.worlds[0].steps == 1 and result.truncated
        assert session.phase is Phase.PAUSED
        result.observation.qpos[:] = 999
        np.testing.assert_allclose(session.state.get_qpos(), [0.5, 0, 0])


@pytest.mark.parametrize(
    'targets',
    [
        ControlTargets(positions={'shoulder': 0.4}, velocities={'wheel': [1.0, 2.0]}),
        ControlTargets(positions={'shoulder': 0.4}, efforts={'wheel': 2.0}),
        ControlTargets(positions={'shoulder': 4.0}),
    ],
)
def test_invalid_control_is_rejected_before_any_write(grouped, targets):
    with Backend().open(robot_scene(grouped)) as session:
        before = grouped.robots[0].targets.copy()
        with pytest.raises(ModelError):
            session.control.apply(targets)
        np.testing.assert_array_equal(grouped.robots[0].targets, before)
        assert grouped.worlds[0].steps == 0


def test_pose_mass_joint_state_changes_are_native_and_do_not_step(grouped):
    with Backend().open(robot_scene(grouped)) as session:
        handle = session.scene.handle('box')
        for change in (
            SetPose('box', Pose((1.0, 2.0, 3.0))),
            SetParameter('box', 'mass', 2.5),
            SetJointState('mobile_arm', positions=(('shoulder', 0.6),)),
        ):
            session.scene.apply(session.scene.plan(change))
        assert session.scene.resolve(handle) == 'box'
        assert session.state.for_entity('box').get_pose().position == (1.0, 2.0, 3.0)
        assert session._native.entities['box'].native.mass == 2.5
        np.testing.assert_allclose(session.state.get_qpos(), [0.6, 0, 0])
        assert grouped.worlds[0].steps == 0


def test_topology_requires_permission_before_mutation(grouped):
    with Backend().open(robot_scene(grouped)) as session:
        before = list(grouped.events)
        plan = session.scene.plan(RemoveEntity('box'))
        with pytest.raises(UnsupportedOperation):
            session.scene.apply(plan)
        assert grouped.events == before
        assert session.scene.revision == 0 and len(grouped.worlds) == 1


def test_permitted_spawn_rebuilds_world_and_invalidates_bound_views(grouped):
    with Backend().open(robot_scene(grouped)) as session:
        old = session.state.for_entity('mobile_arm')
        change = SpawnEntity(
            EntitySpec(
                'marker',
                grouped.asset('visual'),
                requirements=ImportRequirements(role='visual'),
            )
        )
        session.scene.apply(session.scene.plan(change, require_same_episode=False))
        assert len(grouped.worlds) == 2 and grouped.worlds[0].instance_cleared
        assert session.scene.revision == 1
        with pytest.raises(ModelError):
            old.get_qpos()
        assert session.scene.resolve(session.scene.handle('marker')) == 'marker'
        np.testing.assert_allclose(session.state.get_qpos(), [0.1, 0, 0])


def test_topology_replacement_rebinds_real_articulation(grouped):
    with Backend().open(robot_scene(grouped)) as session:
        original = session._native.entities['mobile_arm'].native
        source = grouped.asset()
        plan = session.scene.plan(
            EditArticulation('mobile_arm', source), require_same_episode=False
        )
        session.scene.apply(plan)
        assert session._native.entities['mobile_arm'].native is not original
        assert session.agents.describe('mobile_arm').model == source
        assert len(grouped.worlds) == 2


def test_model_first_articulation_binds_without_agent_registration(grouped):
    spec = SceneSpec(entities=(EntitySpec('robot', grouped.asset()),))
    with Backend().open(spec) as session:
        session.control.set_positions({'shoulder': 0.4})
        result = session.execution.step()
        assert session.agents.names == ()
        assert result.observation.qpos[2] == 0.4
        assert len(grouped.robots) == 1


def test_registered_agents_require_external_model_source_before_starting_app(grouped):
    spec = SceneSpec(
        agents=(
            AgentSpec(
                'arm',
                RegisteredAgentRef('isaacsim', 'external'),
                requirements=ImportRequirements(preserve=frozenset({'joint_structure'})),
            ),
        )
    )
    with pytest.raises(ModelError, match='ModelSource'):
        Backend().open(spec)
    assert not grouped.apps
    assert (
        Backend().assess(CapabilityRequest('agents.registered')).status
        is SupportStatus.UNSUPPORTED
    )


def test_joint_controls_preserve_native_values_when_not_injected(grouped):
    with Backend().open(robot_scene(grouped)):
        robot = grouped.robots[0]
        np.testing.assert_allclose(robot.kp, [0, 17, 17, 0])
        np.testing.assert_allclose(robot.kd, [2, 2, 2, 0])
        np.testing.assert_allclose(robot.efforts, 8)


def test_injected_joint_controls_follow_selected_order_and_modes(grouped):
    spec = robot_scene(grouped)
    agent = spec.agents[0]
    control = replace(
        agent.control,
        joint_stiffness=(21.0, 32.0, 43.0),
        joint_damping=(4.0, 5.0, 6.0),
        joint_force_limits=(7.0, 8.0, 9.0),
    )
    with Backend().open(replace(spec, agents=(replace(agent, control=control),))):
        robot = grouped.robots[0]
        np.testing.assert_allclose(robot.kp, [0, 17, 21, 0])
        np.testing.assert_allclose(robot.kd, [5, 2, 4, 0])
        np.testing.assert_allclose(robot.efforts, [8, 8, 7, 9])


def test_permitted_removal_rebuilds_and_removes_native_entity(grouped):
    with Backend().open(robot_scene(grouped)) as session:
        session.scene.apply(session.scene.plan(RemoveEntity('box'), require_same_episode=False))
        with pytest.raises(ModelError):
            session.scene.handle('box')
        assert tuple(session._native.entities) == ('mobile_arm',)
        assert len(grouped.worlds) == 2


def test_permitted_agent_removal_rebinds_agent_and_state_groups(grouped):
    with Backend().open(robot_scene(grouped)) as session:
        session.scene.apply(
            session.scene.plan(RemoveEntity('mobile_arm'), require_same_episode=False)
        )
        assert session.agents.names == ()
        assert session.state.get_qpos().size == 0
        assert tuple(session._native.entities) == ('box',)


def test_rebuild_failure_marks_session_failed_instead_of_claiming_rollback(grouped, monkeypatch):
    session = Backend().open(robot_scene(grouped))
    bound = session.state.for_entity('mobile_arm')

    def fail_build():
        raise RuntimeError('fixture rebuild failure')

    monkeypatch.setattr(session, '_build', fail_build)
    with pytest.raises(RuntimeError):
        session.scene.apply(session.scene.plan(RemoveEntity('box'), require_same_episode=False))
    assert session.phase is Phase.FAILED
    with pytest.raises(ModelError):
        bound.get_qpos()
    session.close()
    assert grouped.apps[0].closed


def test_native_loader_rejects_role_selector_and_unresolved_dependencies(grouped):
    source = grouped.asset()
    with Backend().imports() as runtime:
        for selected, requirements in (
            (source, ImportRequirements(role='visual')),
            (
                ModelSource(source.path, ModelFormat.USD, '/missing'),
                ImportRequirements(),
            ),
        ):
            with pytest.raises(ModelError):
                runtime.models.plan(selected, requirements)
        grouped.unresolved.append('missing.usd')
        with pytest.raises(ModelError):
            runtime.models.plan(source)


def test_native_loader_selects_requested_subtree_and_retains_source(grouped):
    source = grouped.asset()
    selected = ModelSource(source.path, source.format, '/Asset/tool')
    with Backend().imports() as runtime:
        plan = runtime.models.plan(selected, ImportRequirements(role='rigid_body'))
        with runtime.models.load(plan) as loaded:
            assert loaded.value.root == '/Asset/tool'
            assert loaded.value.rigid_body == '/Asset/tool'
            assert loaded.value.articulation is None
    assert source.path.is_file()


def test_xml_transport_is_owned_and_does_not_claim_mesh_fidelity(grouped, tmp_path):
    path = tmp_path / 'robot.urdf'
    path.write_text(
        '<robot name="r"><link name="base"><collision><geometry>'
        '<mesh filename="part.stl"/></geometry></collision></link></robot>',
        encoding='utf-8',
    )
    (tmp_path / 'part.stl').write_text('fixture', encoding='utf-8')
    source = ModelSource.from_path(path)
    backend = Backend()
    with pytest.raises(ModelError):
        backend.models.plan(source)
    requirements = ImportRequirements(preserve=frozenset({'joint_structure'}), base='floating')
    plan = backend.models.plan(source, requirements)
    assert plan.report.changed >= {'collision', 'base'}
    assert not grouped.apps
    with backend.imports() as runtime:
        with runtime.models.load(runtime.models.plan(source, requirements)) as imported:
            output = imported.value.path
            assert output.is_file()
            assert grouped.imports[0].fix_base is False
        assert not output.exists()
    assert path.exists() and grouped.apps[0].closed


def test_urdf_assessment_requires_valid_authored_physics(tmp_path):
    path = tmp_path / 'incomplete.urdf'
    path.write_text(
        '<robot name="r"><link name="base"><inertial><inertia/></inertial>'
        '<collision><geometry><box/></geometry></collision></link></robot>',
        encoding='utf-8',
    )
    backend = Backend()
    with pytest.raises(ModelError):
        backend.models.plan(ModelSource.from_path(path))
    plan = backend.models.plan(
        ModelSource.from_path(path), ImportRequirements(preserve=frozenset())
    )
    assert plan.report.unknown >= {'inertia', 'collision'}


def test_continuous_has_no_episode_driver_and_stops_native_loop(grouped, monkeypatch):
    with Backend().open(robot_scene(grouped), ExecutionSpec(mode='continuous')) as session:
        world = grouped.worlds[0]
        original = world.step

        def step(*, render):
            original(render=render)
            session.execution.stop()

        monkeypatch.setattr(world, 'step', step)
        with pytest.raises(UnsupportedOperation):
            session.execution.step()
        session.execution.run()
        assert world.steps == 1 and session.phase is Phase.PAUSED


def test_close_invalidates_views_and_releases_native_before_app(grouped):
    session = Backend().open(robot_scene(grouped))
    state = session.state.for_entity('mobile_arm')
    session.close()
    session.close()
    assert grouped.apps[0].closed and grouped.worlds[0].instance_cleared
    assert session.phase is Phase.CLOSED
    with pytest.raises(ModelError):
        state.get_qpos()


def test_second_runtime_rejects_before_touching_first_world(grouped):
    with Backend().open(robot_scene(grouped)) as session:
        with pytest.raises(ModelError):
            Backend().open(SceneSpec())
        assert len(grouped.apps) == 1
        assert not grouped.worlds[0].instance_cleared
        np.testing.assert_allclose(session.state.get_qpos(), [0.1, 0, 0])


def test_attached_rgbd_sensor_returns_named_frames_without_advancing(grouped):
    spec = robot_scene(grouped)
    agent = replace(
        spec.agents[0],
        sensors=(
            SensorSpec(
                'hand',
                'rgbd',
                parent='tool',
                pose=Pose((0.1, 0.2, 0.3)),
                width=4,
                height=3,
            ),
        ),
    )
    with Backend().open(replace(spec, agents=(agent,)), ExecutionSpec(obs_mode='rgbd')) as session:
        observation = session.state.get_observation()
        assert observation.rgb_image.shape == (3, 4, 3)
        assert observation.depth_image.shape == (3, 4)
        assert set(observation.extra['sensors']) == {'hand/rgb', 'hand/depth'}
        assert [(layout.name, layout.shape) for layout in session.state.schema.sensors] == [
            ('hand/rgb', (3, 4, 3)),
            ('hand/depth', (3, 4)),
        ]
        assert grouped.cameras[0].path.endswith('/tool/UsimSensor_0')
        np.testing.assert_allclose(
            grouped.cameras[0].local_pose['translations'], [[0.1, 0.2, 0.3]]
        )
        assert grouped.worlds[0].steps == 0 and grouped.events.count('capture') == 1
        observation.rgb_image[:] = 0
        assert np.all(session.state.get_observation().rgb_image == 23)
        assert grouped.events.count('capture') == 1


def test_lidar_uses_live_raycast_and_excludes_robot_collisions(grouped, monkeypatch):
    from types import ModuleType

    spec = robot_scene(grouped)
    agent = replace(
        spec.agents[0],
        sensors=(SensorSpec('scan', 'lidar', width=4, height=1),),
    )
    rays = []

    def raycast(origin, direction, maximum, report):
        rays.append((origin, direction, maximum))
        report(
            SimpleNamespace(
                collision=session._native.entities['mobile_arm'].prim_path + '/self',
                rigid_body='',
                distance=0.1,
            )
        )
        report(SimpleNamespace(collision='/World/Obstacle', rigid_body='', distance=2.0))

    module = ModuleType('omni.physx')
    module.get_physx_scene_query_interface = lambda: SimpleNamespace(raycast_all=raycast)
    monkeypatch.setitem(sys.modules, 'omni.physx', module)
    monkeypatch.setattr(sys.modules['carb'], 'Float3', lambda *value: value, raising=False)
    with Backend().open(replace(spec, agents=(agent,))) as session:
        observed = session.state.get_observation()
        np.testing.assert_allclose(observed.extra['sensors']['scan/lidar'], [2.0] * 4)
        assert len(rays) == 4 and grouped.worlds[0].steps == 0
        assert session.state.schema.sensors[0].shape == (4,)


def test_invalid_sensor_parent_releases_native_resources(grouped):
    spec = robot_scene(grouped)
    agent = replace(spec.agents[0], sensors=(SensorSpec('camera', 'rgb', parent='missing'),))
    with pytest.raises(ModelError):
        Backend().open(replace(spec, agents=(agent,)))
    assert grouped.apps[0].closed and grouped.worlds[0].instance_cleared


def test_request_specific_capabilities_do_not_overpromise():
    backend = Backend()
    assert backend.assess(CapabilityRequest('unknown')).status is SupportStatus.UNSUPPORTED
    assert (
        backend.assess(CapabilityRequest('scene.set_pose', Phase.RUNNING)).status
        is SupportStatus.CONDITIONAL
    )
    assert (
        backend.assess(CapabilityRequest('state.read', n_envs=2)).status
        is SupportStatus.UNSUPPORTED
    )
