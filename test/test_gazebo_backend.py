"""Grouped Gazebo tests at the XML, wire and ROS-service boundaries; no native runtime."""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
import base64
import io
import json
from pathlib import Path
import queue
import threading
from types import SimpleNamespace
from urllib.parse import unquote
from urllib.request import url2pathname
import xml.etree.ElementTree as ET

import numpy as np
import pytest

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
    SceneSpec,
    SensorSpec,
)
from usim_gazebo.backend import Backend
from usim_gazebo.transport import WireClient
from usim_gazebo.worker import NativeWorker, SERVICE_TYPES, serve
from usim_gazebo.sensors import decode, SensorChannel
from usim.facets import SensorLayout


@pytest.fixture
def model(tmp_path: Path) -> ModelSource:
    path = tmp_path / 'robot.urdf'
    path.write_text(
        '<robot name="mobile_arm">'
        '<link name="base"/><link name="wheel"/><link name="arm"/>'
        '<joint name="wheel" type="continuous"><parent link="base"/><child link="wheel"/>'
        '<limit effort="10" velocity="3"/></joint>'
        '<joint name="shoulder" type="revolute"><parent link="base"/><child link="arm"/>'
        '<limit effort="20" lower="-1" upper="1" velocity="2"/></joint></robot>',
        encoding='utf-8',
    )
    return ModelSource.from_path(path)


class ServiceWorld:
    """Service-boundary fake whose spawn/delete/state mutations obey real preconditions."""

    def __init__(self, *, missing=()) -> None:
        self.services = set(SERVICE_TYPES) - set(missing)
        self.entities = {}
        self.calls = []
        self.closed = False
        self.paused = True
        self.fail_service = None
        self.resumed = threading.Event()
        self.ended = Future()
        self.channels = {}
        self.frames = {}

    def request(self, operation, payload):
        if self.closed:
            raise ModelError('Fake worker is closed')
        if operation == 'initialize':
            assert ET.fromstring(payload['world']).find('world') is not None
            return {'services': sorted(self.services)}
        if operation == 'subscribe_sensors':
            self.channels.update({item['topic']: item for item in payload['channels']})
            return {}
        if operation == 'read_sensor':
            topic = payload['topic']
            assert topic in self.channels
            return self.frames[topic]
        assert operation == 'service'
        service, fields = payload['service'], payload['fields']
        self.calls.append((service, fields))
        if service not in self.services or service == self.fail_service:
            raise ModelError(service)
        match service:
            case '/spawn_entity':
                assert fields['name'] not in self.entities
                self.entities[fields['name']] = {
                    'xml': fields['xml'],
                    'pose': fields['initial_pose'],
                    'positions': {'wheel': 0.0, 'shoulder': 0.0},
                    'velocities': {'wheel': 0.0, 'shoulder': 0.0},
                }
                return {}
            case '/delete_entity':
                self.entities.pop(fields['name'])
                return {}
            case '/get_entity_state':
                return {'state': {'pose': self.entities[fields['name']]['pose']}}
            case '/set_entity_state':
                self.entities[fields['state']['name']]['pose'] = fields['state']['pose']
                return {}
            case '/get_joint_properties':
                entity, _, joint = fields['joint_name'].partition('::')
                return {
                    'position': [self.entities[entity]['positions'][joint]],
                    'rate': [self.entities[entity]['velocities'][joint]],
                }
            case '/set_model_configuration':
                self.entities[fields['model_name']]['positions'].update(
                    zip(fields['joint_names'], fields['joint_positions'])
                )
                return {}
            case '/set_joint_properties':
                entity, _, joint = fields['joint_name'].partition('::')
                self.entities[entity]['velocities'][joint] = fields['ode_joint_config']['vel'][0]
                return {}
            case '/get_link_properties':
                return {
                    'com': {'position': {'x': 0.1, 'y': 0.2, 'z': 0.3}},
                    'gravity_mode': True,
                    'mass': 2.0,
                    'ixx': 1.0,
                    'iyy': 2.0,
                    'izz': 3.0,
                    'ixy': 0.1,
                    'ixz': 0.2,
                    'iyz': 0.3,
                }
            case '/set_link_properties' | '/clear_joint_forces' | '/apply_joint_effort':
                return {}
            case '/pause_physics':
                self.paused = True
                return {}
            case '/unpause_physics':
                self.paused = False
                self.resumed.set()
                return {}
            case _:
                raise AssertionError(service)

    def close(self) -> None:
        self.closed = True
        if not self.ended.done():
            self.ended.set_result(None)


@pytest.fixture
def world() -> ServiceWorld:
    return ServiceWorld()


def test_direct_load_parses_joint_graph_without_agent_registration(model: ModelSource) -> None:
    # Given an explicit robot file and a dependency-free provider.
    models = Backend().models
    plan = models.plan(model)
    # When loading the parsed asset without opening a native world.
    with models.load(plan) as loaded:
        # Then both mobile and manipulator joints belong to one articulation.
        assert [joint.name for joint in loaded.value.joints] == ['wheel', 'shoulder']
        assert loaded.value.links == ('base', 'wheel', 'arm')
        assert 'joint_structure' in loaded.report.preserved


def test_mesh_and_adjacent_material_resources_are_owned(tmp_path: Path) -> None:
    # Given a relative mesh with an adjacent texture.
    assets = tmp_path / 'meshes'
    assets.mkdir()
    (assets / 'mesh.obj').write_text('mtllib material.mtl')
    (assets / 'material.mtl').write_text('map_Kd texture.png')
    (assets / 'texture.png').write_bytes(b'texture')
    robot = tmp_path / 'asset.urdf'
    robot.write_text(
        '<robot name="r"><link name="base"><visual><geometry>'
        '<mesh filename="meshes/mesh.obj"/></geometry></visual></link></robot>'
    )
    models = Backend().models
    # When loading into an owned import directory.
    with models.load(models.plan(ModelSource.from_path(robot))) as loaded:
        uri = ET.fromstring(loaded.value.xml).find('.//mesh').get('filename')
        path = Path(url2pathname(unquote(uri.removeprefix('file://'))))
        # Then material dependencies coexist until the owner closes.
        assert path.read_text() == 'mtllib material.mtl'
        assert (path.parent / 'texture.png').read_bytes() == b'texture'
    assert not path.exists()
    assert (assets / 'texture.png').is_file()


@pytest.mark.parametrize('format', [ModelFormat.MJCF, ModelFormat.USD])
def test_no_uninstalled_converter_is_claimed(model: ModelSource, format: ModelFormat) -> None:
    # Given a format with no supplied conversion edge.
    source = replace(model, format=format)
    # When assessing a route.
    report = Backend().models.assess(source)
    # Then native URDF/SDF support cannot masquerade as universal conversion.
    assert report.status is SupportStatus.UNSUPPORTED


def test_plugin_loss_requires_permission_and_removes_ungated_driver(model: ModelSource) -> None:
    # Given a caller-authored control plugin.
    model.path.write_text(
        model.path.read_text().replace(
            '</robot>', '<gazebo><plugin name="drive" filename="drive.so"/></gazebo></robot>'
        )
    )
    models = Backend().models
    assert models.assess(model).status is SupportStatus.UNSUPPORTED
    # When explicitly allowing plugin removal.
    requirements = ImportRequirements(allow_loss=frozenset({'plugins'}))
    with models.load(models.plan(model, requirements)) as loaded:
        # Then the model cannot bypass the grouped motor gate through that plugin.
        assert loaded.report.lost == frozenset({'plugins'})
        assert ET.fromstring(loaded.value.xml).find('.//plugin') is None


def test_sdf_selector_loads_exactly_one_model(tmp_path: Path) -> None:
    # Given two independently named SDF models.
    path = tmp_path / 'scene.sdf'
    path.write_text(
        '<sdf version="1.6"><model name="a"><link name="x"/></model>'
        '<model name="b"><link name="y"/></model></sdf>'
    )
    models = Backend().models
    assert not models.assess(ModelSource.from_path(path)).supported
    # When selecting one model.
    with models.load(models.plan(ModelSource.from_path(path, selector='b'))) as loaded:
        # Then the selection is parsed and represented in the returned native XML.
        assert loaded.value.links == ('y',)
        assert ET.fromstring(loaded.value.xml).find('model').get('name') == 'b'


def test_multi_object_scene_spawns_and_closes_owned_resources(model, world) -> None:
    # Given an independent prop and one combined mobile arm robot.
    spec = SceneSpec(
        entities=(EntitySpec('prop', model, Pose((1.0, 2.0, 3.0))),),
        agents=(AgentSpec('robot', model, ControlSpec(velocity_joints=('wheel',))),),
    )
    backend = Backend(transport_factory=lambda _: world)
    # When opening the actual grouped lifecycle through a service fake.
    with backend.open(spec) as session:
        # Then models are independently spawned in one paused native world.
        assert set(world.entities) == {'prop', 'robot'}
        assert session.phase is Phase.READY
        assert session.agents.describe('robot').model == model
        assert session.state['robot'].schema.batched is False
        assert len(session.state['robot'].schema.joints) == 2
        directory = session.directory
    assert world.closed and not directory.exists()
    assert session.phase is Phase.CLOSED


def test_rgbd_and_lidar_attach_to_same_robot_and_return_native_frames(model, world) -> None:
    # Given independent RGBD and scan specifications on the same mobile arm articulation.
    sensors = (
        SensorSpec('camera', 'rgbd', parent='arm', width=2, height=1),
        SensorSpec('scan', 'lidar', parent='base', width=2),
    )
    with Backend(transport_factory=lambda _: world).open(
        SceneSpec(agents=(AgentSpec('robot', model, sensors=sensors),))
    ) as session:
        root = ET.fromstring(world.entities['robot']['xml'])
        assert len(root.findall('.//sensor')) == 2
        assert root.find("gazebo[@reference='arm']/sensor/camera/image/width").text == '2'
        for channel in session._sensor_channels['robot']:
            if channel.layout.kind == 'lidar':
                world.frames[channel.topic] = {'ranges': [1.5, None], 'stamp': 1.0}
            else:
                depth = channel.layout.kind == 'depth'
                raw = np.asarray([0.5, 2.0], dtype='<f4').tobytes() if depth else bytes(range(6))
                world.frames[channel.topic] = {
                    'height': 1,
                    'width': 2,
                    'step': 8 if depth else 6,
                    'encoding': '32FC1' if depth else 'rgb8',
                    'is_bigendian': False,
                    'data': base64.b64encode(raw).decode(),
                    'stamp': 1.0,
                }
        # When reading actual wire image and scan payloads.
        observation = session.state.get_observation()
        # Then images retain native shape/units, scan no-returns stay infinite, and one robot exists.
        assert observation.rgb_image.shape == (1, 2, 3)
        np.testing.assert_allclose(observation.depth_image, [[0.5, 2.0]])
        np.testing.assert_allclose(observation.extra['sensors']['scan'], [1.5, np.inf])
        assert [item.kind for item in session.state.schema.sensors] == ['rgb', 'depth', 'lidar']
        scan = session.state.schema.sensors[-1]
        assert root.find("gazebo[@reference='base']/sensor/plugin/frame_name").text == scan.frame
        assert session.assess(CapabilityRequest('sensors.rgbd', session.phase)).supported
        assert set(world.entities) == {'robot'}


def test_depth_decoding_preserves_padding_endianness_and_millimeter_units() -> None:
    # Given a big-endian uint16 depth frame with row padding.
    channel = SensorChannel(
        SensorLayout('d', 'depth', (1, 2), 'float32', 'm', 'link'), '/d', 'Image'
    )
    response = {
        'height': 1,
        'width': 2,
        'step': 6,
        'encoding': '16UC1',
        'is_bigendian': True,
        'data': base64.b64encode(b'\x03\xe8\x07\xd0\x00\x00').decode(),
    }
    # When decoding the native bytes.
    frame = decode(channel, response)
    # Then padding is discarded and millimeters become meters.
    np.testing.assert_allclose(frame, [[1.0, 2.0]])


def test_sensor_parent_is_validated_before_native_launch(model) -> None:
    # Given a sensor referring to a nonexistent link.
    backend = Backend(transport_factory=lambda _: pytest.fail('must not launch'))
    # When opening the scene.
    with pytest.raises(ModelError):
        backend.open(
            SceneSpec(
                agents=(
                    AgentSpec(
                        'robot', model, sensors=(SensorSpec('camera', 'rgb', parent='absent'),)
                    ),
                )
            )
        )


def test_observation_mode_requires_actual_sensor_before_native_launch(model) -> None:
    # Given an RGB observation request without an attached camera.
    backend = Backend(transport_factory=lambda _: pytest.fail('must not launch'))
    # When opening the scene.
    with pytest.raises(UnsupportedOperation):
        backend.open(
            SceneSpec(entities=(EntitySpec('robot', model),)),
            ExecutionSpec(mode='continuous', obs_mode='rgb'),
        )


def test_native_sensor_request_completes_only_from_subscription_callback(monkeypatch, tmp_path):
    # Given a subscribed image topic and a fake executor that delivers a native Image.
    import sys

    monkeypatch.setitem(sys.modules, 'rclpy.task', SimpleNamespace(Future=Future))
    message = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=0), frame_id='camera'),
        data=b'\x01\x02\x03',
        height=1,
        width=1,
        step=3,
        encoding='rgb8',
        is_bigendian=False,
    )
    ros = SimpleNamespace()
    worker = NativeWorker(SimpleNamespace(), ros, tmp_path)
    worker.subscriptions['/camera'] = True
    ros.spin_until_future_complete = lambda node, future, timeout_sec: worker._sensor_message(
        '/camera', message
    )
    # When the worker awaits a frame.
    result = worker.dispatch('read_sensor', {'topic': '/camera'}, 3)
    # Then the acknowledgement contains the delivered bytes and simulator timestamp.
    assert base64.b64decode(result['data']) == message.data
    assert result['stamp'] == 1.0
    assert worker.sensor_waiters == {}


def test_pose_joint_state_and_mass_edits_reach_services(model, world) -> None:
    # Given an available configuration service and a paused model.
    with Backend(transport_factory=lambda _: world).open(
        SceneSpec(entities=(EntitySpec('robot', model),))
    ) as session:
        # When applying explicit pose, position, and mass changes.
        for change in (
            SetPose('robot', Pose((1.0, 0.0, 0.5))),
            SetJointState('robot', (('shoulder', 0.4),)),
            SetParameter('robot', 'base.mass', 3.0),
        ):
            session.scene.apply(session.scene.plan(change))
        # Then state comes from native responses and non-mass inertia is retained.
        assert session.state['robot'].get_pose().position == (1.0, 0.0, 0.5)
        np.testing.assert_allclose(session.state.get_qpos(), [0.0, 0.4])
        written = next(
            fields for service, fields in world.calls if service == '/set_link_properties'
        )
        assert written['mass'] == 3.0 and written['ixx'] == 1.0
        assert written['com']['position']['x'] == 0.1


def test_missing_joint_service_is_unsupported_before_edit(model) -> None:
    # Given stock ROS support without a model-configuration service.
    world = ServiceWorld(missing=('/set_model_configuration',))
    with Backend(transport_factory=lambda _: world).open(
        SceneSpec(entities=(EntitySpec('robot', model),))
    ) as session:
        previous = len(world.calls)
        # When planning and trying a joint-position edit.
        plan = session.scene.plan(SetJointState('robot', (('wheel', 1.0),)))
        with pytest.raises(UnsupportedOperation):
            session.scene.apply(plan)
        # Then unsupported semantics cause no native operation or failure phase.
        assert plan.support.status is SupportStatus.UNSUPPORTED
        assert len(world.calls) == previous
        assert session.phase is Phase.READY


def test_control_gate_validates_all_modes_before_any_native_write(model, world) -> None:
    # Given one articulation with a disabled motor.
    with Backend(transport_factory=lambda _: world).open(
        SceneSpec(entities=(EntitySpec('robot', model),))
    ) as session:
        with pytest.raises(ModelError):
            session.control.set_velocities({'wheel': 1.0})
        session.control.set_motor(True)
        previous = len(world.calls)
        # When a mixed command includes unsupported position control.
        with pytest.raises(UnsupportedOperation):
            session.control.apply(
                ControlTargets(
                    velocities={'wheel': 1.0},
                    positions={'shoulder': 0.2},
                )
            )
        # Then even the supported half is not written.
        assert len(world.calls) == previous


def test_velocity_control_does_not_step_and_disable_clears_native_motor(model, world) -> None:
    # Given an enabled controller in a paused world.
    with Backend(transport_factory=lambda _: world).open(
        SceneSpec(entities=(EntitySpec('robot', model),))
    ) as session:
        session.control.set_motor(True)
        session.control.set_velocities({'wheel': 1.0})
        assert world.entities['robot']['velocities']['wheel'] == 1.0
        # When motor disable is acknowledged.
        session.control.set_motor(False)
        # Then the native motor is off without advancing simulation.
        assert world.entities['robot']['velocities']['wheel'] == 0.0
        assert world.paused
        assert world.calls[-1][1]['ode_joint_config']['fmax'] == [0.0]
        with pytest.raises(UnsupportedOperation):
            session.execution.step()


def test_topology_replacement_rebuilds_and_invalidates_handles(model, world) -> None:
    # Given a bound agent and old state/scene handles.
    with Backend(transport_factory=lambda _: world).open(
        SceneSpec(agents=(AgentSpec('robot', model),))
    ) as session:
        old = session.scene.handle('robot')
        reader = session.state['robot']
        change = EditArticulation('robot', model)
        assert not session.scene.plan(change).support.supported
        # When explicit state-reset permission allows delete and native respawn.
        plan = session.scene.plan(change, require_same_episode=False)
        assert 'entity_state_reset' in plan.support.effects
        session.scene.apply(plan)
        # Then the real service sequence rebuilds and old bindings cannot survive.
        assert [service for service, _ in world.calls][-2:] == ['/delete_entity', '/spawn_entity']
        assert session.agents.names == ()
        with pytest.raises(ModelError):
            session.scene.resolve(old)
        with pytest.raises(ModelError):
            reader.get_qpos()


def test_spawn_remove_invalidates_old_controls_but_new_controls_work(model, world) -> None:
    # Given a controller holding a topology-scoped handle.
    with Backend(transport_factory=lambda _: world).open(
        SceneSpec(entities=(EntitySpec('robot', model),))
    ) as session:
        control = session.control['robot']
        control.set_motor(True)
        # When topology changes through spawn then removal.
        session.scene.apply(session.scene.plan(SpawnEntity(EntitySpec('prop', model))))
        session.scene.apply(session.scene.plan(RemoveEntity('prop')))
        # Then stale controller references fail but fresh ones are usable.
        with pytest.raises(ModelError):
            control.set_motor(True)
        session.control['robot'].set_motor(True)
        session.control['robot'].set_velocities({'wheel': 0.5})
        assert set(world.entities) == {'robot'}


def test_service_failure_invalidates_phase_and_releases_resources(model, world) -> None:
    # Given a live world that rejects a pose change.
    session = Backend(transport_factory=lambda _: world).open(
        SceneSpec(entities=(EntitySpec('robot', model),))
    )
    old = session.scene.handle('robot')
    world.fail_service = '/set_entity_state'
    # When the native error propagates through SceneEditor.
    with pytest.raises(ModelError):
        session.scene.apply(session.scene.plan(SetPose('robot', Pose())))
    # Then no success is fabricated and ownership is released.
    assert session.phase is Phase.FAILED
    assert world.closed and not session.directory.exists()
    with pytest.raises(ModelError):
        session.scene.resolve(old)
    session.close()


def test_running_edits_require_pause_and_continuous_run_cancels(model, world, monkeypatch) -> None:
    # Given a run subscription registered before resume.
    with (
        Backend(transport_factory=lambda _: world).open(
            SceneSpec(entities=(EntitySpec('robot', model),))
        ) as session,
        ThreadPoolExecutor(max_workers=1) as pool,
    ):
        resumed = threading.Event()
        native_resume = session.execution.resume

        def resume():
            native_resume()
            resumed.set()

        monkeypatch.setattr(session.execution, 'resume', resume)
        running = pool.submit(session.execution.run)
        assert resumed.wait(timeout=2)
        # When a running edit is requested and the continuous run is stopped.
        plan = session.scene.plan(SetPose('robot', Pose()))
        assert plan.support.required_phase is Phase.PAUSED
        session.execution.stop()
        running.result(timeout=2)
        # Then pause is acknowledged, without an invented episode boundary.
        assert session.phase is Phase.PAUSED
        assert world.paused


def test_worker_disconnect_fails_run_and_releases_resources(model, world, monkeypatch) -> None:
    # Given a continuous run waiting on its subscribed termination event.
    session = Backend(transport_factory=lambda _: world).open(
        SceneSpec(entities=(EntitySpec('robot', model),))
    )
    resumed = threading.Event()
    native_resume = session.execution.resume

    def resume():
        native_resume()
        resumed.set()

    monkeypatch.setattr(session.execution, 'resume', resume)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(session.execution.run)
        assert resumed.wait(timeout=2)
        # When the worker's channel closes unexpectedly.
        world.ended.set_result(None)
        # Then a run cannot wait forever or report success after the native process dies.
        with pytest.raises(ModelError):
            pending.result(timeout=2)
    assert session.phase is Phase.FAILED
    assert world.closed and not session.directory.exists()
    session.close()


class PipeProcess:
    """Blocking pipe-shaped fake: explicit writes wake the worker, EOF wakes the reader."""

    def __init__(self, reply=True) -> None:
        self.lines = queue.Queue()
        self.requested = threading.Event()
        self.reply = reply
        self.stdin = self
        self.stdout = self
        self.returncode = None

    def write(self, line):
        request = json.loads(line)
        self.requested.set()
        if self.reply:
            self.lines.put(
                'USIM_RESPONSE '
                + json.dumps(
                    {
                        'id': request['id'],
                        'ok': True,
                        'result': {'echo': request['operation']},
                    }
                )
            )

    def flush(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        item = self.lines.get()
        if item is None:
            raise StopIteration
        return item

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -1
        self.lines.put(None)

    def kill(self):
        self.terminate()

    def wait(self, timeout):
        return self.returncode

    def close(self):
        pass


def test_wire_matches_reply_id_and_returns_acknowledged_result() -> None:
    # Given a live reader and a fake pipe with exact ID echo.
    process = PipeProcess()
    client = WireClient(process)
    try:
        # When a request is written.
        result = client.request('service', {})
        # Then only the correlated acknowledgement supplies the result.
        assert result == {'echo': 'service'}
    finally:
        client.close()


def test_close_cancels_inflight_wire_request_without_polling() -> None:
    # Given a worker that has received a request but has not acknowledged it.
    process = PipeProcess(reply=False)
    client = WireClient(process)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(client.request, 'service', {})
        assert process.requested.wait(timeout=2)
        # When the owning session cancels the channel.
        client.close()
        # Then the blocked caller is woken by cancellation, not a timed sleep.
        with pytest.raises(ModelError):
            pending.result(timeout=2)


def test_wire_wrong_id_fails_pending_request() -> None:
    # Given a request whose worker sends an unrelated ID.
    process = PipeProcess(reply=False)
    client = WireClient(process)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(client.request, 'service', {})
            assert process.requested.wait(timeout=2)
            # When the malformed acknowledgement arrives.
            process.lines.put('USIM_RESPONSE {"id":"wrong","ok":true,"result":{}}')
            # Then it cannot complete a different request successfully.
            with pytest.raises(ModelError):
                pending.result(timeout=2)
    finally:
        client.close()


def test_worker_error_response_preserves_request_id() -> None:
    # Given a dispatch failure at the actual worker protocol boundary.
    class RejectingWorker:
        def dispatch(self, operation, payload, timeout):
            raise RuntimeError('native rejected')

    incoming = io.StringIO(
        json.dumps(
            {
                'id': 'abc',
                'operation': 'service',
                'payload': {},
                'timeout': 3,
            }
        )
        + '\n'
    )
    outgoing = io.StringIO()
    # When the worker frames the request.
    serve(RejectingWorker(), incoming, outgoing)
    # Then an error acknowledgement, not a success payload, reaches the caller.
    reply = json.loads(outgoing.getvalue().removeprefix('USIM_RESPONSE '))
    assert reply['id'] == 'abc' and reply['ok'] is False
    assert 'result' not in reply


@pytest.mark.parametrize('case', ['unavailable', 'timeout', 'rejected', 'success'])
def test_native_service_waits_for_real_ack(monkeypatch, tmp_path, case) -> None:
    # Given ROS message and client fakes at the native service boundary.
    future = Future()
    if case in {'rejected', 'success'}:
        future.set_result(SimpleNamespace(success=case == 'success', status_message='native'))
    client = SimpleNamespace(
        wait_for_service=lambda timeout_sec: case != 'unavailable',
        call_async=lambda request: future,
    )
    node = SimpleNamespace(create_client=lambda message_type, service: client)
    ros = SimpleNamespace(spin_until_future_complete=lambda node, future, timeout_sec: None)
    import sys

    monkeypatch.setitem(
        sys.modules,
        'rosidl_runtime_py.convert',
        SimpleNamespace(
            message_to_ordereddict=lambda response: {'success': response.success},
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        'rosidl_runtime_py.set_message',
        SimpleNamespace(
            set_message_fields=lambda request, fields: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        'gazebo_msgs.srv',
        SimpleNamespace(
            SetEntityState=SimpleNamespace(Request=SimpleNamespace),
        ),
    )
    worker = NativeWorker(node, ros, tmp_path)
    # When the real native dispatch path executes.
    if case == 'success':
        result = worker.dispatch('service', {'service': '/set_entity_state', 'fields': {}}, 3)
        # Then the response is grounded in service success.
        assert result == {'success': True}
    else:
        with pytest.raises((RuntimeError, TimeoutError)):
            worker.dispatch('service', {'service': '/set_entity_state', 'fields': {}}, 3)
        if case == 'timeout':
            assert future.cancelled()


def test_episode_configuration_is_rejected_before_opening_container(model, world) -> None:
    # Given an explicitly requested unsupported execution mode.
    backend = Backend(transport_factory=lambda _: pytest.fail('must not launch'))
    # When opening an episode.
    with pytest.raises(UnsupportedOperation):
        backend.open(SceneSpec(entities=(EntitySpec('robot', model),)), ExecutionSpec())
    # Then advertised execution modes remain continuous only.
    assert backend.describe().execution_modes == frozenset({'continuous'})
