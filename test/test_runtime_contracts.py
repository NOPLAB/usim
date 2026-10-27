"""Portable grouped contracts and native scene revision behavior."""

from dataclasses import replace

import pytest

from usim.capability import (
    CapabilityContext,
    CapabilityRequest,
    Phase,
    SupportReport,
    SupportStatus,
    UnsupportedOperation,
    assess_context,
)
from usim.facets import ControlTargets
from usim.groups import ControlBindings, JointControlGroup
from usim.model import ModelError, ModelFormat, ModelSource
from usim.scene import RemoveEntity, SceneBindings, SceneEditor, SetPose, SpawnEntity
from usim.specs import ControlSpec, EntitySpec, Pose, SensorSpec


@pytest.fixture
def source(tmp_path):
    path = tmp_path / 'robot.xml'
    path.write_text('<mujoco><worldbody/></mujoco>')
    return ModelSource.from_path(path)


def test_xml_model_detection_reads_content_instead_of_guessing_extension(source):
    # Given an XML model file.
    # When parsing the source boundary.
    # Then its actual root selects the model format.
    assert source.format is ModelFormat.MJCF


@pytest.mark.parametrize('suffix', ['.world', '.xml', '.sdf'])
def test_sdf_world_sources_keep_existing_gazebo_filename_conventions(tmp_path, suffix):
    # Given a native SDF world with a conventional Gazebo filename.
    path = tmp_path / ('room' + suffix)
    path.write_text('<sdf version="1.6"><world name="room"/></sdf>')
    # When resolving the portable source.
    source = ModelSource.from_path(path)
    # Then native world files route to an SDF provider, not an inferred robot importer.
    assert source.format is ModelFormat.SDF


@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_nonfinite_targets_reject_before_control_dispatch(value):
    # Given invalid physical targets.
    # When constructing a command.
    # Then no backend sees an invalid demand.
    with pytest.raises(ModelError):
        ControlTargets(positions={'joint': value})


@pytest.mark.parametrize(
    'settings',
    [
        {'joint_stiffness': (-1.0,)},
        {'joint_damping': (float('nan'),)},
        {'joint_force_limits': (0.0,)},
        {'joint_force_limits': (float('inf'),)},
        {'joint_stiffness': (10.0, 20.0)},
    ],
)
def test_invalid_joint_controller_settings_reject_before_native_loading(settings):
    # Given controller values that cannot describe the declared joint order.
    # When parsing an external controller specification.
    # Then no native backend is needed to reject invalid physical settings.
    with pytest.raises(ModelError):
        ControlSpec(joint_names=('hinge',), **settings)


def test_joint_controller_overrides_require_explicit_joint_order():
    # Given numeric gains without a declared mapping.
    # When parsing the controller boundary.
    # Then gains cannot silently bind to an engine-dependent order.
    with pytest.raises(ModelError):
        ControlSpec(joint_stiffness=(100.0,))


def test_targets_snapshot_caller_owned_mappings_and_batches():
    # Given a caller-owned batch and mapping.
    batch = [0.1, 0.2]
    values = {'joint': batch}
    command = ControlTargets(positions=values)
    # When the caller later mutates its inputs.
    batch[0] = 99
    values.clear()
    # Then the queued command retains the validated values.
    assert command.positions['joint'] == (0.1, 0.2)


def test_group_rejects_unsupported_second_mode_before_first_native_write():
    # Given a position-only controller and a mixed request.
    writes = []
    group = JointControlGroup(
        ControlSpec(joint_names=('arm', 'wheel')),
        ControlBindings(validate=lambda _: None, positions=writes.append),
    )
    # When submitting both modes together.
    with pytest.raises(UnsupportedOperation):
        group.apply(ControlTargets(positions={'arm': 0.1}, velocities={'wheel': 1.0}))
    # Then there was no partial native write.
    assert writes == []


@pytest.fixture
def editable():
    native_names = {'robot'}
    state = {'phase': Phase.PAUSED}

    def fail():
        state['phase'] = Phase.FAILED

    def apply(change):
        match change:
            case RemoveEntity(entity=name):
                native_names.remove(name)
            case SpawnEntity(spec=spec):
                native_names.add(spec.name)
            case SetPose():
                pass

    bindings = SceneBindings(
        assess=lambda change, request: SupportReport(SupportStatus.SUPPORTED),
        apply=apply,
        phase=lambda: state['phase'],
        fail=fail,
    )
    return SceneEditor(('robot',), bindings), native_names, state, bindings


def test_topology_change_invalidates_previously_issued_handles(editable, source):
    # Given a handle obtained before a topology change.
    editor, native_names, _, _ = editable
    handle = editor.handle('robot')
    # When adding an independent object.
    editor.apply(editor.plan(SpawnEntity(EntitySpec('box', source))))
    # Then native topology changed and the old generation is unusable.
    assert native_names == {'robot', 'box'}
    with pytest.raises(ModelError):
        editor.resolve(handle)


def test_state_change_preserves_entity_handle(editable):
    # Given an existing native entity handle.
    editor, _, _, _ = editable
    handle = editor.handle('robot')
    # When changing only its pose.
    editor.apply(editor.plan(SetPose('robot', Pose(position=(1, 0, 0)))))
    # Then the same topology generation remains valid.
    assert editor.resolve(handle) == 'robot'


def test_plan_rejects_a_changed_safe_point(editable):
    # Given an edit planned while paused.
    editor, native_names, state, _ = editable
    plan = editor.plan(RemoveEntity('robot'))
    # When execution has resumed before application.
    state['phase'] = Phase.RUNNING
    # Then no native deletion takes place.
    with pytest.raises(ModelError):
        editor.apply(plan)
    assert native_names == {'robot'}


def test_edit_rechecks_backend_conditions_before_mutating(editable):
    # Given a backend whose conditions change after planning.
    editor, native_names, _, bindings = editable
    allowed = [True]
    editor._bindings = replace(
        bindings,
        assess=lambda change, request: SupportReport(
            SupportStatus.SUPPORTED if allowed[0] else SupportStatus.UNSUPPORTED
        ),
    )
    plan = editor.plan(RemoveEntity('robot'))
    # When native readiness is lost.
    allowed[0] = False
    # Then the original positive assessment does not authorize the write.
    with pytest.raises(UnsupportedOperation):
        editor.apply(plan)
    assert native_names == {'robot'}


def test_native_mutation_failure_marks_session_failed(editable):
    # Given a failing native operation.
    editor, _, state, bindings = editable

    def fail_native(change):
        raise RuntimeError('native mutation failed')

    editor._bindings = replace(bindings, apply=fail_native)
    # When native application fails.
    with pytest.raises(RuntimeError):
        editor.apply(editor.plan(RemoveEntity('robot')))
    # Then the session cannot continue with unknown partial native state.
    assert state['phase'] is Phase.FAILED
    with pytest.raises(ModelError):
        editor.plan(RemoveEntity('robot'))


def test_foreign_handle_rejects_even_when_names_match(editable):
    # Given separate scenes with the same entity name.
    editor, _, _, bindings = editable
    other = SceneEditor(('robot',), bindings)
    # When resolving the other scene's handle.
    with pytest.raises(ModelError):
        editor.resolve(other.handle('robot'))


def test_invalid_sensor_rate_rejects_before_engine_initialization():
    # Given an invalid sensor cadence.
    # When defining the scene sensor.
    # Then optional native engines are unnecessary for rejection.
    with pytest.raises(ModelError):
        SensorSpec('camera', 'rgb', rate=0)


def test_explicit_control_partition_does_not_accept_wrong_mode():
    # Given a native velocity writer but a position-only joint binding.
    writes = []
    group = JointControlGroup(
        ControlSpec(joint_names=('arm',), position_joints=('arm',)),
        ControlBindings(validate=lambda _: None, velocities=writes.append),
    )
    # When addressing that position joint through velocity mode.
    with pytest.raises(ModelError):
        group.set_velocities({'arm': 1.0})
    # Then a mode mismatch never reaches the native writer.
    assert writes == []


def test_closed_scene_invalidates_existing_handles(editable):
    # Given a handle acquired from a ready scene.
    editor, _, state, _ = editable
    handle = editor.handle('robot')
    # When the scene is closed.
    state['phase'] = Phase.CLOSED
    # Then state access cannot retain a valid native handle.
    with pytest.raises(ModelError):
        editor.resolve(handle)


def test_live_capability_assessment_cannot_spoof_a_building_phase():
    # Given an installed provider and an actual closed runtime context.
    from usim import create_backend

    provider = create_backend('genesis')
    context = CapabilityContext(Phase.CLOSED, 1, 'episode')
    # When a caller supplies a different phase in its request.
    support = assess_context(provider, CapabilityRequest('models.load'), context)
    # Then assessment uses the runtime state rather than the caller's claimed phase.
    assert support.status is SupportStatus.UNSUPPORTED


def test_live_execution_assessment_matches_the_selected_driver():
    # Given an engine that advertises both execution modes.
    from usim import create_backend

    provider = create_backend('isaacsim')
    # When the actual session was opened with only an episode driver.
    support = assess_context(
        provider,
        CapabilityRequest('execution.continuous'),
        CapabilityContext(Phase.READY, 1, 'episode'),
    )
    # Then engine-level support cannot imply a live continuous driver exists.
    assert support.status is SupportStatus.UNSUPPORTED
