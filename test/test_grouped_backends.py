"""Installed provider discovery and shared requests cross distribution boundaries."""

import subprocess
import sys

import pytest

from usim import (
    CapabilityRequest,
    ModelSource,
    Phase,
    SupportStatus,
    create_backend,
    list_backends,
    ModelIO,
)


@pytest.mark.parametrize('engine', ['genesis', 'isaacsim', 'maniskill', 'gazebo'])
def test_grouped_provider_is_registered_and_rejects_closed_scene(engine):
    # Given an installed independent backend.
    assert engine in list_backends()
    provider = create_backend(engine)
    # When querying a closed session context.
    support = provider.assess(CapabilityRequest('models.load', phase=Phase.CLOSED))
    # Then discovery identifies the engine but never authorizes a closed-scene operation.
    assert provider.describe().name == engine
    assert support.status is SupportStatus.UNSUPPORTED


@pytest.mark.parametrize('engine', ['genesis', 'isaacsim', 'maniskill', 'gazebo'])
def test_provider_description_and_assessment_agree_about_execution(engine):
    # Given a backend's advertised execution modes.
    provider = create_backend(engine)
    modes = provider.describe().execution_modes
    # When asking about episode support.
    support = provider.assess(CapabilityRequest('execution.episode'))
    # Then a continuous-only backend cannot acquire invented lockstep stepping.
    assert (support.status is not SupportStatus.UNSUPPORTED) == ('episode' in modes)


@pytest.mark.parametrize('engine', ['genesis', 'isaacsim', 'maniskill', 'gazebo'])
def test_named_control_capabilities_use_shared_operation_names(engine):
    provider = create_backend(engine)
    operations = provider.describe().operations
    assert {'control.velocity', 'control.effort'} <= operations
    assert not {'control.positions', 'control.velocities', 'control.efforts'} & operations
    if engine != 'gazebo':
        assert 'control.position' in operations
    assert {'sensors.rgb', 'sensors.depth', 'sensors.rgbd'} <= operations


def test_core_grouped_discovery_imports_no_optional_engines():
    # Given an interpreter that forbids optional physics and array libraries.
    code = """
import importlib.abc
import sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'numpy', 'torch', 'genesis', 'mani_skill', 'sapien', 'isaacsim', 'pxr', 'rclpy'
        }:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Guard())
from usim import ModelIO, ModelSource, SceneSpec, list_backends
assert {'genesis', 'isaacsim', 'maniskill', 'gazebo'} <= set(list_backends())
"""
    # When importing the public grouped API and discovering providers.
    result = subprocess.run(
        [sys.executable, '-B', '-c', code], capture_output=True, text=True, timeout=30
    )
    # Then metadata discovery and portable contracts need no native dependency.
    assert result.returncode == 0, result.stdout + result.stderr


def test_legacy_capabilities_export_remains_callable():
    # Given the public package after all grouped capability modules were imported.
    from usim import capabilities

    # When using the original execution-capability function.
    result = capabilities('genesis')
    # Then adding the new capability module did not replace the legacy callable.
    assert result == frozenset({'episode'})


@pytest.mark.parametrize('engine', ['genesis', 'isaacsim', 'maniskill', 'gazebo'])
def test_description_formats_follow_composed_providers(engine, monkeypatch):
    # Given a provider whose model composition is replaced with an empty composition.
    provider = create_backend(engine)
    assert provider.describe().model_formats
    monkeypatch.setattr(provider, 'models', ModelIO.compose(()))
    # Then discovery reflects installed loaders rather than a hardcoded format list.
    assert provider.describe().model_formats == frozenset()


@pytest.mark.parametrize('engine', ['genesis', 'isaacsim', 'maniskill', 'gazebo'])
def test_format_planning_does_not_initialize_native_physics(engine, tmp_path):
    # Given a real, small articulation with complete inertial and collision data.
    path = tmp_path / 'robot.urdf'
    inertial = (
        '<inertial><mass value="1"/>'
        '<inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>'
    )
    geometry = '<collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision>'
    path.write_text(
        '<robot name="test"><link name="base">'
        + inertial
        + geometry
        + '</link><link name="arm">'
        + inertial
        + geometry
        + '</link>'
        '<joint name="hinge" type="revolute"><parent link="base"/><child link="arm"/>'
        '<axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="10" velocity="2"/>'
        '</joint></robot>'
    )
    provider = create_backend(engine)
    # When planning that actual source through the selected provider's loaders.
    plan = provider.models.plan(ModelSource.from_path(path))
    # Then a native route or explicit conditional conversion exists before engine startup.
    assert plan.support.status in {SupportStatus.SUPPORTED, SupportStatus.CONDITIONAL}
    assert plan.route
