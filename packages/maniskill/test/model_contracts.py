"""Model import fidelity and lazy optional-dependency contracts."""

from importlib import import_module
import subprocess
import sys

import pytest

from usim.capability import CapabilityRequest, SupportStatus, UnsupportedOperation
from usim.model import ImportRequirements, ModelError, ModelSource
from usim.specs import AgentSpec, ControlSpec, SceneSpec, SensorSpec
from usim_maniskill.backend import Backend
from packages.maniskill.test.native_env import direct


def test_mjcf_defaults_and_native_top_level_includes_are_inspected(tmp_path):
    included = tmp_path / 'body.xml'
    included.write_text(
        '<mujocoinclude><worldbody><body name="base"><joint name="hinge"/>'
        '<geom type="box" size=".1 .1 .1"/></body></worldbody></mujocoinclude>'
    )
    path = tmp_path / 'robot.xml'
    path.write_text(
        '<mujoco><default><joint damping="1"/></default><include file="body.xml"/></mujoco>'
    )
    source = ModelSource.from_path(path)
    backend = Backend()
    requirements = ImportRequirements(
        preserve=frozenset({'joint_structure', 'collision'}),
        allow_loss=frozenset({'inertia'}),
    )
    with backend.models.load(backend.models.plan(source, requirements)) as model:
        assert model.value.joints == (('hinge', 'hinge'),)
        assert model.report.lost == frozenset({'inertia'})


def test_fidelity_and_unsupported_features_reject_before_native_mutation(native, sources):
    backend = Backend()
    with pytest.raises(ModelError, match='inertia'):
        backend.open(SceneSpec(agents=(AgentSpec('robot', sources[1]),)))
    with pytest.raises(ModelError, match='known control'):
        backend.open(
            SceneSpec(
                agents=(
                    direct(
                        sources[0],
                        control=ControlSpec(position_joints=('missing',)),
                    ),
                )
            )
        )
    with pytest.raises(UnsupportedOperation, match='lidar'):
        backend.open(
            SceneSpec(
                agents=(
                    direct(
                        sources[0],
                        sensors=(SensorSpec('scan', 'lidar'),),
                    ),
                )
            )
        )
    assert not native[0]
    assert (
        backend.assess(CapabilityRequest('not.a.capability')).status is SupportStatus.UNSUPPORTED
    )


def test_import_and_backend_discovery_are_lazy():
    result = subprocess.run(
        [
            sys.executable,
            '-c',
            'import sys; from usim_maniskill import Backend; Backend().describe(); '
            "assert not {'numpy', 'torch', 'gymnasium', 'sapien', 'mani_skill'} & set(sys.modules)",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert import_module('usim_maniskill').Backend is Backend
