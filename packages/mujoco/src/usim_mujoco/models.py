"""Conservative import planning followed by MuJoCo's native XML compiler."""

from dataclasses import dataclass
from xml.etree import ElementTree as ET

from usim.model import (
    ImportContext,
    ImportedModel,
    ImportReport,
    ImportRequirements,
    ModelError,
    ModelFormat,
    ModelSource,
)
from usim.model_io import ModelIO


@dataclass(frozen=True, slots=True)
class Loader:
    format: ModelFormat

    def assess(self, source: ModelSource, requirements: ImportRequirements) -> ImportReport:
        if source.format is not self.format:
            raise ModelError('Wrong format for the MuJoCo native loader')
        if source.selector is not None or source.resources:
            raise ModelError(
                'MuJoCo requires a complete model with native relative resource paths'
            )
        root = ET.parse(source.path).getroot()
        expected = 'mujoco' if self.format is ModelFormat.MJCF else 'robot'
        if root.tag != expected:
            raise ModelError(f'Expected a {expected} XML root')
        lost = set()
        if self.format is ModelFormat.URDF:
            for tag, feature in (
                ('transmission', 'actuators'),
                ('.//gazebo', 'sensors'),
                ('.//sensor', 'sensors'),
            ):
                if root.find(tag) is not None:
                    lost.add(feature)
            if root.find('.//mimic') is not None:
                raise ModelError('MuJoCo URDF imports do not preserve mimic joint constraints')
            if any(
                joint.get('type') not in {'fixed', 'revolute', 'continuous', 'prismatic'}
                for joint in root.findall('joint')
            ):
                raise ModelError('Unsupported URDF joint type for native MuJoCo import')
        preserved = {'joint_structure', 'inertia', 'collision'} - lost
        if self.format is ModelFormat.MJCF:
            preserved |= {'actuators', 'sensors', 'solver', 'visual'}
        return ImportReport(
            preserved=frozenset(preserved),
            lost=frozenset(lost),
            unknown=requirements.preserve - preserved - lost,
            notes=('Native MuJoCo compilation is performed when the import plan is loaded.',),
        )

    def load(self, source: ModelSource, context: ImportContext):
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(source.path))
        report = self.assess(source, context.requirements)
        free_base = any(int(kind) == int(mujoco.mjtJoint.mjJNT_FREE) for kind in model.jnt_type)
        base = context.requirements.base
        if base != 'authored' and (base == 'floating') != free_base:
            raise ModelError('Requested base mode must match the authored native model')
        if context.requirements.role == 'rigid_body' and model.njnt:
            raise ModelError('A rigid_body import cannot contain active joints')
        return ImportedModel(source, model, report)


def model_io():
    return ModelIO.compose((Loader(ModelFormat.MJCF), Loader(ModelFormat.URDF)))
