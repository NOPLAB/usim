"""Native assessment evidence and actual import name mappings remain distinct."""

from dataclasses import dataclass
from xml.etree import ElementTree

from usim.model import (
    ImportContext,
    ImportReport,
    ImportRequirements,
    ImportedModel,
    ModelFormat,
    ModelSource,
)
from usim.model_io import ModelIO


@dataclass(frozen=True, slots=True)
class XmlLoader:
    format: ModelFormat = ModelFormat.MJCF

    def assess(self, source: ModelSource, requirements: ImportRequirements) -> ImportReport:
        ElementTree.parse(source.path)
        return ImportReport(lost=frozenset({'materials'}))

    def load(self, source: ModelSource, context: ImportContext) -> ImportedModel[str]:
        root = ElementTree.parse(source.path).getroot()
        return ImportedModel(
            source,
            root.tag,
            ImportReport(joint_mapping=(('source_joint', 'native_joint'),)),
        )


def test_allowed_native_loss_survives_a_sparse_loader_report(tmp_path):
    # Given a native importer which reports no material detail in its returned value.
    path = tmp_path / 'model.xml'
    path.write_text('<mujoco/>')
    models = ModelIO.compose((XmlLoader(),))
    requirements = ImportRequirements(preserve=frozenset(), allow_loss=frozenset({'materials'}))
    plan = models.plan(ModelSource.from_path(path), requirements)
    # When the model is actually imported.
    with models.load(plan) as loaded:
        # Then the preflight loss evidence is retained with the real native name mapping.
        assert loaded.report.lost == frozenset({'materials'})
        assert loaded.report.joint_mapping == (('source_joint', 'native_joint'),)
        assert loaded.value == 'mujoco'
