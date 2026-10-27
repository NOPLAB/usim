"""Parsed USD assets with explicit selection and authored-role validation."""

from collections.abc import Callable
from dataclasses import dataclass
from math import isclose
from pathlib import Path

from usim.model import (
    ImportContext,
    ImportedModel,
    ImportReport,
    ImportRequirements,
    ModelError,
    ModelFormat,
    ModelSource,
)


@dataclass(frozen=True, slots=True)
class UsdAsset:
    path: Path
    root: str
    articulation: str | None
    rigid_body: str | None
    joints: tuple[tuple[str, str], ...]
    physics_variant: str | None = None


@dataclass(frozen=True, slots=True)
class UsdLoader:
    require_runtime: Callable[[], None]
    format: ModelFormat = ModelFormat.USD

    def _inspect(
        self, source: ModelSource, requirements: ImportRequirements
    ) -> tuple[UsdAsset, ImportReport]:
        self.require_runtime()
        from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils

        if source.format != ModelFormat.USD:
            raise ModelError('Isaac native loader requires USD')
        stage = Usd.Stage.Open(str(source.path))
        if stage is None:
            raise ModelError(f'Cannot open USD: {source.path}')
        if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z or not isclose(
            UsdGeom.GetStageMetersPerUnit(stage), 1.0
        ):
            raise ModelError('USD assets must be metre-scale and Z-up')
        _, _, unresolved = UsdUtils.ComputeAllDependencies(str(source.path))
        if unresolved:
            raise ModelError(f'Unresolved USD dependencies: {unresolved}')
        selected = (
            stage.GetPrimAtPath(source.selector) if source.selector else stage.GetDefaultPrim()
        )
        if not selected or not selected.IsActive():
            raise ModelError('USD requires an active selected prim or a default prim')
        if not selected.IsA(UsdGeom.Xformable):
            raise ModelError('Selected USD prim must be transformable')
        variants = selected.GetVariantSet('Physics')
        physics_variant = 'physx' if 'physx' in variants.GetVariantNames() else None
        if physics_variant is not None:
            stage.SetEditTarget(stage.GetSessionLayer())
            variants.SetVariantSelection(physics_variant)
            stage.Load()
        roots = []
        bodies = []
        joints = []
        physics_joints = []
        prims = tuple(Usd.PrimRange(selected))
        for prim in prims:
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                roots.append(str(prim.GetPath()))
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                bodies.append(prim)
            if prim.IsA(UsdPhysics.Joint):
                physics_joints.append(prim)
                if prim.IsA(UsdPhysics.RevoluteJoint):
                    joints.append((prim.GetName(), 'revolute'))
                elif prim.IsA(UsdPhysics.PrismaticJoint):
                    joints.append((prim.GetName(), 'prismatic'))
                elif not prim.IsA(UsdPhysics.FixedJoint):
                    raise ModelError('Only scalar revolute and prismatic USD joints are supported')
        if len({name for name, _ in joints}) != len(joints):
            raise ModelError('USD scalar joint names must be unique within an entity')
        root_path = str(selected.GetPath())
        for prim in physics_joints:
            joint = UsdPhysics.Joint(prim)
            for target in joint.GetBody0Rel().GetTargets() + joint.GetBody1Rel().GetTargets():
                path = str(target)
                if path != root_path and not path.startswith(root_path + '/'):
                    raise ModelError(
                        'Selected USD articulation references a body outside its root'
                    )
        match requirements.role:
            case 'articulation':
                if len(roots) != 1 or not joints:
                    raise ModelError(
                        'Articulation role requires one root and scalar movable joints'
                    )
            case 'rigid_body':
                if roots or len(bodies) != 1 or physics_joints:
                    raise ModelError(
                        'Rigid-body role requires one body without articulation joints'
                    )
            case 'visual':
                if (
                    roots
                    or bodies
                    or physics_joints
                    or any(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in prims)
                ):
                    raise ModelError('Visual role requires an asset without authored physics')
            case 'scene':
                pass
        fixed = any(
            prim.IsA(UsdPhysics.FixedJoint)
            and (
                not UsdPhysics.Joint(prim).GetBody0Rel().GetTargets()
                or not UsdPhysics.Joint(prim).GetBody1Rel().GetTargets()
            )
            for prim in physics_joints
        )
        if requirements.base != 'authored':
            if not roots or fixed != (requirements.base == 'fixed'):
                raise ModelError(
                    'USD base mode differs from request; author the base in the asset'
                )
        # Referencing the selected subtree does not rewrite these authored USD fields.
        # Other fidelity claims (controllers, sensors, engine semantics) remain unknown.
        preserved = frozenset({'joint_structure', 'inertia', 'collision', 'materials', 'base'})
        report = ImportReport(
            preserved=preserved,
            unknown=requirements.preserve - preserved,
            joint_mapping=tuple((name, name) for name, _ in joints),
            link_mapping=tuple((prim.GetName(), prim.GetName()) for prim in bodies),
        )
        return UsdAsset(
            source.path,
            root_path,
            roots[0] if len(roots) == 1 else None,
            str(bodies[0].GetPath()) if len(bodies) == 1 else None,
            tuple(joints),
            physics_variant,
        ), report

    def assess(self, source: ModelSource, requirements: ImportRequirements) -> ImportReport:
        return self._inspect(source, requirements)[1]

    def load(self, source: ModelSource, context: ImportContext) -> ImportedModel[UsdAsset]:
        asset, report = self._inspect(source, context.requirements)
        return ImportedModel(source, asset, report)
