"""Native USD placement and world rebuilding, separate from edit policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from usim.model import ModelError
from usim.model_io import OwnedModel
from usim.specs import ControlSpec, EntitySpec, ExecutionSpec, Pose
from usim_isaacsim.native_runtime import ImportRuntime
from usim_isaacsim.usd_loader import UsdAsset

if TYPE_CHECKING:
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation, SingleRigidPrim, SingleXFormPrim
    from pxr import Usd


@dataclass(frozen=True, slots=True)
class NativeEntity:
    spec: EntitySpec
    imported: OwnedModel[UsdAsset]
    prim_path: str
    native: SingleArticulation | SingleRigidPrim | SingleXFormPrim
    control: ControlSpec
    joint_names: tuple[str, ...] = ()
    joint_indices: tuple[int, ...] = ()


class NativeScene:
    """Mutable native scene, rebuilt only at an explicitly permitted episode boundary."""

    def __init__(self, runtime: ImportRuntime, execution: ExecutionSpec) -> None:
        self.runtime, self.execution = runtime, execution
        self.world: World | None = None
        self.stage: Usd.Stage | None = None
        self.entities: dict[str, NativeEntity] = {}
        self._serial = 0

    def begin(self) -> None:
        import omni.usd
        from isaacsim.core.api import World
        from pxr import UsdGeom

        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=1 / self.execution.sim_rate,
            rendering_dt=1 / self.execution.sim_rate,
        )
        self.stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(self.stage, '/World/Usim')

    def add(self, spec: EntitySpec, control: ControlSpec = ControlSpec()) -> None:
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim, SingleXFormPrim
        from pxr import UsdGeom

        assert self.stage is not None and self.world is not None
        plan = self.runtime.models.plan(spec.source, spec.requirements)
        imported = self.runtime.resources.enter_context(self.runtime.models.load(plan))
        asset = imported.value
        self._serial += 1
        path = f'/World/Usim/entity_{self._serial}'
        prim = UsdGeom.Xform.Define(self.stage, path).GetPrim()
        if not prim.GetReferences().AddReference(str(asset.path), asset.root):
            raise ModelError(f'Could not reference USD asset: {asset.path}')
        if asset.physics_variant is not None:
            prim.GetVariantSet('Physics').SetVariantSelection(asset.physics_variant)
        self.stage.Load()
        match spec.requirements.role:
            case 'articulation':
                assert asset.articulation is not None
                native_path = path + asset.articulation[len(asset.root) :]
                native = SingleArticulation(prim_path=native_path, name=spec.name)
            case 'rigid_body':
                assert asset.rigid_body is not None
                native_path = path + asset.rigid_body[len(asset.root) :]
                native = SingleRigidPrim(prim_path=native_path, name=spec.name)
            case 'visual' | 'scene':
                native = SingleXFormPrim(prim_path=path, name=spec.name)
            case _:
                raise ModelError(f'Unsupported role: {spec.requirements.role}')
        self.world.scene.add(native)
        self.entities[spec.name] = NativeEntity(spec, imported, path, native, control)

    def initialize(self) -> None:
        from dataclasses import replace

        assert self.world is not None
        self.world.reset()
        for name, entry in tuple(self.entities.items()):
            self.pose(name, entry.spec.pose)
            if entry.spec.requirements.role != 'articulation':
                continue
            robot = entry.native
            names = entry.control.joint_names or tuple(robot.dof_names)
            kinds = dict(entry.imported.value.joints)
            if not names or any(robot.dof_names.count(name) != 1 for name in names):
                raise ModelError('Selected joints must occur exactly once in the articulation')
            if not set(names) <= kinds.keys():
                raise ModelError('Native DOFs differ from the selected USD scalar joints')
            indices = tuple(robot.get_dof_index(joint) for joint in names)
            control = entry.control
            if not set(
                control.position_joints + control.velocity_joints + control.effort_joints
            ) <= set(names):
                raise ModelError('Control group contains an unselected joint')
            self.entities[name] = replace(entry, joint_names=names, joint_indices=indices)
            controller = robot.get_articulation_controller()
            kp, kd = (np.array(values, copy=True) for values in controller.get_gains())
            if control.joint_stiffness is not None:
                kp[list(indices)] = control.joint_stiffness
            if control.joint_damping is not None:
                kd[list(indices)] = control.joint_damping
            for joint, index in zip(names, indices):
                mode = (
                    'effort'
                    if joint in control.effort_joints
                    else 'velocity'
                    if joint in control.velocity_joints
                    else 'position'
                )
                if mode != 'position':
                    kp[index] = 0.0
                if mode == 'effort':
                    kd[index] = 0.0
                controller.switch_dof_control_mode(dof_index=index, mode=mode)
            controller.set_gains(kps=kp, kds=kd)
            if control.joint_force_limits is not None:
                controller.set_max_efforts(
                    np.asarray(control.joint_force_limits), joint_indices=np.asarray(indices)
                )
            if control.initial_qpos is not None:
                values = np.asarray(control.initial_qpos, dtype=float)
                properties = robot.dof_properties
                if values.shape != (len(names),) or np.any(
                    (values < properties['lower'][list(indices)])
                    | (values > properties['upper'][list(indices)])
                ):
                    raise ModelError('Initial positions do not fit selected joint limits')
                robot.set_joint_positions(values, joint_indices=np.asarray(indices))
            robot.set_joint_velocities(np.zeros(len(names)), joint_indices=np.asarray(indices))
            from isaacsim.core.utils.types import ArticulationAction

            position_names = set(control.position_joints) or (
                set(names) - set(control.velocity_joints) - set(control.effort_joints)
            )
            for group, field in (
                (position_names, 'joint_positions'),
                (set(control.velocity_joints), 'joint_velocities'),
                (set(control.effort_joints), 'joint_efforts'),
            ):
                active = np.asarray(
                    [index for joint, index in zip(names, indices) if joint in group]
                )
                if active.size:
                    targets = (
                        robot.get_joint_positions(joint_indices=active)
                        if field == 'joint_positions'
                        else np.zeros(len(active))
                    )
                    robot.apply_action(
                        ArticulationAction(**{field: targets}, joint_indices=active)
                    )
        self.world.pause()

    def pose(self, name: str, pose: Pose) -> None:
        entry = self.entities[name]
        q = pose.orientation_xyzw
        entry.native.set_world_pose(
            position=np.asarray(pose.position), orientation=np.asarray((q[3], *q[:3]))
        )

    def dispose(self) -> None:
        try:
            if self.world is not None:
                try:
                    self.world.stop()
                    self.world.clear()
                finally:
                    self.world.clear_instance()
        finally:
            self.entities.clear()
            if self.stage is not None:
                self.stage.RemovePrim('/World/Usim')
            self.world = self.stage = None
