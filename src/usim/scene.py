"""Explicit scene edits, safe-point assessment and revision-scoped handles."""

from dataclasses import dataclass, replace
from math import isfinite
from typing import Callable
from uuid import uuid4

from usim.capability import CapabilityRequest, Phase, SupportReport
from usim.model import ModelError, ModelSource
from usim.specs import EntitySpec, Pose


@dataclass(frozen=True, slots=True)
class SetPose:
    entity: str
    pose: Pose


@dataclass(frozen=True, slots=True)
class SetJointState:
    entity: str
    positions: tuple[tuple[str, float], ...] = ()
    velocities: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if not self.positions and not self.velocities:
            raise ModelError('A joint state edit must specify positions or velocities')
        for targets in (self.positions, self.velocities):
            names = tuple(name for name, _ in targets)
            if len(names) != len(set(names)) or any(not name.strip() for name in names):
                raise ModelError('Joint state names must be nonempty and unique')
            if not all(isfinite(value) for _, value in targets):
                raise ModelError('Joint state values must be finite')


@dataclass(frozen=True, slots=True)
class SetParameter:
    entity: str
    parameter: str
    value: float

    def __post_init__(self) -> None:
        if not self.parameter.strip() or not isfinite(self.value):
            raise ModelError('A parameter edit needs a name and a finite value')


@dataclass(frozen=True, slots=True)
class SpawnEntity:
    spec: EntitySpec


@dataclass(frozen=True, slots=True)
class RemoveEntity:
    entity: str


@dataclass(frozen=True, slots=True)
class ReplaceModel:
    entity: str
    source: ModelSource


@dataclass(frozen=True, slots=True)
class EditArticulation:
    """Replace an articulation's declared topology with the supplied asset."""

    entity: str
    source: ModelSource


SceneChange = (
    SetPose
    | SetJointState
    | SetParameter
    | SpawnEntity
    | RemoveEntity
    | ReplaceModel
    | EditArticulation
)


def operation(change: SceneChange) -> str:
    names = {
        SetPose: 'scene.set_pose',
        SetJointState: 'scene.set_joint_state',
        SetParameter: 'scene.set_parameter',
        SpawnEntity: 'scene.spawn',
        RemoveEntity: 'scene.remove',
        ReplaceModel: 'scene.replace',
        EditArticulation: 'scene.edit_articulation',
    }
    return names[type(change)]


@dataclass(frozen=True, slots=True)
class EntityHandle:
    owner: str
    name: str
    generation: int


@dataclass(frozen=True, slots=True)
class EditPlan:
    owner: str
    revision: int
    phase: Phase
    change: SceneChange
    request: CapabilityRequest
    support: SupportReport


@dataclass(frozen=True, slots=True)
class SceneBindings:
    assess: Callable[[SceneChange, CapabilityRequest], SupportReport]
    apply: Callable[[SceneChange], None]
    phase: Callable[[], Phase]
    fail: Callable[[], None]
    n_envs: int = 1


class SceneEditor:
    """Track mutable native scene revisions; never simulate rollback guarantees."""

    def __init__(self, names: tuple[str, ...], bindings: SceneBindings):
        self._owner: str = uuid4().hex
        self._revision: int = 0
        self._names: set[str] = set(names)
        self._bindings: SceneBindings = bindings

    @property
    def revision(self) -> int:
        return self._revision

    def handle(self, name: str) -> EntityHandle:
        if self._bindings.phase() in {Phase.CLOSED, Phase.FAILED}:
            raise ModelError('Cannot acquire handles from a closed or failed scene')
        if name not in self._names:
            raise ModelError(f'Unknown scene entity: {name}')
        return EntityHandle(self._owner, name, self._revision)

    def resolve(self, handle: EntityHandle) -> str:
        if self._bindings.phase() in {Phase.CLOSED, Phase.FAILED}:
            raise ModelError('Cannot resolve handles in a closed or failed scene')
        if handle.owner != self._owner or handle.generation != self._revision:
            raise ModelError('Entity handle belongs to another scene or an old topology')
        if handle.name not in self._names:
            raise ModelError(f'Unknown scene entity: {handle.name}')
        return handle.name

    def plan(self, change: SceneChange, *, require_same_episode: bool = True) -> EditPlan:
        phase = self._bindings.phase()
        if phase in {Phase.CLOSED, Phase.FAILED}:
            raise ModelError(f'Cannot edit a {phase.value} scene')
        match change:
            case SpawnEntity(spec=spec):
                if spec.name in self._names:
                    raise ModelError(f'Duplicate scene entity: {spec.name}')
            case (
                SetPose(entity=name)
                | SetJointState(entity=name)
                | SetParameter(entity=name)
                | RemoveEntity(entity=name)
                | ReplaceModel(entity=name)
                | EditArticulation(entity=name)
            ):
                if name not in self._names:
                    raise ModelError(f'Unknown scene entity: {name}')
        request = CapabilityRequest(
            operation(change), phase, require_same_episode, self._bindings.n_envs
        )
        support = self._bindings.assess(change, request)
        if (
            request.operation
            in {'scene.spawn', 'scene.remove', 'scene.replace', 'scene.edit_articulation'}
            and support.supported
        ):
            support = replace(
                support, effects=tuple(dict.fromkeys(support.effects + ('invalidate_handles',)))
            )
        return EditPlan(self._owner, self._revision, phase, change, request, support)

    def apply(self, plan: EditPlan) -> None:
        if plan.owner != self._owner or plan.revision != self._revision:
            raise ModelError('Scene edit plan belongs to another scene or an old revision')
        if self._bindings.phase() is not plan.phase:
            raise ModelError('Scene phase changed after edit planning')
        plan.support.require()
        self._bindings.assess(plan.change, plan.request).require()
        succeeded = False
        try:
            self._bindings.apply(plan.change)
            succeeded = True
        finally:
            if not succeeded:
                self._bindings.fail()
        match plan.change:
            case SpawnEntity(spec=spec):
                self._names.add(spec.name)
                self._revision += 1
            case RemoveEntity(entity=name):
                self._names.remove(name)
                self._revision += 1
            case ReplaceModel() | EditArticulation():
                self._revision += 1
            case SetPose() | SetJointState() | SetParameter():
                pass
