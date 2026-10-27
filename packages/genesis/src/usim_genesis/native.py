"""Genesis's dynamically imported native boundary and built entity bindings."""

from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np
from numpy.typing import NDArray

from usim.capability import assert_never
from usim.facets import JointLayout, StateSchema
from usim.model import ModelError, ModelFormat
from usim.specs import EntitySpec

from .models import ParsedAsset


def numpy(value: Any) -> NDArray[np.generic]:
    """Detach native Torch values at the optional SDK boundary."""
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu().numpy()
    return np.asarray(value).copy()


@dataclass(frozen=True, slots=True)
class BoundEntity:
    # Genesis is optional and has no dependency-free runtime stubs.
    native: Any
    spec: EntitySpec
    schema: StateSchema


def resolve_joint_name(bound: BoundEntity, name: str) -> str:
    """USD uses prim paths natively; accept a basename only when it is unambiguous."""
    names = tuple(joint.name for joint in bound.schema.joints)
    if name in names:
        return name
    if bound.spec.source.format is ModelFormat.USD:
        matches = tuple(candidate for candidate in names if candidate.rsplit('/', 1)[-1] == name)
        if len(matches) == 1:
            return matches[0]
    raise ModelError(f'Unknown or ambiguous joint name: {name}; available: {names}')


def add_entity(scene: Any, asset: ParsedAsset, spec: EntitySpec) -> Any:
    """Create one native entity, preserving fixed links and authored inertia."""
    gs = import_module('genesis')
    x, y, z, w = spec.pose.orientation_xyzw
    options = {
        'file': asset.root if asset.root is not None else str(asset.source.path),
        'pos': spec.pose.position,
        'quat': (w, x, y, z),
        'convexify': False,
        'decimate': False,
        'watertighten': None,
        'recompute_inertia': False,
        'default_armature': 0.0,
        'align': False,
        'batch_fixed_verts': True,
    }
    match asset.source.format:
        case ModelFormat.URDF:
            morph = gs.morphs.URDF(
                **options, fixed=spec.requirements.base == 'fixed', merge_fixed_links=False
            )
        case ModelFormat.MJCF:
            morph = gs.morphs.MJCF(**options)
        case ModelFormat.USD:
            fixed = (
                None
                if spec.requirements.base == 'authored'
                else (spec.requirements.base == 'fixed')
            )
            morph = gs.morphs.USD(**options, fixed=fixed, prim_path=asset.source.selector)
        case ModelFormat.SDF:
            raise AssertionError('SDF never passes import planning')
        case unreachable:
            assert_never(unreachable)
    return scene.add_entity(morph)


def bind(entity: Any, spec: EntitySpec, n_envs: int) -> BoundEntity:
    """Describe actual built generalized coordinates, including floating bases."""
    layouts = []
    for joint in entity.joints:
        qs = tuple(joint.qs_idx_local)
        dofs = tuple(joint.dofs_idx_local)
        kind = joint.type.name.lower()
        unit = 'm' if kind == 'prismatic' else 'rad'
        if len(qs) == 7:
            kind, unit = 'free', 'm,quaternion'
        elif len(qs) == 4:
            kind, unit = 'spherical', 'quaternion'
        layouts.append(
            JointLayout(
                joint.name,
                kind,
                (qs[0], qs[-1] + 1) if qs else (0, 0),
                (dofs[0], dofs[-1] + 1) if dofs else (0, 0),
                unit,
            )
        )
    return BoundEntity(entity, spec, StateSchema(spec.name, tuple(layouts), n_envs=n_envs))
