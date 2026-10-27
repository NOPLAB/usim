"""USD namespace aliases resolve only when native joint identity is unambiguous."""

import pytest

from usim import EntitySpec, JointLayout, ModelError, ModelFormat, ModelSource, StateSchema
from usim_genesis.native import BoundEntity, resolve_joint_name


def bound_usd(tmp_path, names):
    path = tmp_path / 'names.usda'
    path.write_text('#usda 1.0\n')
    source = ModelSource(path, ModelFormat.USD)
    joints = tuple(
        JointLayout(name, 'revolute', (index, index + 1), (index, index + 1))
        for index, name in enumerate(names)
    )
    return BoundEntity(None, EntitySpec('robot', source), StateSchema('robot', joints))


def test_usd_basename_resolves_to_the_native_prim_path(tmp_path):
    # Given a native importer whose joint identifiers are full USD paths.
    bound = bound_usd(tmp_path, ('/Robot/hinge',))
    # When a caller uses the source prim's unique basename.
    native_name = resolve_joint_name(bound, 'hinge')
    # Then the command addresses the real native joint, not a fabricated short name.
    assert native_name == '/Robot/hinge'


def test_ambiguous_usd_basename_requires_an_explicit_prim_path(tmp_path):
    # Given two authored joints with the same basename.
    bound = bound_usd(tmp_path, ('/Robot/left/hinge', '/Robot/right/hinge'))
    # When a caller leaves their intended native identity ambiguous.
    with pytest.raises(ModelError):
        resolve_joint_name(bound, 'hinge')
    # Then explicit paths remain distinct and resolvable.
    assert resolve_joint_name(bound, '/Robot/right/hinge') == '/Robot/right/hinge'
