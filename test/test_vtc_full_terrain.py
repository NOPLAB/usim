"""Derived road correction must not flatten unrelated terrain or create tile seams."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('scipy')

SCRIPT = Path(__file__).parents[1] / 'src' / 'usim_env_vtc' / 'repair_terrain.py'
SPEC = importlib.util.spec_from_file_location('vtc_full_repair_terrain', SCRIPT)
assert SPEC is not None and SPEC.loader is not None
repair = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repair)


def test_abrupt_paved_interior_becomes_continuous_without_changing_unpaved_ground():
    # Given a source road step and neighboring grass with independent relief.
    heights = np.zeros((181, 181), dtype=np.float32)
    heights[:, 91:] = 1
    heights[:20] = -2
    paved = np.zeros_like(heights, dtype=bool)
    paved[40:140] = True
    before = heights.copy()
    # When deriving the road correction.
    actual = repair.repair_heights(heights, paved)
    # Then the abrupt in-road step becomes a gradual transition, retaining the archive.
    assert np.max(np.abs(np.diff(actual[70:110], axis=1))) < 0.08
    assert np.array_equal(actual[~paved], before[~paved])
    assert np.array_equal(heights, before)
    assert np.isfinite(actual).all()


def test_gentle_grade_and_material_boundary_curb_are_preserved():
    # Given a gentle road grade, with a curb at the boundary to unpaved terrain.
    heights = np.tile(np.arange(181, dtype=np.float32) * 0.003, (181, 1))
    heights[:40] += 0.15
    paved = np.zeros_like(heights, dtype=bool)
    paved[40:140] = True
    # When deriving the road correction.
    actual = repair.repair_heights(heights, paved)
    # Then neither an ordinary grade nor the material-boundary curb is flattened.
    assert np.array_equal(actual, heights)


def test_local_mesh_correction_wins_over_parent_terrain_reference(tmp_path):
    pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom

    # Given original terrain introduced by a parent reference, not a leaf payload.
    original = Usd.Stage.CreateNew(str(tmp_path / 'terrain.usda'))
    UsdGeom.Xform.Define(original, '/Terrain')
    mesh = UsdGeom.Mesh.Define(original, '/Terrain/Tile/Road')
    source = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    mesh.GetPointsAttr().Set(source.tolist())
    original.GetRootLayer().Save()
    base = Usd.Stage.CreateNew(str(tmp_path / 'base.usda'))
    terrain = UsdGeom.Xform.Define(base, '/World/Terrain')
    terrain.GetPrim().GetReferences().AddReference('terrain.usda', '/Terrain')
    base.GetRootLayer().Save()
    correction = Usd.Stage.CreateNew(str(tmp_path / 'correction.usdc'))
    world = Usd.Stage.CreateNew(str(tmp_path / 'world.usda'))
    world.GetRootLayer().subLayerPaths = ['correction.usdc', 'base.usda']
    corrected = source.copy()
    corrected[:, 2] = 1
    # When authoring a correction on the full composed mesh path and reopening.
    repair.author_road_mesh(correction, '/World/Terrain/Tile/Road', corrected)
    correction.GetRootLayer().Save()
    world.GetRootLayer().Save()
    reopened = Usd.Stage.Open(str(tmp_path / 'world.usda'))
    # Then the actual composed mesh, not just an isolated correction file, is corrected.
    actual = UsdGeom.Mesh(reopened.GetPrimAtPath('/World/Terrain/Tile/Road'))
    assert np.array_equal(np.asarray(actual.GetPointsAttr().Get()), corrected)
    assert np.array_equal(np.asarray(mesh.GetPointsAttr().Get()), source)
