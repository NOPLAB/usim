"""Envelope-level buildings get original facades without changing footprint or occupancy."""

import pytest

pytest.importorskip('pxr')
from pxr import Usd, UsdGeom, UsdPhysics

from usim_env_vtc.authored_buildings import author_building, building_dimensions


def test_generic_facade_keeps_unit_bounds_and_original_solid_collision():
    # Given a large envelope with no surveyed facade mesh.
    stage = Usd.Stage.CreateInMemory()
    # When adding independently designed facade bays.
    author_building(stage, '/Building', (40, 60, 18))
    # Then windows are visible detail, while placement/solid occupancy can remain exact.
    root = stage.GetPrimAtPath('/Building')
    bounds = (
        UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default'])
        .ComputeWorldBound(root)
        .ComputeAlignedRange()
    )
    assert tuple(bounds.GetMin()) == pytest.approx((-0.5, -0.5, -0.5))
    assert tuple(bounds.GetMax()) == pytest.approx((0.5, 0.5, 0.5))
    windows = [prim for prim in stage.Traverse() if prim.GetName().endswith('_Window')]
    assert len(windows) > 200
    colliders = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)]
    assert [str(prim.GetPath()) for prim in colliders] == ['/Building/CollisionEnvelope']
    assert UsdGeom.Imageable(colliders[0]).GetVisibilityAttr().Get() == 'invisible'
    assert stage.GetRootLayer().GetExternalReferences() == ()
    assert not UsdGeom.Xformable(root).GetOrderedXformOps()


def test_building_classification_excludes_doors_and_thin_walls():
    # Given differently scaled unit cubes at original actor transforms.
    stage = Usd.Stage.CreateInMemory()
    actors = []
    for name, scale in [('Hall', (40, 60, 18)), ('Door', (2, 0.1, 3)), ('Wall', (20, 0.3, 5))]:
        actor = UsdGeom.Xform.Define(stage, '/' + name)
        actor.AddRotateZOp().Set(31)
        actor.AddScaleOp().Set(scale)
        actors.append(actor.GetPrim())
    # When interpreting their physical envelope sizes.
    dimensions = [building_dimensions(actor) for actor in actors]
    # Then only the building receives a facade, without confusing orientation with scale.
    assert dimensions[0] == pytest.approx((40, 60, 18))
    assert dimensions[1:] == [None, None]
