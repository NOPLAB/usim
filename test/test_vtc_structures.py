"""Measured vertical structure proxies must not duplicate trees or float above ground."""

import numpy as np
import pytest

pytest.importorskip('scipy')

from usim_env_vtc.structures import Post, author_posts, infer_posts
from usim_env_vtc.vegetation import Tree, relative_heights


def column_points(center):
    angle, height = np.meshgrid(
        np.linspace(0, 2 * np.pi, 40, endpoint=False),
        np.linspace(0, 7, 60),
    )
    column = np.column_stack(
        (
            (center + 0.2 * np.cos(angle)).ravel(),
            (0.2 * np.sin(angle)).ravel(),
            height.ravel(),
        )
    )
    x, y = np.meshgrid(np.arange(-5, 15, 0.2), np.arange(-5, 6, 0.2))
    ground = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    return np.concatenate((ground, column)).astype(np.float64)


def test_narrow_multilevel_support_creates_a_post_without_modifying_points():
    # Given a measured vertical column with surrounding ground.
    points = column_points(8)
    original = points.copy()
    # When interpreting nonvegetation structural support.
    posts = infer_posts(points, [], relative_heights(points))
    # Then one supported column retains measured center and height.
    assert len(posts) == 1
    assert posts[0].x == pytest.approx(8, abs=0.1)
    assert 6 < posts[0].height < 8
    assert 0.1 < posts[0].radius < 0.3
    assert np.array_equal(points, original)


def test_tree_trunk_is_not_added_as_a_duplicate_structure():
    # Given narrow support inside a known canopy footprint.
    points = column_points(0)
    trees = [Tree(0, 0, 10, 3, 2000)]
    # When locating additional structural proxies.
    posts = infer_posts(points, trees, relative_heights(points))
    # Then the already represented tree trunk adds no pole.
    assert posts == []


def test_post_payload_preserves_grounding_and_collision_at_negative_coordinates(tmp_path):
    pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom, UsdPhysics

    # Given a supported post over terrain at Z=2.5 m.
    world = Usd.Stage.CreateNew(str(tmp_path / 'world.usda'))
    world.SetDefaultPrim(UsdGeom.Xform.Define(world, '/World').GetPrim())
    posts = [(Post(-45, 40, 7, 0.15, 200), 2.5)]
    # When authoring streamed geometry and reopening the composed scene.
    entries = author_posts(world, posts, tmp_path)
    world.GetRootLayer().Save()
    actual = Usd.Stage.Open(str(tmp_path / 'world.usda'))
    prim = actual.GetPrimAtPath('/World/InferredStructures/posts_m2_1/Post_0')
    bounds = (
        UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_],
        )
        .ComputeWorldBound(prim)
        .ComputeAlignedRange()
    )
    # Then the post is grounded, collidable and conservatively indexed.
    assert bounds.GetMin()[2] == pytest.approx(2.5)
    assert bounds.GetMax()[2] == pytest.approx(9.5)
    assert prim.HasAPI(UsdPhysics.CollisionAPI)
    assert entries[0]['min'][2] == pytest.approx(2.5)
    assert entries[0]['max'][2] == pytest.approx(9.5)
