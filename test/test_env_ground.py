"""Ground repairs retain measured vertices, holes, roof separation and tile seams."""

import numpy as np
import pytest

pytest.importorskip('scipy')

from usim_env_pipeline.ground import author_ground, reconstruct_ground
from usim_env_pipeline.models import Support
from usim_env_pipeline.usd import LayerMetadata, stage_base


def sloped_ground():
    x, y = np.meshgrid(np.arange(0, 3.01, 0.1), np.arange(0, 3.01, 0.1))
    z = 0.05 * x + 0.005 * np.sin(7 * x) * np.cos(5 * y)
    return np.column_stack((x.ravel(), y.ravel(), z.ravel())).astype(np.float32)


def test_noisy_ground_retains_exact_source_vertices_and_upward_coverage():
    # Given measured sloped ground without usable source normals.
    points = sloped_ground()
    original = points.copy()
    # When triangulating the ground separately from BPA.
    surface = reconstruct_ground(points, np.zeros(len(points)), Support())
    # Then winding is upward, source positions are exact, and coverage is substantial.
    triangles = points[surface.ids][surface.faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    assert (cross[:, 2] > 0).all()
    assert cross[:, 2].sum() / 2 > 8
    assert np.array_equal(points, original)
    assert len(np.unique(surface.ids)) == len(surface.ids)


def test_unsupported_ground_hole_is_not_bridged():
    # Given a meter-wide region with no ground returns.
    points = sloped_ground()
    hole = ((points[:, :2] > 0.9) & (points[:, :2] < 2.1)).all(axis=1)
    points = points[~hole]
    # When reconstructing the measured lower envelope.
    surface = reconstruct_ground(points, np.zeros(len(points)), Support())
    # Then no triangle appears inside the unsupported gap.
    triangles = points[surface.ids][surface.faces]
    centroids = triangles.mean(axis=1)
    assert len(triangles) > 0
    # Boundary corners within the explicit support distance are supported;
    # the interior farther from every source return must stay open.
    unsupported = ((centroids[:, :2] > 1.1) & (centroids[:, :2] < 1.9)).all(axis=1)
    assert not unsupported.any()
    edges = triangles[:, [1, 2, 0]] - triangles
    assert np.linalg.norm(edges, axis=2).max() <= Support().max_edge


def test_detached_roof_only_region_is_not_classified_as_ground():
    # Given an isolated roof whose locally relative height is misleadingly zero.
    floor = sloped_ground()
    roof = floor[::4].copy()
    roof[:, 0] += 12
    roof[:, 2] += 10
    points = np.concatenate((floor, roof))
    # When selecting the connected measured lower envelope.
    surface = reconstruct_ground(points, np.zeros(len(points)), Support())
    # Then none of the isolated elevated floor's source IDs enters the ground.
    assert len(surface.faces) > 0
    assert (surface.ids < len(floor)).all()


def test_ground_payloads_preserve_exact_source_ids_collision_and_tile_seams(tmp_path):
    pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom, UsdPhysics

    # Given one triangulation crossing several half-open meter-wide tile cores.
    points = sloped_ground()
    surface = reconstruct_ground(points, np.zeros(len(points)), Support())
    metadata = LayerMetadata('CC-BY-4.0', 'fixture', '', 'XYZ', 'measured ground')
    world = stage_base(tmp_path / 'authored.usda', '/World', metadata)
    # When partitioning source triangles without independently remeshing each tile.
    entries = author_ground(world, tmp_path, points, surface, 1.0, metadata)
    world.GetRootLayer().Save()
    stage = Usd.Stage.Open(str(tmp_path / 'authored.usda'))
    # Then every face occurs once and shared seam coordinates remain identical.
    actual = []
    all_ids = []
    for entry in entries:
        mesh = UsdGeom.Mesh.Get(stage, entry['path'] + '/Surface')
        ids = np.asarray(mesh.GetPrim().GetAttribute('source:recordIds').Get())
        vertices = np.asarray(mesh.GetPointsAttr().Get())
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        np.testing.assert_array_equal(vertices, points[ids])
        assert UsdPhysics.CollisionAPI(mesh.GetPrim()).GetCollisionEnabledAttr().Get()
        assert UsdPhysics.MeshCollisionAPI(mesh.GetPrim()).GetApproximationAttr().Get() == 'none'
        actual.extend(map(tuple, ids[faces]))
        all_ids.extend(ids)
    assert set(actual) == set(map(tuple, surface.ids[surface.faces]))
    assert len(actual) == len(surface.faces)
    assert len(all_ids) > len(set(all_ids))
