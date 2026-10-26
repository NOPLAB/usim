"""Planar presentation straightens supported surfaces and repairs only bounded gaps."""

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip('scipy')
pytest.importorskip('open3d')

from usim_env_pipeline.planes import (
    Plane,
    PlaneConfig,
    covered_faces,
    extract_planes,
    merge_planes,
    patch_mesh,
    refit,
)


def wall(*, hole=False):
    y, z = np.meshgrid(np.arange(-4, 4.01, 0.1), np.arange(0, 4.01, 0.1))
    x = 0.018 * np.sin(3 * y) * np.cos(4 * z)
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel())).astype(np.float32)
    if hole:
        missing = (np.abs(points[:, 1]) < 0.3) & (np.abs(points[:, 2] - 2) < 0.3)
        points = points[~missing]
    return points


def test_noisy_wall_becomes_a_single_flat_low_triangle_surface():
    # Given noisy observations from one architectural wall.
    points = wall()
    original = points.copy()
    config = PlaneConfig()
    # When extracting and meshing supported planes.
    planes = extract_planes(points, np.arange(len(points)), config)
    patch = patch_mesh(points, planes[0], config)
    # Then the output is planar and compact, while the originals remain exact.
    assert len(planes) == 1
    assert patch is not None
    assert len(patch.faces) < len(points) / 10
    distances = patch.positions @ patch.plane.normal + patch.plane.offset
    assert np.abs(distances).max() < 1e-5
    triangles = patch.positions[patch.faces]
    area = (
        np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
        ).sum()
        / 2
    )
    assert area == pytest.approx(patch.mask.sum() * config.cell**2, rel=1e-5)
    assert np.array_equal(points, original)


def test_same_plane_measurement_gap_is_connected():
    # Given a small enclosed gap surrounded by the same flat facade.
    points = wall(hole=True)
    config = PlaneConfig()
    plane = refit(points, np.arange(len(points)))
    # When repairing the plane's supported footprint.
    patch = patch_mesh(points, plane, config)
    # Then the missing center is connected and explicitly distinguishable from observation.
    assert patch is not None
    center = np.array([0, 0, 2])
    key = np.floor([center @ patch.u / config.cell, center @ patch.v / config.cell]).astype(int)
    key -= patch.lower
    assert patch.mask[tuple(key)]
    assert not patch.observed[tuple(key)]
    assert (patch.mask & ~patch.observed).sum() > 0


def test_recess_returns_protect_opening_from_gap_closing():
    # Given a window-sized recess with observed returns off the facade plane.
    facade = wall(hole=True)
    y, z = np.meshgrid(np.arange(-0.3, 0.31, 0.025), np.arange(1.7, 2.31, 0.025))
    behind = np.column_stack((np.full(y.size, 0.8), y.ravel(), z.ravel())).astype(np.float32)
    points = np.concatenate((facade, behind))
    config = PlaneConfig()
    plane = Plane(np.array([1.0, 0.0, 0.0]), 0.0, np.arange(len(facade)))
    # When planar repair sees recess evidence on either side.
    patch = patch_mesh(points, plane, config)
    # Then the center remains an opening instead of becoming a flat wall.
    assert patch is not None
    key = np.floor(np.array([0, 2]) / config.cell).astype(int) - patch.lower
    assert patch.protected[tuple(key)]
    assert not patch.mask[tuple(key)]


def test_separate_parallel_surfaces_are_not_merged_or_removed():
    # Given two separate walls with identical orientation.
    first = wall()
    second = first.copy()
    second[:, 0] += 1
    points = np.concatenate((first, second))
    config = PlaneConfig()
    planes = [
        refit(points, np.arange(len(first))),
        refit(points, np.arange(len(first), len(points))),
    ]
    # When combining regional fits.
    merged = merge_planes(points, planes, config)
    # Then their geometric offset survives, and one patch cannot remove the other wall.
    assert len(merged) == 2
    patch = patch_mesh(points, merged[0], config)
    assert patch is not None
    triangle = np.array([[len(first), len(first) + 1, len(first) + 81]], dtype=np.int32)
    assert not covered_faces(points, triangle, patch, config).any()


def test_adjacent_regional_fits_share_one_plane_and_support():
    # Given two regional fits from opposite sides of the same noisy wall.
    points = wall()
    ids = np.arange(len(points))
    planes = [refit(points, ids[points[:, 1] < 0]), refit(points, ids[points[:, 1] >= 0])]
    # When unifying coplanar regions.
    merged = merge_planes(points, planes, PlaneConfig())
    # Then a single fitted equation and all source support span the boundary.
    assert len(merged) == 1
    np.testing.assert_array_equal(merged[0].ids, ids)


def test_curved_volume_is_not_replaced_by_a_building_plane():
    # Given a smooth spherical object rather than a facade.
    a, b = np.meshgrid(np.linspace(0, 2 * np.pi, 80), np.linspace(-1.4, 1.4, 40))
    points = np.column_stack(
        (
            (2 * np.cos(a) * np.cos(b)).ravel(),
            (2 * np.sin(a) * np.cos(b)).ravel(),
            (2 * np.sin(b)).ravel(),
        )
    ).astype(np.float32)
    # When requiring substantial, tight planar support.
    planes = extract_planes(
        points, np.arange(len(points)), replace(PlaneConfig(), min_support=500)
    )
    # Then no architectural patch is invented from that curved surface.
    assert planes == []


def test_planar_usd_has_flat_normals_collision_and_explicit_fitted_provenance(tmp_path):
    pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom, UsdPhysics

    from usim_env_pipeline.planar import author_patch
    from usim_env_pipeline.usd import LayerMetadata

    # Given a repaired plane with source support rather than source-identical corners.
    points = wall(hole=True)
    patch = patch_mesh(points, refit(points, np.arange(len(points))), PlaneConfig())
    assert patch is not None
    path = tmp_path / 'plane.usdc'
    metadata = LayerMetadata('CC-BY-4.0', 'fixture', '', 'XYZ', 'fitted plane')
    # When authoring that supported surface as a USD payload.
    author_patch(path, patch, metadata)
    # Then it has flat normals and exact triangle collisions without false original vertex IDs.
    stage = Usd.Stage.Open(str(path))
    mesh = UsdGeom.Mesh.Get(stage, '/Tile/Surface')
    assert mesh.GetNormalsInterpolation() == UsdGeom.Tokens.constant
    assert not mesh.GetPrim().HasAttribute('source:recordIds')
    assert mesh.GetPrim().GetAttribute('plane:supportRecords').Get() == len(patch.plane.ids)
    assert UsdPhysics.CollisionAPI(mesh.GetPrim()).GetCollisionEnabledAttr().Get()
    assert UsdPhysics.MeshCollisionAPI(mesh.GetPrim()).GetApproximationAttr().Get() == 'none'
    np.testing.assert_array_equal(mesh.GetPointsAttr().Get(), patch.positions)


def test_default_optimizer_retains_curved_surface_instead_of_filling_it():
    # Given a densely sampled round object under the production fitting thresholds.
    a, b = np.meshgrid(np.linspace(0, 2 * np.pi, 100), np.linspace(-1.4, 1.4, 60))
    points = np.column_stack(
        (
            (2 * np.cos(a) * np.cos(b)).ravel(),
            (2 * np.sin(a) * np.cos(b)).ravel(),
            (2 * np.sin(b)).ravel(),
        )
    ).astype(np.float32)
    config = PlaneConfig()
    # When extracting candidate patches with the default configuration.
    planes = extract_planes(points, np.arange(len(points)), config)
    patches = [patch_mesh(points, plane, config) for plane in planes]
    # Then planar subsets of the curved shell do not become flat replacement patches.
    assert all(patch is None for patch in patches)


def floor_ring():
    x, y = np.meshgrid(np.arange(-5, 5.01, 0.1), np.arange(-5, 5.01, 0.1))
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))).astype(np.float32)
    return points[~((np.abs(points[:, 0]) < 3) & (np.abs(points[:, 1]) < 3))]


def mask_at(patch, point, config):
    cells = np.floor([point @ patch.u / config.cell, point @ patch.v / config.cell]).astype(int)
    return tuple(cells - patch.lower)


def test_large_enclosed_floor_hole_is_completed_under_an_above_floor_obstacle():
    # Given a six-meter enclosed measurement gap and returns above the floor.
    floor = floor_ring()
    x, y = np.meshgrid(np.arange(-0.5, 0.51, 0.025), np.arange(-0.5, 0.51, 0.025))
    obstacle = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 0.8))).astype(np.float32)
    points = np.concatenate((floor, obstacle))
    plane = Plane(np.array([0.0, 0.0, 1.0]), 0.0, np.arange(len(floor)))
    config = PlaneConfig()
    # When completing the enclosed plane interior.
    patch = patch_mesh(points, plane, config)
    # Then the center becomes an explicitly inferred floor rather than an opening.
    assert patch is not None
    key = mask_at(patch, np.zeros(3), config)
    assert patch.mask[key]
    assert patch.enclosed[key]
    assert not patch.observed[key]


def test_floor_opening_below_plane_stays_open_without_blocking_other_completion():
    # Given a large floor gap containing a smaller observed downward opening.
    floor = floor_ring()
    x, y = np.meshgrid(np.arange(-0.4, 0.41, 0.025), np.arange(-0.4, 0.41, 0.025))
    below = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, -0.8))).astype(np.float32)
    points = np.concatenate((floor, below))
    plane = Plane(np.array([0.0, 0.0, 1.0]), 0.0, np.arange(len(floor)))
    config = PlaneConfig()
    # When repairing the enclosing floor.
    patch = patch_mesh(points, plane, config)
    # Then the actual depth-supported opening remains, while unrelated missing floor fills.
    assert patch is not None
    assert not patch.mask[mask_at(patch, np.zeros(3), config)]
    assert patch.mask[mask_at(patch, np.array([2, 0, 0]), config)]


def test_open_boundary_is_not_extrapolated_as_an_enclosed_floor():
    # Given a missing interior connected to the exterior through a wide break.
    points = floor_ring()
    points = points[~((np.abs(points[:, 0]) < 0.8) & (points[:, 1] > 0))]
    config = PlaneConfig()
    # When attempting enclosed-hole completion.
    patch = patch_mesh(points, refit(points, np.arange(len(points))), config)
    # Then the unbounded gap stays empty rather than being filled by a hull.
    assert patch is not None
    assert not patch.mask[mask_at(patch, np.zeros(3), config)]
