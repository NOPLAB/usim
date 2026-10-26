"""Measured wall fragments become bounded solids without merging streets or ground."""

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip('scipy')

from usim_env_pipeline.building_fit import facade_components, fit_cuboids, merge_facades
from usim_env_pipeline.building_types import BuildingConfig
from usim_env_pipeline.models import PipelineInputError


def wall(y0: float = 0, y1: float = 8, *, hole=False):
    y, z = np.meshgrid(np.arange(y0, y1 + 0.01, 0.2), np.arange(0, 6.01, 0.2))
    points = np.column_stack((np.zeros(y.size), y.ravel(), z.ravel()))
    if hole:
        points = points[
            ~((points[:, 1] > 2) & (points[:, 1] < 5) & (points[:, 2] > 2) & (points[:, 2] < 4))
        ]
    return points


def components(points, number=0, ids=None, equation=None):
    return facade_components(
        number,
        np.array([1.0, 0, 0, 0]) if equation is None else equation,
        np.arange(len(points)) if ids is None else ids,
        points,
        BuildingConfig(),
    )


def test_wall_with_measurement_holes_becomes_one_bounded_cuboid():
    points = wall(hole=True)
    original = points.copy()
    facades = components(points)
    models = fit_cuboids(facades, points, BuildingConfig())
    assert len(facades) == len(models) == 1
    assert models[0].kind == 'facade'
    np.testing.assert_allclose(sorted(models[0].size), [0.3, 6, 8])
    np.testing.assert_array_equal(facades[0].support_ids, np.arange(len(points)))
    np.testing.assert_array_equal(points, original)


def test_ground_and_short_props_are_not_building_facades():
    points = wall()
    assert not components(points, equation=np.array([0.0, 0, 1, 0]))
    assert not components(points[points[:, 2] <= 1])


def test_coplanar_buildings_separated_by_a_street_remain_separate():
    points = np.concatenate((wall(0, 4), wall(8, 12)))
    facades = merge_facades(components(points), points, BuildingConfig())
    models = fit_cuboids(facades, points, BuildingConfig())
    assert len(facades) == len(models) == 2
    assert all(model.kind == 'facade' for model in models)
    assert max(model.size[0] for model in models) == pytest.approx(4)


def test_nearby_coplanar_fragments_share_a_model_and_keep_both_sources():
    first, second = wall(0, 4), wall(4.5, 8.5)
    points = np.concatenate((first, second))
    facades = components(points, 4, np.arange(len(first))) + components(
        points, 7, np.arange(len(first), len(points))
    )
    merged = merge_facades(facades, points, BuildingConfig())
    assert len(merged) == 1
    assert merged[0].source_planes == (4, 7)
    np.testing.assert_array_equal(merged[0].support_ids, np.arange(len(points)))


@pytest.mark.parametrize('angle', [0.0, 0.63])
def test_perpendicular_observed_corner_supports_rotated_building_envelope(angle):
    first = wall()
    second = wall(0, 10)[:, [1, 0, 2]]
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    origin = np.array([20.0, -7.0, 2.0])
    points = np.concatenate((first, second)) @ rotation.T + origin
    facades = []
    for number, normal, ids in (
        (0, rotation[:, 0], np.arange(len(first))),
        (1, rotation[:, 1], np.arange(len(first), len(points))),
    ):
        facades += components(points, number, ids, np.r_[normal, -normal @ origin])
    models = fit_cuboids(facades, points, BuildingConfig())
    assert len(models) == 1
    assert models[0].kind == 'building'
    np.testing.assert_allclose(models[0].center, rotation @ [5, 4, 3] + origin, atol=1e-8)
    assert np.prod(models[0].size) == pytest.approx(480)
    assert models[0].facade_indices == (0, 1)
    np.testing.assert_allclose(models[0].axes.T @ models[0].axes, np.eye(3), atol=1e-12)
    assert np.linalg.det(models[0].axes) == pytest.approx(1)


def test_perpendicular_wall_at_an_interior_junction_does_not_enclose_a_street():
    first, second = wall(-4, 4), wall(0, 10)[:, [1, 0, 2]]
    points = np.concatenate((first, second))
    facades = components(points, 0, np.arange(len(first))) + components(
        points, 1, np.arange(len(first), len(points)), np.array([0.0, 1, 0, 0])
    )
    assert all(model.kind == 'facade' for model in fit_cuboids(facades, points, BuildingConfig()))


@pytest.mark.parametrize(
    'changes',
    [
        {'cell': 0},
        {'gap': -1},
        {'wall_thickness': np.nan},
        {'minimum_points': 0},
        {'minimum_coverage': 1.1},
        {'corner_angle': 90},
    ],
)
def test_invalid_geometry_configuration_fails_at_boundary(changes):
    with pytest.raises(PipelineInputError):
        replace(BuildingConfig(), **changes)
