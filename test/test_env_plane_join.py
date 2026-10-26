"""Gaussian board compatibility produces deterministic, source-bounded connected planes."""

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip('scipy')

from usim_env_pipeline.plane_join import gaussian_cost, join_planes, link_costs
from usim_env_pipeline.planes import PlaneConfig, patch_mesh, refit


def boards(offset=0.0):
    y, z = np.meshgrid(np.arange(-5, -0.99, 0.2), np.arange(0, 4.01, 0.2))
    first = np.column_stack((np.zeros(y.size), y.ravel(), z.ravel())).astype(np.float32)
    second = first.copy()
    second[:, 1] += 6
    second[:, 0] += offset
    points = np.concatenate((first, second))
    planes = [
        refit(points, np.arange(len(first))),
        refit(points, np.arange(len(first), len(points))),
    ]
    return points, planes


def test_gaussian_distance_cost_matches_normal_negative_log_likelihood():
    # Given normal distances of zero, one sigma and two sigma.
    config = PlaneConfig()
    distances = np.array([[0, 0], [config.distance_sigma, 0], [2 * config.distance_sigma, 0]])
    # When computing the normal-distribution distance cost.
    cost = gaussian_cost(distances, np.zeros(3), config)
    # Then the relative negative log likelihood increases quadratically.
    np.testing.assert_allclose(cost, [0, 0.5, 2])


def test_gaussian_join_connects_two_coplanar_boards_across_missing_measurements():
    # Given two coplanar boards separated by a two-meter measurement gap.
    points, planes = boards()
    config = PlaneConfig(workers=4)
    # When calculating per-board Gaussian costs and joining compatible boundaries.
    joined = join_planes(points, planes, config)
    patch = patch_mesh(points, joined.planes[0], config)
    # Then one common plane and explicit Gaussian bridges connect the missing interval.
    assert len(joined.planes) == 1
    assert joined.accepted.sum() == 1
    assert patch is not None
    center = np.array([0, 0, 2])
    key = np.floor([center @ patch.u, center @ patch.v] / np.array(config.cell)).astype(int)
    key = tuple(key - patch.lower)
    assert patch.mask[key]
    assert patch.bridged[key]
    assert not patch.observed[key]
    plane = joined.planes[0]
    costs = link_costs(points, plane.links, joined.equations[plane.link_models], config)
    assert len(costs) > 0
    assert costs.max() <= config.join_cost


def test_separate_levels_are_not_flattened_by_gaussian_boundary_proximity():
    # Given neighboring parallel boards on distinctly different levels.
    points, planes = boards(offset=0.3)
    # When computing the same Gaussian join objective.
    joined = join_planes(points, planes, PlaneConfig(workers=4))
    # Then spatial proximity cannot override the incompatible normal-plane distance.
    assert len(joined.planes) == 2
    assert not joined.accepted.any()


def test_parallel_and_single_worker_join_results_are_identical():
    # Given the same measured boards and fitting/cost parameters.
    points, planes = boards()
    config = PlaneConfig(workers=4)
    # When evaluating independent board costs with one worker and several workers.
    parallel = join_planes(points, planes, config)
    serial = join_planes(points, planes, replace(config, workers=1))
    # Then scheduling cannot alter the cost, acceptance or connected geometry.
    np.testing.assert_array_equal(parallel.edges, serial.edges)
    np.testing.assert_array_equal(parallel.accepted, serial.accepted)
    np.testing.assert_array_equal(parallel.planes[0].ids, serial.planes[0].ids)
    np.testing.assert_array_equal(parallel.planes[0].links, serial.planes[0].links)
