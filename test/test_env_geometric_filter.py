"""Deterministic local surface evidence and original-record provenance."""

from dataclasses import asdict, replace
import json
import math

import numpy as np
from numpy.typing import NDArray
import pytest

from usim_env_pipeline.geometric_filter import (
    DEFAULT_GEOMETRY,
    GeometricConfig,
    GeometricResult,
    GeometricSelection,
    geometric_outliers,
)
from usim_env_pipeline.models import PipelineInputError

pytest.importorskip('scipy')


def plane(spacing: float = 0.12, slope: float = 0.0) -> NDArray[np.float64]:
    x, y = np.meshgrid(np.arange(-5, 6) * spacing, np.arange(-5, 6) * spacing)
    return np.column_stack((x.ravel(), y.ravel(), slope * x.ravel()))


def evaluate(
    points: NDArray[np.float64],
    candidates: NDArray[np.bool_] | None = None,
    config: GeometricConfig = DEFAULT_GEOMETRY,
) -> GeometricResult:
    keep = np.ones(len(points), dtype=np.bool_)
    if candidates is None:
        candidates = keep.copy()
    return geometric_outliers(points, GeometricSelection(keep, candidates), config)


@pytest.mark.parametrize('slope', [0.0, 0.7, -1.2])
def test_sparse_planes_are_preserved(slope: float):
    points = plane(spacing=0.22, slope=slope)
    result = evaluate(points)
    assert result.keep.all()
    assert not result.geometry_rejected.any()
    assert (result.consistent_scales > 0).all()


def test_sparse_line_preserved():
    t = np.linspace(-2, 2, 81)
    points = np.column_stack((t, 0.4 * t, -0.8 * t))
    result = evaluate(points)
    assert result.keep.all()
    assert (result.consistent_scales > 0).all()


@pytest.mark.parametrize('surface', ['plane', 'line'])
def test_off_surface_candidate_rejected_without_fitting_itself(surface: str):
    if surface == 'plane':
        support = plane()
        query = [0.013, -0.019, 0.21]
    else:
        t = np.linspace(-1.5, 1.5, 121)
        support = np.column_stack((t, np.zeros_like(t), np.zeros_like(t)))
        query = [0.013, 0.21, 0]
    points = np.vstack((support, query))
    candidate = np.zeros(len(points), dtype=np.bool_)
    candidate[-1] = True
    result = evaluate(points, candidate)
    assert result.keep[:-1].all()
    assert result.supported_scales[-1] >= 2
    assert result.consistent_scales[-1] == 0
    assert result.geometry_rejected[-1]
    assert not result.keep[-1]
    assert not result.evaluated[:-1].any()


def test_one_confident_scale_is_insufficient_to_reject():
    points = np.vstack((plane(), [0.013, -0.019, 0.21]))
    candidate = np.zeros(len(points), dtype=np.bool_)
    candidate[-1] = True
    config = replace(DEFAULT_GEOMETRY, scales=((12, 0.1), (32, 0.15), (64, 3.0)))
    result = evaluate(points, candidate, config)
    assert result.supported_scales[-1] == 1
    assert result.consistent_scales[-1] == 0
    assert result.keep.all()


def test_consistent_local_scale_vetoes_two_contradictions():
    # A small raised flat patch surrounded by a larger ground surface.
    angle = np.arange(12) * (2 * math.pi / 12)
    local = np.column_stack((0.12 * np.cos(angle), 0.12 * np.sin(angle), np.full(12, 0.25)))
    ground = plane(spacing=0.16)
    ground = ground[np.linalg.norm(ground[:, :2], axis=1) > 0.35]
    ground[:, 2] = 0.15
    points = np.vstack((local, ground, [0, 0, 0.25]))
    candidate = np.zeros(len(points), dtype=np.bool_)
    candidate[-1] = True
    result = evaluate(points, candidate)
    assert result.consistent_scales[-1] >= 1
    assert result.supported_scales[-1] - result.consistent_scales[-1] == 2
    assert result.keep.all()


def test_insufficient_and_empty_neighborhoods_preserved():
    for size in (0, 1, 5, 8):
        points = np.column_stack(
            (np.arange(size, dtype=np.float64), np.zeros(size), np.zeros(size))
        )
        result = evaluate(points)
        assert result.keep.all()
        assert not result.supported_scales.any()


def test_nonplanar_curved_neighborhood_preserved():
    # A deterministic spherical canopy: support surrounds the query in three dimensions.
    i = np.arange(96)
    z = 1 - 2 * (i + 0.5) / len(i)
    phi: NDArray[np.float64] = i * math.pi * (3 - math.sqrt(5))
    r = np.sqrt(1 - z**2)
    support = 0.25 * np.column_stack((r * np.cos(phi), r * np.sin(phi), z))
    points = np.vstack((support, [0, 0, 0]))
    candidate = np.zeros(len(points), dtype=np.bool_)
    candidate[-1] = True
    result = evaluate(points, candidate)
    assert result.keep.all()
    assert result.supported_scales[-1] == 0
    assert evaluate(support).keep.all()


def test_original_order_immutable_support_and_batch_independence():
    points = np.vstack((plane(), [0.013, -0.019, 0.21], [0.2, 0.2, 2]))
    permutation: NDArray[np.int64] = np.random.default_rng(481).permutation(len(points))
    points = points[permutation]
    keep = np.not_equal(permutation, len(points) - 1)
    candidate = np.logical_or(
        np.equal(permutation, len(points) - 2), np.equal(permutation, len(points) - 1)
    )
    candidate = np.logical_or(candidate, np.equal(np.remainder(permutation, 5), 0))
    original = (points.copy(), keep.copy(), candidate.copy())
    for value in (points, keep, candidate):
        value.setflags(write=False)
    selection = GeometricSelection(keep, candidate)
    small = geometric_outliers(points, selection, replace(DEFAULT_GEOMETRY, batch_size=1))
    large = geometric_outliers(points, selection, replace(DEFAULT_GEOMETRY, batch_size=256))
    for first, second in (
        (small.keep, large.keep),
        (small.geometry_rejected, large.geometry_rejected),
        (small.supported_scales, large.supported_scales),
        (small.consistent_scales, large.consistent_scales),
        (small.evaluated, large.evaluated),
    ):
        np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(small.geometry_rejected, np.equal(permutation, len(points) - 2))
    np.testing.assert_array_equal(small.evaluated, candidate & keep)
    for value, unchanged in zip((points, keep, candidate), original):
        np.testing.assert_array_equal(value, unchanged)
    # An already rejected candidate remains support for its neighboring query.
    two = np.vstack((plane(), [0.013, -0.019, 0.21], [0.02, -0.03, 0.22]))
    both = np.zeros(len(two), dtype=np.bool_)
    both[-2:] = True
    together = evaluate(two, both, replace(DEFAULT_GEOMETRY, batch_size=1))
    for index in (-2, -1):
        only = np.zeros(len(two), dtype=np.bool_)
        only[index] = True
        separate = evaluate(two, only)
        assert together.supported_scales[index] == separate.supported_scales[index]
        assert together.consistent_scales[index] == separate.consistent_scales[index]
        assert together.keep[index] == separate.keep[index]


def test_coincident_observations_do_not_imply_confidence():
    points = np.vstack((np.zeros((100, 3)), [0, 0, 0.2]))
    result = evaluate(points)
    assert result.keep.all()
    assert not result.supported_scales.any()
    # Two distinct locations remain underdetermined despite many duplicate records.
    points[:50, 0] = 0.1
    result = evaluate(points)
    assert result.keep.all()
    assert not result.supported_scales.any()


@pytest.mark.parametrize(
    'config',
    [
        replace(DEFAULT_GEOMETRY),
        GeometricConfig(scales=((8, 0.5), (24, 2.0)), workers=1, batch_size=7),
    ],
)
def test_parameters_are_json_reproducible(config: GeometricConfig):
    encoded = json.dumps(asdict(config), allow_nan=False)
    assert json.dumps(json.loads(encoded)) == encoded
    assert evaluate(plane(), config=config).config == config


@pytest.mark.parametrize(
    'field,value',
    [
        ('workers', 0),
        ('batch_size', 1.5),
        ('min_neighbors', 5),
        ('min_contradicted_scales', 1),
        ('residual_floor_m', np.nan),
        ('residual_ceiling_m', np.inf),
        ('trim_fraction', 1),
        ('min_spread_m', 0),
        ('line_max_ratio', 0.2),
        ('scales', ((12, np.nan),)),
        ('scales', ((12, -1),)),
        ('scales', ((12, 0.4), (12, 0.4))),
    ],
)
def test_invalid_parameters_rejected(field: str, value: float | tuple[tuple[int, float], ...]):
    with pytest.raises(PipelineInputError):
        _ = replace(DEFAULT_GEOMETRY, **{field: value})


@pytest.mark.parametrize('points', [np.zeros((3, 2)), np.array([[0, np.inf, 0]])])
def test_invalid_coordinates_rejected(points: NDArray[np.float64]):
    with pytest.raises(PipelineInputError):
        _ = evaluate(points)


@pytest.mark.parametrize('bad_mask', [np.ones(2, dtype=bool), np.ones((3, 1), dtype=bool)])
def test_invalid_selection_shape_rejected(bad_mask: NDArray[np.bool_]):
    points = np.zeros((3, 3))
    for selection in (
        GeometricSelection(bad_mask, np.ones(3, dtype=bool)),
        GeometricSelection(np.ones(3, dtype=bool), bad_mask),
    ):
        with pytest.raises(PipelineInputError):
            _ = geometric_outliers(points, selection)


def test_invalid_selection_dtype_rejected():
    points = np.zeros((3, 3))
    selection = GeometricSelection(np.ones(3, dtype=bool), np.ones(3, dtype=bool))
    # Deliberately malformed runtime input, without weakening the typed domain API.
    object.__setattr__(selection, 'sor_keep', np.ones(3, dtype=np.uint8))
    with pytest.raises(PipelineInputError):
        _ = geometric_outliers(points, selection)
