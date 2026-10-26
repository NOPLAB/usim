"""Straight polygon slabs remain source-bounded, open where measured, and closed."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

pytest.importorskip('shapely')

from shapely import union_all
from shapely.geometry import MultiPolygon, Point, Polygon, box

from usim_env_pipeline.models import PipelineInputError
from usim_env_pipeline.simulation_polygon_slab import polygon_slab


def footprint(slab):
    top = slab.positions[slab.faces]
    return union_all([Polygon(face[:, :2]) for face in top if (face[:, 2] > 0).all()])


def test_many_observed_holes_keep_every_original_hole_center():
    # Given a planar support containing one hundred separate measured openings.
    holes = [(x, y) for x in range(2, 50, 5) for y in range(2, 50, 5)]
    removed = set(holes)
    keys = np.array(
        [[x, y] for x in range(50) for y in range(50) if (x, y) not in removed], np.int64
    )
    # When simplifying all rings together as bounded contour candidates.
    slab = polygon_slab(keys, 0.1, 0.16)
    result = footprint(slab)
    # Then no opening silently becomes solid during the batched simplification.
    assert isinstance(result, Polygon)
    assert len(result.interiors) == len(holes)
    assert all(not result.covers(Point((x + 0.5) * 0.1, (y + 0.5) * 0.1)) for x, y in holes)


def assert_closed(slab, thickness):
    faces = slab.faces
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert (counts == 2).all()
    triangles = slab.positions[faces] - slab.positions.mean(axis=0)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    assert (np.linalg.norm(normals, axis=1) > 0).all()
    volume = np.einsum('ij,ij->i', triangles[:, 0], normals).sum() / 6
    assert volume > 0
    np.testing.assert_allclose(volume, slab.area_m2 * thickness, rtol=1e-12)
    np.testing.assert_array_equal(np.unique(slab.positions[:, 2]), [-thickness / 2, thickness / 2])
    # Opposite orientation on each shared edge proves consistent outward winding.
    directed = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    np.testing.assert_array_equal(
        np.unique(directed, axis=0), np.unique(directed[:, ::-1], axis=0)
    )


def test_jagged_near_straight_outline_becomes_compact_straight_segments():
    keys = np.array([[x, y] for x in range(40) for y in range(10 + x % 2)], np.int64)
    original = union_all([box(x, y, x + 1, y + 1) for x, y in keys])
    slab = polygon_slab(keys, 1.0, 0.2)
    result = footprint(slab)
    assert isinstance(original, Polygon)
    assert isinstance(result, Polygon)
    assert len(result.exterior.coords) <= 8
    assert len(result.exterior.coords) < len(original.exterior.coords) / 10
    assert len(slab.faces) <= 24
    assert 0 < slab.outline_error_m <= 1.5
    assert original.boundary.covered_by(result.boundary.buffer(slab.outline_error_m))
    assert result.boundary.covered_by(original.boundary.buffer(slab.outline_error_m))
    assert_closed(slab, 0.2)


def test_holes_and_separate_components_remain_empty():
    keys = np.array(
        [[x, y] for x in range(12) for y in range(12) if not (4 <= x < 8 and 4 <= y < 8)]
        + [[x, y] for x in range(18, 22) for y in range(4)],
        np.int64,
    )
    slab = polygon_slab(keys, 0.1, 0.16)
    result = footprint(slab)
    assert isinstance(result, MultiPolygon)
    assert len(result.geoms) == 2
    assert sum(len(polygon.interiors) for polygon in result.geoms) == 1
    assert not result.covers(Point(0.6, 0.6))
    assert not result.intersects(box(1.3, 0.05, 1.7, 0.35))
    np.testing.assert_allclose(slab.area_m2, len(keys) * 0.1**2)
    assert_closed(slab, 0.16)


def test_tiny_hole_next_to_jagged_outline_is_not_closed_by_simplification():
    keys = np.array(
        [[x, y] for x in range(20) for y in range(5 + x % 2) if (x, y) != (11, 3)],
        np.int64,
    )
    slab = polygon_slab(keys, 1, 0.3)
    result = footprint(slab)
    assert isinstance(result, Polygon)
    assert len(result.interiors) == 1
    assert not result.covers(Point(11.5, 3.5))
    assert slab.outline_error_m <= 1.5
    assert_closed(slab, 0.3)


def test_translation_duplicates_order_and_readonly_inputs_are_deterministic():
    keys = np.array([[x, y] for x in range(20) for y in range(6 + x % 2)], np.int64)
    keys.setflags(write=False)
    before = keys.copy()
    slab = polygon_slab(keys, 0.1, 0.16)
    shifted = polygon_slab(np.concatenate((keys[::-1], keys[:5])) + [-200, 700], 0.1, 0.16)
    np.testing.assert_allclose(shifted.positions, slab.positions + [-20, 70, 0])
    np.testing.assert_array_equal(shifted.faces, slab.faces)
    np.testing.assert_array_equal(keys, before)
    assert slab.area_m2 == shifted.area_m2
    assert slab.outline_error_m == shifted.outline_error_m
    assert slab.positions.dtype == np.float64 and slab.faces.dtype == np.int32
    assert not slab.positions.flags.writeable and not slab.faces.flags.writeable
    with pytest.raises(FrozenInstanceError):
        setattr(slab, 'area_m2', 0)
    assert_closed(shifted, 0.16)


def test_corner_touching_cells_are_individually_closed_without_filling_gap():
    slab = polygon_slab(np.array([[0, 0], [1, 1]], np.int64), 1, 0.2)
    assert slab.area_m2 == 2
    assert not footprint(slab).covers(Point(0.5, 1.5))
    assert_closed(slab, 0.2)


def test_empty_input_returns_empty_mesh():
    slab = polygon_slab(np.empty((0, 2), np.int64), 0.1, 0.2)
    assert slab.positions.shape == slab.faces.shape == (0, 3)
    assert slab.area_m2 == slab.outline_error_m == 0


@pytest.mark.parametrize(
    ('keys', 'cell', 'thickness'),
    [
        (np.array([[0, 0]], np.int32), 1, 1),
        (np.array([0, 0], np.int64), 1, 1),
        (np.array([[0, 0]], np.int64), 0, 1),
        (np.array([[0, 0]], np.int64), float('nan'), 1),
        (np.array([[0, 0]], np.int64), 1, -1),
        (np.array([[0, 0]], np.int64), 1, float('inf')),
    ],
)
def test_invalid_inputs_raise_pipeline_input_error(keys, cell, thickness):
    with pytest.raises(PipelineInputError):
        polygon_slab(keys, cell, thickness)
