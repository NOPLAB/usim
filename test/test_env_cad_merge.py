"""CAD duplicate removal respects physical separation and actual face coverage."""

import numpy as np
import pytest

pytest.importorskip('OCP')

from usim_env_pipeline.cad_kernel import make_planar_face, surface_area
from usim_env_pipeline.cad_merge import consolidate
from usim_env_pipeline.cad_types import CadBoard, CadConfig


def plate(number, x0, x1, z=0.0):
    points = np.array([[x0, 0, z], [x1, 0, z], [x1, 10, z], [x0, 10, z]], dtype=float)
    return CadBoard(
        make_planar_face(points),
        np.array([0.0, 0.0, 1.0, -z]),
        (number,),
        np.array([number], dtype=np.int64),
    )


def test_contained_parallel_alias_is_removed_but_its_source_support_is_retained():
    big, small = plate(0, 0, 10), plate(1, 3, 5, 0.05)
    result, decisions = consolidate([small, big], CadConfig(), np.array([[0, 0, 0], [3, 0, 0.05]]))
    assert len(result) == 1
    assert surface_area(result[0].shape) == pytest.approx(100)
    assert result[0].source_planes == (0, 1)
    np.testing.assert_array_equal(result[0].support_ids, [0, 1])
    assert decisions[0]['kind'] == 'contained'


def test_distinct_floor_levels_are_not_deleted_as_contained_aliases():
    result, decisions = consolidate(
        [plate(0, 0, 10), plate(1, 3, 5, 3)], CadConfig(), np.array([[0, 0, 0], [3, 0, 3]])
    )
    assert len(result) == 2
    assert not decisions


def test_overlapping_coplanar_faces_become_one_kernel_surface():
    result, decisions = consolidate(
        [plate(0, 0, 10), plate(1, 8, 12)], CadConfig(), np.array([[0, 0, 0], [8, 0, 0]])
    )
    assert len(result) == 1
    assert surface_area(result[0].shape) == pytest.approx(120)
    assert decisions[0]['kind'] == 'fused'


def test_parallel_faces_outside_each_others_footprint_remain_separate():
    result, decisions = consolidate(
        [plate(0, 0, 10), plate(1, 12, 14)], CadConfig(), np.array([[0, 0, 0], [12, 0, 0]])
    )
    assert len(result) == 2
    assert not decisions


def test_omitted_fragments_support_still_bounds_parallel_projection():
    normal = np.array([-0.03, 0, 1])
    normal /= np.linalg.norm(normal)
    corners = np.array([[3, 0, 0.09], [5, 0, 0.15], [5, 2, 0.15], [3, 2, 0.09]])
    small = CadBoard(make_planar_face(corners), np.r_[normal, 0], (1,), np.array([1, 2]))
    source = np.array([[0, 0, 0], [3, 0, 0.09], [10, 0, 0.30]])
    result, decisions = consolidate([plate(0, 0, 10), small], CadConfig(), source)
    assert len(result) == 2
    assert not decisions
    np.testing.assert_array_equal(result[1].support_ids, [1, 2])
