"""Planar occupancy becomes oriented, bounded CAD wires without triangle authority."""

import numpy as np

from usim_env_pipeline.cad_contours import boundary_loops, signed_area, simplify_loop, world_loops


def test_outer_and_inner_wires_retain_exact_occupied_area():
    mask = np.ones((10, 12), dtype=bool)
    mask[2:4, 3:5] = False
    mask[6:9, 7:10] = False
    loops = boundary_loops(mask)
    assert len(loops) == 3
    assert sum(signed_area(loop) for loop in loops) == mask.sum()
    assert sum(signed_area(loop) > 0 for loop in loops) == 1
    assert sum(signed_area(loop) < 0 for loop in loops) == 2


def test_diagonal_contacts_are_not_self_intersecting_wires():
    loops = boundary_loops(np.eye(3, dtype=bool))
    assert len(loops) == 3
    assert all(len(loop) == 4 and signed_area(loop) == 1 for loop in loops)


def test_diagonally_touching_holes_are_separate_simple_cad_wires():
    mask = np.ones((8, 8), dtype=bool)
    mask[3, 3] = mask[4, 4] = False
    loops = boundary_loops(mask)
    holes = [loop for loop in loops if signed_area(loop) < 0]
    assert len(holes) == 2
    assert all(len(loop) == 4 and signed_area(loop) == -1 for loop in holes)
    assert sum(signed_area(loop) for loop in loops) == mask.sum()


def test_concave_outline_and_disconnected_patches_retain_winding():
    mask = np.zeros((8, 8), dtype=bool)
    mask[:5, :2] = True
    mask[:2, :5] = True
    mask[6:, 6:] = True
    loops = boundary_loops(mask)
    assert len(loops) == 2
    assert all(signed_area(loop) > 0 for loop in loops)
    assert sum(signed_area(loop) for loop in loops) == mask.sum()


def test_simplification_removes_bounded_staircase_noise():
    loop = np.array([[0, 0], [2, 0], [2, 1], [2.1, 1], [2.1, 2], [0, 2]], dtype=float)
    simplified = simplify_loop(loop, 0.15)
    assert len(simplified) == 4
    assert signed_area(simplified) > 0


def test_world_wires_use_the_original_plane_coordinate_system():
    archive = {
        'mask': np.ones((2, 3), dtype=bool),
        'lower': np.array([4, -2]),
        'u': np.array([0.0, 1.0, 0.0]),
        'v': np.array([0.0, 0.0, 1.0]),
        'equation': np.array([1.0, 0.0, 0.0, -5.0]),
    }
    loop = world_loops(archive, 0.5, 0.0)[0]
    np.testing.assert_allclose(loop[:, 0], 5.0)
    np.testing.assert_allclose(loop[:, 1:].min(0), [2.0, -1.0])
    np.testing.assert_allclose(loop[:, 1:].max(0), [3.0, 0.5])
