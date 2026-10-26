"""CAD boolean boundaries retain real notches without invalid touching inner wires."""

import numpy as np
import pytest

pytest.importorskip('OCP')

from usim_env_pipeline.cad_kernel import make_planar_faces, shape_stats, surface_area


def test_large_void_touching_the_outer_boundary_is_a_valid_cad_notch():
    # Given a plate with a four-square-meter void touching its left boundary.
    outer = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]], dtype=float)
    hole = np.array([[0, 5, 0], [0, 7, 0], [2, 7, 0], [2, 5, 0]], dtype=float)
    # When constructing the CAD face with the one-square-meter tiny-hole threshold.
    result = make_planar_faces([outer, hole], np.array([0.0, 0.0, 1.0]), hole_threshold_m2=1)
    # Then the exact kernel boundary is a valid notch, not an invalid inner wire.
    assert len(result) == 1
    assert surface_area(result[0]) == pytest.approx(96)
    assert shape_stats(result[0])['hole_count'] == 0
