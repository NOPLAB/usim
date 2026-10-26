"""Finite original-point member hypotheses are not physical face-edge claims."""

from dataclasses import replace

import numpy as np
from scipy.spatial import KDTree

from usim_env_pipeline.window_contour_features import contour_features
from usim_env_pipeline.window_member_axes import (
    MemberAxis,
    member_axis_proposals,
)
from usim_env_pipeline.window_axis_selection import (
    competing,
    direction_field_coverage,
    original_orientation_consistency,
    select_member_axes,
    split_member_sections,
)
from usim_env_pipeline.window_member_sections import member_section_centres, section_evidence


def test_full_sheet_is_not_a_thin_member_section() -> None:
    y, z = np.meshgrid(np.arange(-1, 1.01, 0.04), np.arange(0, 3.01, 0.04))
    xyz = np.column_stack((np.zeros(y.size), y.ravel(), z.ravel()))
    axis = np.array([[0, 0, 0.2], [0, 0, 2.8]], dtype=np.float64)
    thin, width = section_evidence(xyz, axis, np.array([1.0, 0, 0]))
    assert thin == 0
    assert width > 0.6


def test_bounded_strip_has_original_support_and_no_large_gap_bridge() -> None:
    y, z = np.meshgrid(np.arange(-0.08, 0.081, 0.04), np.arange(0, 3.01, 0.04))
    xyz = np.column_stack((np.zeros(y.size), y.ravel(), z.ravel()))
    xyz = xyz[(xyz[:, 2] < 1.1) | (xyz[:, 2] > 1.7)]
    before = xyz.copy()
    features = contour_features(xyz, np.arange(len(xyz), dtype=np.int64), (0.15, 0.3, 0.5))
    candidates = member_axis_proposals(xyz, features)
    assert candidates
    for line in candidates:
        assert line.max_gap_m <= 0.25
        assert line.thin_fraction > 0.9
        assert not (line.endpoints[:, 2].min() < 1.1 and line.endpoints[:, 2].max() > 1.7)
        assert len(line.witnesses) >= 12
        assert np.all((line.witnesses >= 0) & (line.witnesses < len(xyz)))
    np.testing.assert_array_equal(xyz, before)


def axis(y: float, *, shared: bool = True) -> MemberAxis:
    return MemberAxis(
        np.array([[0, y, 0], [4, y, 0]], dtype=np.float64),
        np.arange(40, dtype=np.int64) + (0 if shared else 100),
        1.0,
        0.0,
        np.zeros(3, dtype=np.float64),
        1.0,
        0.12,
        40,
    )


def test_shared_support_alias_is_removed_without_moving_endpoints() -> None:
    candidates = [axis(0), axis(0.01)]
    before = np.asarray([line.endpoints.copy() for line in candidates])
    assert competing(*candidates)
    assert select_member_axes(candidates, connectivity=False) == [0]
    assert select_member_axes(candidates, connectivity=True) == [0]
    np.testing.assert_array_equal([line.endpoints for line in candidates], before)


def test_independent_close_sheets_are_not_suppressed_by_distance_alone() -> None:
    assert not competing(axis(0), axis(0.02, shared=False))
    assert not competing(axis(0), axis(0.15))
    assert len(select_member_axes([axis(0), axis(0.02, shared=False)], connectivity=True)) == 2


def test_broad_bounded_bar_is_not_discarded_as_a_wall() -> None:
    y, z = np.meshgrid(np.arange(-0.22, 0.221, 0.02), np.arange(0, 3.01, 0.04))
    xyz = np.column_stack((np.zeros(y.size), y.ravel(), z.ravel()))
    line = np.array([[0, 0, 0.2], [0, 0, 2.8]], dtype=np.float64)
    centred, width = section_evidence(xyz, line, np.array([1.0, 0, 0]))
    assert centred > 0.9
    assert width > 0.36


def test_diagonal_crossing_a_true_upright_is_not_a_member_axis() -> None:
    y, z = np.meshgrid(np.arange(-0.06, 0.061, 0.02), np.arange(0, 3.01, 0.02))
    xyz = np.column_stack((np.zeros(y.size), y.ravel(), z.ravel()))
    diagonal = np.array([[0, -0.22, 1.28], [0, 0.22, 1.72]], dtype=np.float64)
    centred, _ = section_evidence(xyz, diagonal, np.array([1.0, 0, 0]))
    assert centred < 0.75


def test_wall_attachment_does_not_veto_a_locally_supported_member() -> None:
    y, z = np.meshgrid(np.arange(-1, 1.01, 0.04), np.arange(0, 6.01, 0.04))
    xyz = np.column_stack((np.zeros(y.size), y.ravel(), z.ravel()))
    xyz = xyz[(xyz[:, 2] <= 3) | (np.abs(xyz[:, 1]) < 0.09)]
    witnesses = np.flatnonzero(np.abs(xyz[:, 1]) < 0.10).astype(np.int64)
    line = np.array([[0, 0, 0], [0, 0, 6]], dtype=np.float64)
    thin, width = section_evidence(xyz, line, np.array([1.0, 0, 0]))
    assert thin < 0.75
    candidate = MemberAxis(line, witnesses, 1.0, 0.04, line[0], thin, width, 100)
    parts = split_member_sections(xyz, [candidate])
    assert parts
    assert min(part.endpoints[:, 2].min() for _, part in parts) >= 2.8
    assert max(part.endpoints[:, 2].max() for _, part in parts) > 5.8
    assert all(part.thin_fraction > 0.9 for _, part in parts)


def test_a_supported_diagonal_member_is_preserved_without_manhattan_snapping() -> None:
    t, r = np.meshgrid(np.arange(0, 3.01, 0.025), np.arange(-0.06, 0.061, 0.02))
    direction = np.array([0.0, 1, 1]) / np.sqrt(2)
    radial = np.array([0.0, -1, 1]) / np.sqrt(2)
    xyz = t.ravel()[:, None] * direction + r.ravel()[:, None] * radial
    diagonal = np.array([0.2, 2.8])[:, None] * direction
    centred, _ = section_evidence(xyz, diagonal, np.array([1.0, 0, 0]))
    assert centred > 0.9


def test_original_direction_field_rejects_borrowed_diagonal_support() -> None:
    y, z = np.meshgrid(np.arange(-0.06, 0.061, 0.02), np.arange(0, 3.01, 0.03))
    xyz = np.column_stack((np.zeros(y.size), y.ravel(), z.ravel()))
    rows = np.arange(len(xyz), dtype=np.int64)
    features = contour_features(xyz, rows, (0.15, 0.3, 0.5))
    upright = MemberAxis(
        np.array([[0, 0, 0.2], [0, 0, 2.8]], dtype=np.float64),
        rows,
        1.0,
        0.03,
        np.zeros(3),
        1.0,
        0.12,
        len(rows),
    )
    diagonal = MemberAxis(
        np.array([[0, -0.22, 1.28], [0, 0.22, 1.72]], dtype=np.float64),
        rows,
        1.0,
        0.03,
        np.zeros(3),
        1.0,
        0.12,
        len(rows),
    )
    args = (
        features.query_rows,
        features.eigenframe,
        features.linearity,
        features.neighbor_counts,
    )
    assert direction_field_coverage(xyz, upright, *args) > 0.9
    assert direction_field_coverage(xyz, diagonal, *args) < 0.1
    tree = KDTree(xyz)
    assert original_orientation_consistency(xyz, upright, tree)[0] == 1
    assert original_orientation_consistency(xyz, diagonal, tree)[0] == 0


def test_volumetric_sphere_is_not_silently_treated_as_a_facade() -> None:
    angle = np.arange(600, dtype=np.float64) * (np.pi * (3 - np.sqrt(5)))
    z = np.linspace(-0.998, 0.998, len(angle))
    radius = np.sqrt(1 - z * z)
    xyz = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))
    features = contour_features(xyz, np.arange(20, dtype=np.int64), (0.15, 0.3, 0.5))
    assert member_axis_proposals(xyz, features) == []


def test_complete_cross_section_recentres_an_offset_proposal_without_merging_depths() -> None:
    y, z = np.meshgrid(np.arange(-0.20, 0.201, 0.02), np.arange(0, 3.01, 0.02))
    lower = np.column_stack((np.zeros(y.size), y.ravel(), z.ravel()))
    xyz = np.vstack((lower, lower + [0.12, 0, 0]))
    for depth in (0.0, 0.12):
        sections = member_section_centres(
            xyz,
            np.array([depth, 0.15, 0]),
            np.array([0.0, 0, 1]),
            np.array([1.0, 0, 0]),
            0.2,
            2.8,
        )
        centres = sections.centres
        assert len(centres) > 20
        np.testing.assert_allclose(centres[:, 0], depth, atol=1e-12)
        np.testing.assert_allclose(centres[:, 1], 0, atol=1e-12)


def test_competition_uses_complete_component_identity_not_censored_axis_tubes() -> None:
    component = np.arange(120, dtype=np.int64)
    first = replace(axis(0), median_section_width_m=0.4, component_rows=component)
    same_member = replace(
        axis(0.12, shared=False),
        median_section_width_m=0.4,
        component_rows=component,
    )
    assert competing(first, same_member)
    other_sheet = replace(same_member, component_rows=component + 1000)
    assert not competing(first, other_sheet)
