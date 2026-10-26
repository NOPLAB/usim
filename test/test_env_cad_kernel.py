"""Real OpenCASCADE geometry, topology, orientation and exchange regressions."""

import numpy as np
import pytest

pytest.importorskip('OCP')

from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_Transform
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepTools import BRepTools
from OCP.GeomAbs import GeomAbs_Plane
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
from OCP.TopoDS import TopoDS
from OCP.gp import gp_Pnt, gp_Trsf

from usim_env_pipeline.cad_exchange import (
    read_brep,
    read_step,
    tessellate,
    write_brep,
    write_step,
)
from usim_env_pipeline.cad_kernel import (
    common_area,
    compound,
    face_loops,
    face_normal,
    faces,
    fuse_unify,
    make_planar_face,
    make_planar_faces,
    project_to_plane,
    remove_small_holes,
    shape_stats,
    surface_area,
    validate,
)


def rectangle(x, y, width, height, z=0):
    return np.array(
        [(x, y, z), (x + width, y, z), (x + width, y + height, z), (x, y + height, z)],
        dtype=float,
    )


def plate():
    return make_planar_face(
        rectangle(0, 0, 10, 10),
        [rectangle(1, 1, 0.5, 0.5)[::-1], rectangle(5, 5, 2, 2)[::-1]],
        normal=(0, 0, 1),
    )


def assert_oriented(face, expected):
    np.testing.assert_allclose(face_normal(face), expected, atol=1e-10)
    for index, loop in enumerate(face_loops(face)):
        signed = np.dot(np.cross(loop, np.roll(loop, -1, axis=0)).sum(0), expected)
        assert (signed > 0) == (index == 0)


def test_cad_hole_healing_preserves_large_holes_and_exact_threshold():
    source = plate()
    assert shape_stats(source) == {
        'area_m2': pytest.approx(95.75),
        'face_count': 1,
        'hole_count': 2,
        'valid': True,
    }
    healed = remove_small_holes(source, 1)
    assert shape_stats(healed)['area_m2'] == pytest.approx(96)
    assert shape_stats(healed)['hole_count'] == 1
    assert len(face_loops(healed)) == 2
    assert shape_stats(remove_small_holes(source, 0.25))['hole_count'] == 2
    assert shape_stats(source)['hole_count'] == 2
    assert BRepAdaptor_Surface(healed).GetType() == GeomAbs_Plane
    assert_oriented(healed, (0, 0, 1))
    reversed_face = source.Reversed()
    reversed_healed = remove_small_holes(reversed_face, 1)
    assert_oriented(faces(reversed_healed)[0], (0, 0, -1))


def test_real_common_fuse_and_same_domain_unification():
    left = make_planar_face(rectangle(0, 0, 2, 2))
    right = make_planar_face(rectangle(1, 0, 2, 2))
    assert common_area(left, right) == pytest.approx(2)
    assert common_area(left, make_planar_face(rectangle(4, 0, 1, 1))) == 0
    joined = fuse_unify([left, right])
    assert shape_stats(joined)['area_m2'] == pytest.approx(6)
    assert shape_stats(joined)['face_count'] == 1
    assert len(face_loops(faces(joined)[0])) == 1


def test_projected_faces_remain_analytic_with_holes():
    source = plate()
    normal = np.array([0, 1, 1]) / np.sqrt(2)
    origin = np.array([0, 0, 3])
    projected = project_to_plane(source, origin, normal)
    for loop in face_loops(projected):
        np.testing.assert_allclose((loop - origin) @ normal, 0, atol=1e-10)
    assert surface_area(projected) == pytest.approx(95.75 / np.sqrt(2))
    assert shape_stats(projected)['hole_count'] == 2
    assert BRepAdaptor_Surface(projected).GetType() == GeomAbs_Plane
    assert_oriented(projected, normal)


def test_separate_outer_components_own_their_holes():
    items = make_planar_faces(
        [
            rectangle(0, 0, 4, 4),
            rectangle(10, 0, 4, 4),
            rectangle(1, 1, 1, 1)[::-1],
            rectangle(11, 1, 2, 2)[::-1],
        ],
        (0, 0, 1),
    )
    assert len(items) == 2
    assert [surface_area(face) for face in items] == pytest.approx([15, 12])
    assert shape_stats(compound(items))['hole_count'] == 2
    with pytest.raises(ValueError):
        make_planar_faces([rectangle(0, 0, 1, 1), rectangle(3, 3, 1, 1)[::-1]], (0, 0, 1))


def test_tiny_touching_wires_are_removed_before_final_face_validation():
    items = make_planar_faces(
        [
            rectangle(0, 0, 10, 10),
            rectangle(5, 5, 2, 2)[::-1],
            rectangle(4.5, 4.5, 0.5, 0.5)[::-1],
            rectangle(0, 2, 0.5, 0.5)[::-1],
        ],
        (0, 0, 1),
        hole_threshold_m2=1.0,
    )
    assert len(items) == 1
    assert shape_stats(items[0])['hole_count'] == 1
    assert surface_area(items[0]) == pytest.approx(96)


def test_nested_island_holes_are_owned_by_the_smallest_containing_component():
    items = make_planar_faces(
        [
            rectangle(0, 0, 10, 10),
            rectangle(2, 2, 6, 6)[::-1],
            rectangle(3, 3, 4, 4),
            rectangle(4, 4, 1, 1)[::-1],
        ],
        (0, 0, 1),
    )
    assert [surface_area(face) for face in items] == pytest.approx([64, 15])
    assert shape_stats(compound(items))['hole_count'] == 2


@pytest.mark.parametrize(
    'outer,holes',
    [
        ([(0, 0, 0), (1, 1, 0), (0, 1, 0), (1, 0, 0)], []),
        ([(0, 0, 0), (1, 0, 0), (1, 1, 1), (0, 1, 0)], []),
        (rectangle(0, 0, 2, 2), [rectangle(3, 3, 1, 1)]),
        (rectangle(0, 0, 2, 2), [rectangle(1, 1, 2, 2)]),
    ],
)
def test_invalid_geometry_is_rejected(outer, holes):
    with pytest.raises(ValueError):
        make_planar_face(outer, holes, normal=(0, 0, 1))


def test_brep_and_step_roundtrip_preserve_meter_units_and_orientation(tmp_path):
    source = make_planar_face(
        rectangle(23, -7, 10, 10, 2),
        [rectangle(24, -6, 0.5, 0.5, 2), rectangle(28, -2, 2, 2, 2)],
        normal=(0, 0, -1),
    )
    for suffix, write, read in [('brep', write_brep, read_brep), ('step', write_step, read_step)]:
        path = tmp_path / f'plate.{suffix}'
        write(source, path)
        restored = read(path)
        assert validate(restored)
        assert surface_area(restored) == pytest.approx(95.75, rel=1e-9)
        assert shape_stats(restored)['hole_count'] == 2
        restored_face = faces(restored)[0]
        assert_oriented(restored_face, (0, 0, -1))
        original_loops, restored_loops = face_loops(source), face_loops(restored_face)
        for original, actual in zip(original_loops, restored_loops, strict=True):
            np.testing.assert_allclose(
                np.sort(original, axis=0), np.sort(actual, axis=0), atol=1e-9
            )
        if suffix == 'step':
            assert 'SI_UNIT(.MILLI.,.METRE.)' in path.read_text()


def test_tessellation_is_kernel_derived_with_holes_and_correct_winding():
    source = plate()
    vertices, indices = tessellate(source)
    triangles = vertices[indices]
    crosses = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    assert np.linalg.norm(crosses, axis=1).sum() / 2 == pytest.approx(surface_area(source))
    assert np.all(crosses[:, 2] > 0)
    centers = triangles.mean(axis=1)
    assert not np.any(np.all((centers[:, :2] > 5) & (centers[:, :2] < 7), axis=1))
    _, reversed_indices = tessellate(source.Reversed())
    reversed_triangles = vertices[reversed_indices]
    assert np.all(
        np.cross(
            reversed_triangles[:, 1] - reversed_triangles[:, 0],
            reversed_triangles[:, 2] - reversed_triangles[:, 0],
        )[:, 2]
        < 0
    )


def test_step_import_reconstructs_intersecting_trims_without_filling_openings(tmp_path):
    outer = make_planar_face(rectangle(0, 0, 10, 10))
    holes = [make_planar_face(rectangle(x, y, 2, 2)) for x, y in ((2, 2), (3, 3))]
    maker = BRepBuilderAPI_MakeFace(outer)
    for hole in holes:
        maker.Add(TopoDS.Wire_s(BRepTools.OuterWire_s(hole).Reversed()))
    source = maker.Face()
    original_loops = [loop.copy() for loop in face_loops(source)]
    transform = gp_Trsf()
    transform.SetScale(gp_Pnt(0, 0, 0), 1000)
    path = tmp_path / 'intersecting.step'
    writer = STEPControl_Writer()
    assert (
        writer.Transfer(
            BRepBuilderAPI_Transform(source, transform, True).Shape(), STEPControl_AsIs
        )
        == IFSelect_RetDone
    )
    assert writer.Write(str(path)) == IFSelect_RetDone
    original_bytes = path.read_bytes()
    reader = STEPControl_Reader()
    assert reader.ReadFile(str(path)) == IFSelect_RetDone
    reader.SetSystemLengthUnit(1)
    assert reader.TransferRoots() > 0
    assert not BRepCheck_Analyzer(reader.OneShape()).IsValid()

    restored = read_step(path)
    assert shape_stats(restored) == {
        'area_m2': pytest.approx(93),
        'face_count': 1,
        'hole_count': 1,
        'valid': True,
    }
    for hole in holes:
        assert common_area(restored, hole) == pytest.approx(0, abs=1e-9)
    assert path.read_bytes() == original_bytes
    for before, after in zip(original_loops, face_loops(source), strict=True):
        np.testing.assert_array_equal(after, before)
