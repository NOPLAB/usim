"""Ground connectivity retains indirect support and removes disconnected islands."""

import numpy as np
import pytest

pytest.importorskip('scipy')
pytest.importorskip('shapely')
o3d = pytest.importorskip('open3d')

connectivity = pytest.importorskip('usim_env_pipeline.simulation_ground_connectivity')


def cube(origin, size=(1, 1, 1)):
    mesh = o3d.geometry.TriangleMesh.create_box(*size)
    return (
        (np.asarray(mesh.vertices) + origin).astype(np.float32),
        np.asarray(mesh.triangles).astype(np.int32),
    )


def classify(meshes):
    points, owners, tables, geometries = [], [], [], []
    offset = 0
    for positions, faces in meshes:
        parts = connectivity.mesh_components(positions, faces)
        points.append(parts.positions)
        owners.append(parts.vertex_parts.astype(np.int64) + offset)
        chunks = connectivity.triangle_chunks(
            positions, faces, parts.face_parts.astype(np.int64) + offset
        )
        tables.append(chunks)
        geometries.extend(
            (positions, faces[chunks.order[start:stop]])
            for start, stop in zip(chunks.starts, chunks.stops)
        )
        offset += parts.count
    count, merged = connectivity.weld_components(
        np.concatenate(points), np.concatenate(owners), offset
    )
    graph = connectivity.ContactGraph(
        merged[np.concatenate([table.owners for table in tables])].astype(np.int64),
        np.concatenate([table.lower for table in tables]),
        np.concatenate([table.upper for table in tables]),
    )
    result = connectivity.ground_connections(
        graph, lambda index: geometries[index], count, int(merged[0])
    )
    return result.keep[merged]


@pytest.mark.parametrize('reverse', [False, True])
def test_indirect_column_and_roof_remain_while_floating_island_is_removed(reverse):
    # Given a finite floor, a narrow column and a large roof whose vertices miss the column.
    ground = (
        np.array([[-5, -5, 0], [5, -5, 0], [0, 5, 0]], np.float32),
        np.array([[0, 1, 2]], np.int32),
    )
    column = cube([0, 0, -0.1], (0.1, 0.1, 3))
    roof = (
        np.array([[-4, -4, 2], [4, -4, 2], [0, 4, 2]], np.float32),
        np.array([[0, 1, 2]], np.int32),
    )
    floating = cube([3, 3, 4])
    bodies = [column, roof, floating]
    if reverse:
        bodies.reverse()
    # When following real surface contacts in either input order.
    keep = classify([ground, *bodies])
    # Then the column-to-roof interior intersection propagates floor support.
    expected = [True, True, False]
    if reverse:
        expected.reverse()
    np.testing.assert_array_equal(keep, [True, *expected])


def test_overlapping_bounds_do_not_count_as_contact():
    # Given crossing bounds but two parallel, separated triangular surfaces.
    first = (
        np.array([[0, 0, 0], [2, 0, 2], [0, 2, 2]], np.float32),
        np.array([[0, 1, 2]], np.int32),
    )
    second = (first[0] + [0, 0, 0.1], first[1])
    # When using native intersection rather than bounds or near vertices.
    connected = connectivity.native_intersection(first, second)
    # Then a genuine positive gap does not retain the floating surface.
    assert not connected


def test_coplanar_overlap_and_crossing_triangle_interiors_are_contacts():
    # Given coplanar star-like overlap and a transverse interior intersection.
    first = (
        np.array([[-2, -1, 0], [2, -1, 0], [0, 2, 0]], np.float32),
        np.array([[0, 1, 2]], np.int32),
    )
    second = (np.array([[-2, 1, 0], [2, 1, 0], [0, -2, 0]], np.float32), first[1])
    third = (np.array([[0, 0, -1], [0, 0, 1], [0, 1, 1]], np.float32), first[1])
    # When checking actual triangle interiors.
    contacts = [connectivity.native_intersection(first, other) for other in (second, third)]
    # Then both are supported even though vertex-distance-only methods miss them.
    assert contacts == [True, True]


def test_exact_negative_coordinate_seams_weld_but_nearby_points_do_not():
    # Given identical seam coordinates from different payloads plus a small real gap.
    positions = np.array([[-15, -0.5, 2], [-15, -0.5, 2], [-15, -0.5, 2.001]], np.float32)
    # When reconnecting tile ownership without spatial rounding.
    count, labels = connectivity.weld_components(positions, np.arange(3, dtype=np.int64), 3)
    # Then the seam is one component but the separated point stays independent.
    assert count == 2 and labels[0] == labels[1] and labels[1] != labels[2]


def test_two_islands_in_one_payload_are_classified_separately():
    # Given one grounded cube and one detached cube in a shared source mesh.
    first, second = cube([0, 0, 0]), cube([0, 0, 3])
    mixed = (
        np.concatenate((first[0], second[0])),
        np.concatenate((first[1], second[1] + len(first[0]))),
    )
    ground = (
        np.array([[-2, -2, 0], [4, -2, 0], [-2, 4, 0]], np.float32),
        np.array([[0, 1, 2]], np.int32),
    )
    # When classifying connected geometry rather than whole payloads.
    keep = classify([ground, mixed])
    # Then only the grounded island survives.
    np.testing.assert_array_equal(keep, [True, True, False])


def test_below_or_outside_finite_ground_is_not_a_floor_contact():
    # Given no artificial infinite floor or half-space underneath it.
    ground = (
        np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], np.float32),
        np.array([[0, 1, 2]], np.int32),
    )
    below, outside = cube([0.1, 0.1, -3]), cube([5, 5, -0.5])
    # When testing the finite actual floor surface.
    keep = classify([ground, below, outside])
    # Then neither disconnected object is seeded merely by its height or bounds.
    np.testing.assert_array_equal(keep, [True, False, False])


def test_sloped_ground_contact_between_vertices_and_real_positive_gap():
    # Given a sloped floor crossed by a vertical triangle, plus a separated copy.
    ground = (
        np.array([[-2, -2, -0.2], [2, -2, 0.2], [0, 2, 0]], np.float32),
        np.array([[0, 1, 2]], np.int32),
    )
    crossing = (np.array([[0, 0, -1], [0, 0, 1], [0, 0.5, 1]], np.float32), ground[1])
    separated = (
        np.array([[0.5, 0, 0.051], [0.6, 0, 0.061], [0.5, 0.1, 0.051]], np.float32),
        ground[1],
    )
    # When querying full triangle bounds and actual interior intersections.
    keep = classify([ground, crossing, separated])
    # Then the transverse crossing is supported; a positive parallel gap is not.
    np.testing.assert_array_equal(keep, [True, True, False])


def fitted_fixture(folder, translated):
    from pxr import Usd, UsdGeom, Vt

    from usim_env_pipeline.simulation_curve_surface import CurveSurface
    from usim_env_pipeline.simulation_fit_output import publish_surface
    from usim_env_pipeline.storage import write_json
    from usim_env_pipeline.usd import LayerMetadata, stage_base

    metadata = LayerMetadata('CC-BY-4.0', 'fixture', '', 'XYZ', 'vertex')
    models = folder / 'simulation-models'
    (models / 'ground').mkdir(parents=True)
    fit = folder / 'geometry-fit'
    fit.mkdir()
    floor = CurveSurface(
        np.array([[-5, -5, 0], [5, -5, 0], [0, 5, 0]], np.float32),
        np.array([[0, 1, 2]], np.int32),
    )
    publish_surface(models / 'ground/floor.usdc', floor, metadata)
    np.savez(models / 'terrain.npz', positions=floor.positions, faces=floor.faces)
    baseline = stage_base(folder / 'simulation.usda', '/World', metadata)
    floor_path = '/World/SimulationGround/floor'
    UsdGeom.Xform.Define(baseline, floor_path).GetPrim().GetPayloads().AddPayload(
        './simulation-models/ground/floor.usdc', '/Tile'
    )
    baseline.GetRootLayer().Save()
    write_json(
        models / 'index.json',
        [
            {
                'path': floor_path,
                'payload': './simulation-models/ground/floor.usdc',
                'kind': 'ground',
            }
        ],
    )
    write_json(models / 'source-hashes.json', {})
    first, second = cube([0, 0, 0]), cube([0, 0, 3])
    positions = np.concatenate((first[0], second[0]))
    faces = np.concatenate((first[1], second[1] + len(first[0])))
    publish_surface(fit / 'mixed.usdc', CurveSurface(positions, faces), metadata)
    np.savez(fit / 'mixed.npz', positions=positions, faces=faces)
    payload = Usd.Stage.Open(str(fit / 'mixed.usdc'))
    colors = np.linspace(0.1, 0.9, len(positions) * 3, dtype=np.float32).reshape(-1, 3)
    mesh = UsdGeom.Mesh.Get(payload, '/Tile/Surface')
    mesh.GetDisplayColorPrimvar().Set(Vt.Vec3fArray.FromNumpy(colors))
    if translated:
        UsdGeom.Xform(payload.GetPrimAtPath('/Tile')).AddTranslateOp().Set((2, 0, 0))
    payload.GetRootLayer().Save()
    stage = stage_base(folder / 'simulation-fitted.usda', '/World', metadata)
    stage.GetRootLayer().subLayerPaths = ['./simulation.usda']
    path = '/World/FittedCurves/mixed'
    UsdGeom.Xform.Define(stage, path).GetPrim().GetPayloads().AddPayload(
        './geometry-fit/mixed.usdc', '/Tile'
    )
    stage.GetRootLayer().Save()
    write_json(
        fit / 'index.json',
        [
            {
                'path': path,
                'payload': './geometry-fit/mixed.usdc',
                'kind': 'curve',
            }
        ],
    )
    return positions, faces, colors, path


@pytest.mark.parametrize('translated', [False, True])
def test_filtered_usd_removes_floating_collision_and_preserves_sources_and_appearance(
    tmp_path, translated
):
    pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom

    from usim_env_pipeline.pcd import sha256
    from usim_env_pipeline.simulation_grounded import remove_floating

    # Given two islands in an actual payload, including an optional payload-local transform.
    positions, faces, colors, path = fitted_fixture(tmp_path, translated)
    authorities = list(tmp_path.rglob('*.usdc')) + list(tmp_path.rglob('*.npz'))
    authorities += [tmp_path / 'simulation.usda', tmp_path / 'simulation-fitted.usda']
    before = {file: sha256(file) for file in authorities}
    original = Usd.Stage.Open(str(tmp_path / 'simulation-fitted.usda'))
    original_mesh = UsdGeom.Mesh.Get(original, path + '/Surface')
    matrix = np.array(
        UsdGeom.XformCache().GetLocalToWorldTransform(original_mesh.GetPrim()), np.float64
    )
    # When publishing and auditing the non-destructive grounded view.
    remove_floating(tmp_path)
    stage = Usd.Stage.Open(str(tmp_path / 'simulation-grounded.usda'))
    mesh = UsdGeom.Mesh.Get(stage, path + '/Surface')
    actual = np.asarray(mesh.GetPointsAttr().Get())
    triangles = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    # Then retained triangle coordinates, vertex appearance, transforms and sources are unchanged.
    np.testing.assert_array_equal(actual[triangles], positions[faces[:12]])
    np.testing.assert_array_equal(mesh.GetDisplayColorPrimvar().ComputeFlattened(), colors[:8])
    np.testing.assert_array_equal(
        UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim()), matrix
    )
    assert {file: sha256(file) for file in authorities} == before
    scene = o3d.t.geometry.RaycastingScene()
    world_points = actual.astype(np.float64) @ matrix[:3, :3] + matrix[3, :3]
    scene.add_triangles(
        o3d.core.Tensor(world_points.astype(np.float32)),
        o3d.core.Tensor(triangles.astype(np.uint32)),
    )
    origin = [2.2 if translated else 0.2, 0.2, 6, 0, 0, -1]
    hit = scene.cast_rays(o3d.core.Tensor([origin], o3d.core.Dtype.Float32))
    assert float(hit['t_hit'].numpy()[0]) == pytest.approx(5.0)


def test_inactive_input_and_its_source_number_are_preserved(tmp_path):
    pytest.importorskip('pxr')
    import json
    from pxr import Usd, UsdGeom

    from usim_env_pipeline.simulation_grounded import remove_floating
    from usim_env_pipeline.storage import write_json

    # Given an inactive shared input before the active payload in the original index.
    _, _, _, path = fitted_fixture(tmp_path, False)
    stage = Usd.Stage.Open(str(tmp_path / 'simulation-fitted.usda'))
    inactive = '/World/FittedCurves/inactive'
    UsdGeom.Xform.Define(stage, inactive).GetPrim().SetActive(False)
    stage.GetRootLayer().Save()
    index = tmp_path / 'geometry-fit/index.json'
    rows = json.loads(index.read_text())
    write_json(index, [{**rows[0], 'path': inactive}, *rows])
    # When source numbers contain a skipped inactive row.
    remove_floating(tmp_path)
    filtered = Usd.Stage.Open(str(tmp_path / 'simulation-grounded.usda'))
    # Then the inactive input stays inactive and the active island is filtered and audited.
    assert not filtered.GetPrimAtPath(inactive).IsActive()
    assert (
        len(UsdGeom.Mesh.Get(filtered, path + '/Surface').GetFaceVertexIndicesAttr().Get()) == 36
    )


def test_audit_rejects_partial_component_mask_tampering(tmp_path):
    pytest.importorskip('pxr')
    import json

    from usim_env_pipeline.models import PipelineInputError
    from usim_env_pipeline.simulation_grounded import remove_floating
    from usim_env_pipeline.simulation_grounded_verify import verify

    # Given an already audited whole-component result.
    fitted_fixture(tmp_path, False)
    remove_floating(tmp_path)
    row = json.loads((tmp_path / 'grounded-models/index.json').read_text())[0]
    mask_file = tmp_path / row['mask']
    mask = np.load(mask_file)
    # When one retained triangle is removed without changing component reachability.
    mask[0] = False
    np.save(mask_file, mask)
    # Then the independent publication audit rejects the partial component.
    with pytest.raises(PipelineInputError):
        verify(tmp_path)
