"""Angular filtering keeps the original USD appearance domains and comparison data."""

import numpy as np
import pytest

pytest.importorskip('scipy')
pytest.importorskip('pxr')
pytest.importorskip('skimage')
pytest.importorskip('open3d')

from test.test_env_ground_connectivity import fitted_fixture


@pytest.mark.parametrize('name', ['angular', 'polyhedral'])
def test_angular_filter_preserves_face_normals_constant_color_and_inactive_data(tmp_path, name):
    from pxr import Usd, UsdGeom, Vt

    from usim_env_pipeline.simulation_angular import flat_normals
    from usim_env_pipeline.simulation_curve_surface import CurveSurface
    from usim_env_pipeline.simulation_grounded import remove_floating
    from usim_env_pipeline.storage import write_json
    from usim_env_pipeline.usd import LayerMetadata, stage_base

    # Given angular triangles with per-face normals and a constant display color.
    positions, faces, _, path = fitted_fixture(tmp_path, False)
    source = stage_base(
        tmp_path / f'simulation-{name}.usda',
        '/World',
        LayerMetadata('CC-BY-4.0', 'fixture', '', 'XYZ', 'vertex'),
    )
    source.GetRootLayer().subLayerPaths = ['./simulation-fitted.usda']
    source.OverridePrim('/World/FittedPlanes').SetActive(False)
    points = UsdGeom.Points.Define(source, '/World/PointRegions/original')
    points.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(positions))
    source.OverridePrim('/World/PointRegions').SetActive(False)
    mesh = UsdGeom.Mesh.Define(source, path + '/Surface')
    normals = flat_normals(CurveSurface(positions, faces))
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals))
    mesh.SetNormalsInterpolation('uniform')
    colors = mesh.CreateDisplayColorPrimvar('constant')
    colors.Set(Vt.Vec3fArray.FromNumpy(np.array([[0.2, 0.3, 0.4]], np.float32)))
    colors.BlockIndices()
    source.GetRootLayer().Save()
    (tmp_path / f'{name}-models').mkdir()
    write_json(
        tmp_path / f'{name}-models/full-index.json',
        [{'path': path, 'kind': 'curve', 'payload': './geometry-fit/mixed.usdc'}],
    )
    # When reclassifying against angular geometry without relying on excluded planes.
    remove_floating(tmp_path, angular=True, angular_name=name)
    stage = Usd.Stage.Open(str(tmp_path / f'simulation-{name}-grounded.usda'))
    actual = UsdGeom.Mesh.Get(stage, path + '/Surface')
    # Then floating faces disappear while flat shading and comparison data remain correct.
    assert len(actual.GetFaceVertexIndicesAttr().Get()) == 36
    assert actual.GetNormalsInterpolation() == 'uniform'
    np.testing.assert_array_equal(actual.GetNormalsAttr().Get(), normals[:12])
    assert actual.GetDisplayColorPrimvar().GetInterpolation() == 'constant'
    np.testing.assert_array_equal(
        actual.GetDisplayColorPrimvar().Get(), np.array([[0.2, 0.3, 0.4]], np.float32)
    )
    assert not stage.GetPrimAtPath('/World/FittedPlanes').IsActive()
    assert not stage.GetPrimAtPath('/World/PointRegions').IsActive()
    stage.OverridePrim('/World/PointRegions').SetActive(True)
    np.testing.assert_array_equal(
        UsdGeom.Points.Get(stage, '/World/PointRegions/original').GetPointsAttr().Get(),
        positions,
    )
