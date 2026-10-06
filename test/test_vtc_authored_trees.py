"""Original vegetation prototypes are bounded, self-contained and reusable in real USD."""

from collections import Counter

import pytest

pytest.importorskip('pxr')
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from usim_env_vtc.authored_trees import author_tree


@pytest.fixture(params=['broadleaf', 'pine', 'bush', 'stump'])
def tree(request):
    # Given an empty stage and each supported vegetation kind.
    stage = Usd.Stage.CreateInMemory()
    # When authoring an original reusable prototype.
    author_tree(stage, '/Tree', request.param, seed=17)
    return stage, request.param


def test_tree_stays_in_normalized_bounds_without_root_transforms(tree):
    # Given an authored prototype.
    stage, _ = tree
    # When reading its world-space geometry bounds.
    root = stage.GetPrimAtPath('/Tree')
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default']).ComputeWorldBound(root)
    extent = bounds.ComputeAlignedRange()
    # Then the caller can place and scale the root without hidden offsets.
    assert UsdGeom.Xform(root).GetOrderedXformOps() == []
    assert extent.GetMin()[2] == pytest.approx(0)
    assert extent.GetMax()[2] == pytest.approx(1)
    assert all(-1.000001 <= extent.GetMin()[i] <= extent.GetMax()[i] <= 1.000001 for i in range(2))


def test_closed_wood_has_consistent_winding_and_only_wood_collides(tree):
    # Given an authored tree with wood and possibly foliage.
    stage, _ = tree
    # When examining the closed wood topology.
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath('/Tree/Wood'))
    indices, points = mesh.GetFaceVertexIndicesAttr().Get(), mesh.GetPointsAttr().Get()
    directed = Counter()
    volume = 0
    for i in range(0, len(indices), 3):
        a, b, c = indices[i : i + 3]
        directed.update(((a, b), (b, c), (c, a)))
        volume += Gf.Dot(points[a], Gf.Cross(points[b], points[c])) / 6
    # Then each edge is paired oppositely and collision excludes canopy surfaces.
    assert all(count == 1 and directed[(b, a)] == 1 for (a, b), count in directed.items())
    assert volume > 0
    colliders = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)]
    assert [str(prim.GetPath()) for prim in colliders] == ['/Tree/Wood']
    assert UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Get() == 'none'


def test_materials_use_local_vertex_pigment_with_valid_normals_and_extents(tree):
    # Given the authored geometry.
    stage, _ = tree
    # When reading its shading and geometric declarations.
    meshes = [UsdGeom.Mesh(prim) for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    # Then both renderers and bounds consumers have complete local data.
    for mesh in meshes:
        points = mesh.GetPointsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        colors = mesh.GetDisplayColorPrimvar()
        assert colors.GetInterpolation() == 'vertex'
        assert len(colors.Get()) == len(points)
        assert len(set(tuple(c) for c in colors.Get())) > 3
        assert all(0 <= channel <= 1 for color in colors.Get() for channel in color)
        normals = mesh.GetNormalsAttr().Get()
        assert mesh.GetNormalsInterpolation() == 'faceVarying'
        assert len(normals) == len(indices)
        assert all(normal.GetLength() == pytest.approx(1, abs=1e-5) for normal in normals)
        extent = mesh.GetExtentAttr().Get()
        assert all(extent[0][i] <= p[i] <= extent[1][i] for p in points for i in range(3))
        material, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
        assert str(material.GetPath()).startswith('/Tree/')
        shader = UsdShade.Shader(stage.GetPrimAtPath(str(material.GetPath()) + '/Surface'))
        assert shader.GetIdAttr().Get() == 'UsdPreviewSurface'
        reader, _, _ = shader.GetInput('diffuseColor').GetConnectedSource()
        assert reader.GetPrim().GetAttribute('inputs:varname').Get() == 'displayColor'


def test_foliage_is_detailed_disconnected_backface_safe_leaf_surfaces(tree):
    # Given a reusable tree prototype.
    stage, kind = tree
    # When counting the authored foliage triangles.
    prim = stage.GetPrimAtPath('/Tree/Leaves')
    # Then living vegetation uses many individual leaves rather than a canopy hull.
    if kind == 'stump':
        assert not prim
        return
    mesh = UsdGeom.Mesh(prim)
    points, indices = mesh.GetPointsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get()
    assert mesh.GetDoubleSidedAttr().Get()
    assert len(points) // 7 >= 2500
    assert all(index // 7 == indices[i - i % 18] // 7 for i, index in enumerate(indices))
    triangles = sum(
        len(UsdGeom.Mesh(p).GetFaceVertexIndicesAttr().Get()) // 3
        for p in stage.Traverse()
        if p.IsA(UsdGeom.Mesh)
    )
    assert 20000 <= triangles <= 50000


@pytest.mark.parametrize('kind', ['broadleaf', 'pine', 'bush', 'stump'])
def test_geometry_and_colors_are_repeatable_for_same_seed(kind):
    # Given independent empty USD stages.
    stages = [Usd.Stage.CreateInMemory() for _ in range(2)]
    # When generating the same species and seed independently.
    for stage in stages:
        author_tree(stage, '/Tree', kind, seed=93)
    # Then every authored value is deterministic.
    assert stages[0].GetRootLayer().ExportToString() == stages[1].GetRootLayer().ExportToString()


@pytest.mark.parametrize('kind', ['broadleaf', 'pine', 'bush'])
def test_seed_changes_geometry(kind):
    # Given independent stages for two specimen seeds.
    stages = [Usd.Stage.CreateInMemory() for _ in range(2)]
    # When authoring different seeds of one species.
    for seed, stage in enumerate(stages):
        author_tree(stage, '/Tree', kind, seed=seed)
    # Then their original foliage is not identical.
    points = [
        UsdGeom.Mesh(stage.GetPrimAtPath('/Tree/Leaves')).GetPointsAttr().Get() for stage in stages
    ]
    assert points[0] != points[1]


def test_exported_prototype_has_no_external_dependencies(tree, tmp_path):
    # Given an original authored prototype.
    stage, _ = tree
    filename = tmp_path / 'tree.usdc'
    # When exporting and reopening it without another asset.
    stage.GetRootLayer().Export(str(filename))
    reopened = Usd.Stage.Open(str(filename))
    # Then all composition and shader dependencies stay inside the prototype.
    assert reopened.GetRootLayer().subLayerPaths == []
    assert len(reopened.GetUsedLayers()) == 2  # Root and anonymous session layer.
    for prim in reopened.Traverse():
        assert not prim.HasAuthoredReferences()
        assert not prim.HasAuthoredPayloads()
        for attribute in prim.GetAttributes():
            assert attribute.GetTypeName() not in (
                Sdf.ValueTypeNames.Asset,
                Sdf.ValueTypeNames.AssetArray,
            )
            assert all(path.HasPrefix('/Tree') for path in attribute.GetConnections())
