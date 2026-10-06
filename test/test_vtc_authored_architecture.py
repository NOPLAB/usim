"""CPU-only USD checks for original architecture envelopes and circulation."""

from pathlib import Path
import json

import pytest

pytest.importorskip('pxr')
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from usim_env_vtc.authored_architecture import author_architecture


CITY = 'TCAssets/CityHall/City_Hall_'
COMPONENTS = [
    '1F_noentry',
    '2F_floor',
    '2F_handrail_1',
    '2F_handrail_2',
    'balcony',
    'counter_1',
    'counter_2',
    'east_entrance',
    'east_handrail',
    'east_roof',
    'EV',
    'floor',
    'Pillar_01',
    'Pillar_10',
    'Pillar_A',
    'Pillar_O',
    'reception',
    'south_entrance',
    'south_roof',
    'south_room',
    'Stairs',
    'top_roof',
    'walls',
    'west_handrail',
]
PROPS = [
    'TCAssets/CityHallChair',
    'TCAssets/CityHallTable1',
    'TCAssets/CityHallTable2',
    'TCAssets/RoadCone',
    'Engine/BasicShapes/Cube',
    'Engine/BasicShapes/Cone',
    'Engine/BasicShapes/Cylinder',
    'Engine/BasicShapes/Plane',
    'Engine/BasicShapes/Sphere',
    'Engine/EngineMeshes/Cylinder',
    'Engine/EngineMeshes/Sphere',
    'Engine/ArtTools/RenderToTexture/Meshes/S_1_Unit_Plane',
    'StarterContent/Shapes/Shape_Cylinder',
    'StarterContent/Shapes/Shape_TriPyramid',
    'ParagonProps/Agora/Props/Meshes/Rock_Formation_Strip_C',
    'Mannequin/Character/Mesh/SK_Mannequin',
]


def bounds(stage, prim):
    return (
        UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_],
        )
        .ComputeWorldBound(prim)
        .ComputeAlignedRange()
    )


def author(name, low=(0, 0, 0), high=(20, 10, 4)):
    stage = Usd.Stage.CreateInMemory()
    author_architecture(stage, '/Prototype', name, low, high)
    return stage


@pytest.mark.parametrize('name', [CITY + part for part in COMPONENTS] + PROPS)
def test_recipes_are_deterministic_bounded_and_self_contained(name):
    # Asymmetric placement bounds exercise local coordinates without root transforms.
    low, high = (-3, -8, 2), (17, 7, 6)
    stage = author(name, low, high)
    other = author(name, low, high)
    assert stage.GetRootLayer().ExportToString() == other.GetRootLayer().ExportToString()
    root = stage.GetPrimAtPath('/Prototype')
    assert not UsdGeom.Xformable(root).GetOrderedXformOps()
    extent = bounds(stage, root)
    for axis in range(3):
        assert extent.GetMin()[axis] >= low[axis] - 1e-5
        assert extent.GetMax()[axis] <= high[axis] + 1e-5
    meshes = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    assert meshes
    collisions = [prim.HasAPI(UsdPhysics.CollisionAPI) for prim in meshes]
    if 'handrail' in name:
        assert not any(collisions)
    else:
        assert any(collisions)
    assert stage.GetRootLayer().GetExternalReferences() == ()
    assert not stage.GetRootLayer().subLayerPaths
    for prim in stage.Traverse():
        assert not prim.HasAuthoredReferences()
        assert not prim.HasAuthoredPayloads()
        for attr in prim.GetAttributes():
            assert attr.GetTypeName() not in (
                Sdf.ValueTypeNames.Asset,
                Sdf.ValueTypeNames.AssetArray,
            )
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            assert not mesh.GetPrim().GetAttribute('primvars:st')
            assert UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            counts = mesh.GetFaceVertexCountsAttr().Get()
            indices = mesh.GetFaceVertexIndicesAttr().Get()
            assert sum(counts) == len(indices)
            assert max(indices) < len(mesh.GetPointsAttr().Get())


def test_actual_inventory_dimensions_are_supported_without_source_art():
    inventory = Path(__file__).resolve().parents[1] / 'runs/vtc-original/inventory.json'
    if not inventory.exists():
        pytest.skip('Optional local dimensions inventory is not distributed with the package')
    models = json.loads(inventory.read_text(encoding='utf-8'))['models']
    for model in models:
        name = model['name']
        if not name.startswith(CITY) and name not in PROPS:
            continue
        stage = author(name, tuple(model['min']), tuple(model['max']))
        extent = bounds(stage, stage.GetPrimAtPath('/Prototype'))
        for axis in range(3):
            assert extent.GetMin()[axis] >= model['min'][axis] - 1e-5, name
            assert extent.GetMax()[axis] <= model['max'][axis] + 1e-5, name


def intersects(low, high, extent):
    return all(
        extent.GetMin()[axis] < high[axis] and extent.GetMax()[axis] > low[axis]
        for axis in range(3)
    )


def test_facades_preserve_empty_interior_and_traversable_entrance_corridors():
    for part, height in [('walls', 10.4), ('1F_noentry', 3.7)]:
        stage = author(CITY + part, (0, -65, 0), (94.1, 0, height))
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            extent = bounds(stage, prim)
            # No geometry, even noncollidable panes, plugs either entry.
            assert not intersects((-1, -19, 0.1), (1, -15, 2.6), extent)
            assert not intersects((49, -1, 0.1), (52, 1, 2.6), extent)
            assert not intersects((5, -60, 0.1), (89, -5, 3), extent)
            if 'Glass' in prim.GetName():
                assert not prim.HasAPI(UsdPhysics.CollisionAPI)
        assert any('Glass' in prim.GetName() for prim in stage.Traverse())


@pytest.mark.parametrize(
    'part,low,high',
    [
        ('south_entrance', (48.3, -3.9, 0), (53, -3.85, 3.7)),
        ('east_entrance', (0, -20.3, 0), (5.7, -13.7, 3.7)),
    ],
)
def test_portals_leave_full_height_passage_without_glass(part, low, high):
    stage = author(CITY + part, low, high)
    thin_axis = 0 if high[0] - low[0] < high[1] - low[1] else 1
    along = 1 - thin_axis
    corridor_low, corridor_high = list(low), list(high)
    span = high[along] - low[along]
    corridor_low[along] += span * 0.2
    corridor_high[along] -= span * 0.2
    corridor_low[2], corridor_high[2] = 0.1, 2.5
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            assert not intersects(corridor_low, corridor_high, bounds(stage, prim))


def test_stair_flight_meets_upper_floor_with_real_treads_and_clear_aperture():
    stage = Usd.Stage.CreateInMemory()
    low, high = (20.163705825805664, -21.25, 0), (36.094017028808594, -12.949999809265137, 5.4)
    author_architecture(stage, '/Stairs', CITY + 'Stairs', low, high)
    author_architecture(
        stage, '/Floor', CITY + '2F_floor', (0, -65, 3.7), (94.0999984741211, 0, 5.4)
    )
    treads = stage.GetPrimAtPath('/Stairs').GetChildren()
    treads = [prim for prim in treads if prim.IsA(UsdGeom.Mesh)]
    assert len(treads) == 30
    previous_x, previous_z = low[0], 0
    for tread in treads:
        extent = bounds(stage, tread)
        assert tread.HasAPI(UsdPhysics.CollisionAPI)
        assert extent.GetMin()[0] == pytest.approx(previous_x, abs=1e-5)
        assert extent.GetMax()[2] - previous_z == pytest.approx(0.18, abs=1e-5)
        assert extent.GetMax()[0] - extent.GetMin()[0] > 0.28
        assert extent.GetSize()[1] > 1
        previous_x, previous_z = extent.GetMax()[0], extent.GetMax()[2]
    assert previous_z == pytest.approx(5.4)
    slabs = [
        prim for prim in stage.GetPrimAtPath('/Floor').GetChildren() if prim.IsA(UsdGeom.Mesh)
    ]
    for slab in slabs:
        extent = bounds(stage, slab)
        assert extent.GetMax()[2] == pytest.approx(5.4)
        assert extent.GetSize()[2] <= 0.181
        assert not intersects(
            (low[0] + 0.01, low[1] + 0.01, 3.7),
            (high[0] - 0.01, high[1] - 0.01, 5.4),
            extent,
        )
    landing = bounds(stage, stage.GetPrimAtPath('/Floor/Slab_1'))
    assert landing.GetMin()[0] == pytest.approx(previous_x, abs=1e-5)


def test_roof_and_balcony_are_thin_and_rails_are_not_collision_obstacles():
    roof = author(CITY + 'top_roof', (0, 0, 10.1), (94.7, 65.6, 10.4))
    assert bounds(roof, roof.GetPrimAtPath('/Prototype/Roof_Slab')).GetSize()[2] <= 0.161
    balcony = author(CITY + 'balcony', (0, 0, 3.7), (89.1, 2.1, 5.2))
    slab = balcony.GetPrimAtPath('/Prototype/Balcony_Slab')
    assert slab.HasAPI(UsdPhysics.CollisionAPI)
    assert bounds(balcony, slab).GetSize()[2] <= 0.25
    rails = [prim for prim in balcony.Traverse() if prim.GetName().startswith('Rail_')]
    assert rails
    assert not any(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in rails)


@pytest.mark.parametrize('name', ['Unknown/Box', CITY + 'missing', CITY + 'Pillar_Z'])
def test_unknown_categories_fail_before_authoring(name):
    stage = Usd.Stage.CreateInMemory()
    with pytest.raises(ValueError, match='Unsupported authored architecture category'):
        author_architecture(stage, '/Bad', name, (0, 0, 0), (1, 1, 1))
    assert not stage.GetPrimAtPath('/Bad')


@pytest.mark.parametrize(
    'low,high',
    [
        ((0, 0, 0), (1, 1, -1)),
        ((0, 0, 0), (0, 1, 1)),
        ((0, 0, 0), (1, float('nan'), 1)),
    ],
)
def test_invalid_envelopes_fail_before_authoring(low, high):
    stage = Usd.Stage.CreateInMemory()
    with pytest.raises(ValueError, match='Invalid prototype bounds'):
        author_architecture(stage, '/Bad', CITY + 'walls', low, high)
    assert not stage.GetPrimAtPath('/Bad')


@pytest.mark.parametrize(
    'name',
    ['Engine/BasicShapes/Cone', 'Engine/BasicShapes/Sphere', PROPS[14], 'TCAssets/RoadCone'],
)
def test_rounded_solids_have_nonzero_faces_and_closed_opposite_edges(name):
    # Given an original rounded prototype rather than an imported mesh.
    stage = author(name)
    # When reading the actual face polygons and directed edges.
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        edges = {}
        start = 0
        for count in mesh.GetFaceVertexCountsAttr().Get():
            face = list(indices[start : start + count])
            start += count
            normal = Gf.Vec3f(0)
            for i in range(1, count - 1):
                normal += Gf.Cross(
                    points[face[i]] - points[face[0]], points[face[i + 1]] - points[face[0]]
                )
            # Then every face has geometric area, including poles and cone tips.
            assert normal.GetLength() > 1e-7
            for a, b in zip(face, face[1:] + face[:1]):
                edges[a, b] = edges.get((a, b), 0) + 1
        assert all(count == 1 and edges.get((b, a)) == 1 for (a, b), count in edges.items())


def test_stair_flight_does_not_fill_existing_ground_floor_furniture_corridor():
    # Given the source stair envelope and an independently placed ground-floor table.
    stage = Usd.Stage.CreateInMemory()
    # When fitting the original flight along the envelope's side.
    author_architecture(
        stage,
        '/Stairs',
        CITY + 'Stairs',
        (20.163705825805664, -21.25, 0),
        (36.094017028808594, -12.949999809265137, 5.4),
    )
    # Then the side flight stays useful without swallowing the retained table row.
    extent = bounds(stage, stage.GetPrimAtPath('/Stairs'))
    assert extent.GetSize()[1] == pytest.approx(2, abs=1e-5)
    assert not intersects((26.67, -18.70, 0), (28.55, -17.82, 1.86), extent)
