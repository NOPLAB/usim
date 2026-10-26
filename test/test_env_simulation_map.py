"""Simulation composition retains sparse observations and whole unresolved source faces."""

import numpy as np
import pytest

pytest.importorskip('scipy')
pytest.importorskip('pxr')

from pxr import Usd, UsdGeom, UsdPhysics

from usim_env_pipeline.simulation_coverage import voxels_cover
from usim_env_pipeline.simulation_ground import GroundConfig, Terrain
from usim_env_pipeline.simulation_obstacles import ObstacleConfig, occupied_obstacles
from usim_env_pipeline.simulation_output import covered_by_buildings, publish
from usim_env_pipeline.simulation_residual import publish_residual
from usim_env_pipeline.storage import write_json
from usim_env_pipeline.usd import Geometry, LayerMetadata, author_geometry, stage_base

METADATA = LayerMetadata('CC-BY-4.0', 'fixture', '', 'XYZ', 'source fixture')


def empty_terrain():
    return Terrain(
        np.empty((0, 3), np.float32),
        np.empty((0, 3), np.int32),
        np.empty((0, 2), np.int64),
        np.empty((0, 2), np.int64),
        np.empty((0, 4), np.int32),
        np.empty((0, 12), np.int64),
    )


def test_whole_face_crossing_an_empty_voxel_is_not_removed_by_its_vertices():
    # Given source vertices inside two occupied volumes but a triangle crossing empty space.
    triangles = np.array([[[0.1, 0.1, 0.1], [0.9, 0.1, 0.1], [0.1, 0.3, 0.1]]])
    keys = np.array([[0, 0, 0], [2, 0, 0]], dtype=np.int64)
    # When proving full-face coverage, rather than sampling the centroid.
    covered = voxels_cover(triangles, keys, 0.4)
    # Then the genuine measured connection remains unresolved, not silently discarded.
    assert not covered.any()


@pytest.mark.parametrize('floor_authority', [False, True])
def test_retained_source_faces_and_archival_bytes_survive_composition(tmp_path, floor_authority):
    # Given one fully proxy-covered source triangle and another crossing a gap.
    positions = np.array(
        [[0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [0.1, 0.2, 0.1], [0.9, 0.1, 0.1]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 3, 2]], dtype=np.int32)
    source = tmp_path / 'source.usdc'
    author_geometry(
        source,
        Geometry(positions, np.arange(4), np.full((4, 3), 0.5, np.float32), faces),
        METADATA,
    )
    original_bytes = source.read_bytes()
    authored = stage_base(tmp_path / 'authored.usda', '/World', METADATA)
    UsdGeom.Xform.Define(authored, '/World/Regions/mixed').GetPrim().GetPayloads().AddPayload(
        './source.usdc', '/Tile'
    )
    for root in ('Ground', 'PointRegions', 'InferredStructures'):
        UsdGeom.Xform.Define(authored, '/World/' + root)
    authored.GetRootLayer().Save()
    write_json(
        tmp_path / 'authored-index.json',
        [{'path': '/World/Regions/mixed', 'payload': 'source.usdc'}],
    )
    buildings = stage_base(tmp_path / 'buildings.usda', '/World', METADATA)
    buildings.GetRootLayer().subLayerPaths = ['./authored.usda']
    UsdGeom.Xform.Define(buildings, '/World/BuildingModels')
    buildings.GetRootLayer().Save()
    (tmp_path / 'building-models').mkdir()
    write_json(tmp_path / 'building-models/models.json', [])
    config = ObstacleConfig(cell=0.4)
    occupied = occupied_obstacles(positions, np.arange(4), config)
    terrain = empty_terrain()
    floor_ids = None
    if floor_authority:
        occupied = occupied_obstacles(positions, np.empty(0, np.int64), config)
        floor_ids = np.arange(3, dtype=np.int64)
        terrain = Terrain(
            np.array([[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0]], np.float32),
            np.array([[0, 1, 2], [0, 2, 3]], np.int32),
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.int64),
            np.array([[0, 0]], np.int64),
            np.array([[0, 1, 2, 3]], np.int32),
            np.full((4, 48), -1, np.int64),
        )
    # When publishing model-only geometry plus complete uncovered original faces.
    retained, total, replaced, discarded = publish_residual(
        tmp_path,
        terrain,
        occupied,
        GroundConfig(),
        config,
        METADATA,
        np.empty(0, np.int64),
        ground_ids=floor_ids,
    )
    tiles = publish(tmp_path, terrain, occupied, config.cell, METADATA)
    write_json(tmp_path / 'simulation-models/index.json', tiles + retained)
    assert (total, replaced) == (2, 1)
    assert discarded == 0
    with np.load(tmp_path / 'simulation-models/residual/mixed.npz') as residual:
        np.testing.assert_array_equal(residual['source_face_indices'], [1])
    stage = Usd.Stage.Open(str(tmp_path / 'simulation.usda'))
    for row in retained:
        UsdGeom.Xform.Define(stage, row['path']).GetPrim().GetPayloads().AddPayload(
            row['payload'], '/Tile'
        )
    stage.GetRootLayer().Save()
    # Then old duplicates are inactive, every unresolved face still collides, archives unchanged.
    for name in ('Regions', 'Ground', 'PointRegions', 'InferredStructures'):
        assert not stage.GetPrimAtPath('/World/' + name).IsActive()
    assert stage.GetPrimAtPath('/World/BuildingModels').IsActive()
    mesh = UsdGeom.Mesh.Get(stage, '/World/SimulationUnresolved/mixed/Surface')
    assert len(mesh.GetFaceVertexCountsAttr().Get()) == 1
    assert UsdPhysics.CollisionAPI(mesh.GetPrim()).GetCollisionEnabledAttr().Get()
    assert source.read_bytes() == original_bytes
    from usim_env_pipeline.simulation_verify_faces import verify_faces

    assert verify_faces(tmp_path, stage, np.empty(0, np.int64), floor_ids) == (2, 1, 1, 0)


def test_expanded_box_corner_is_only_broad_phase_not_source_ownership(tmp_path):
    # Given a point in a box's expanded AABB but beyond its Euclidean support tolerance.
    folder = tmp_path / 'building-models'
    folder.mkdir()
    write_json(folder / 'models.json', [{'name': 'model_0000'}])
    np.savez(folder / 'model_0000.npz', center=np.zeros(3), size=np.ones(3), axes=np.eye(3))
    points = np.array([[0.6, 0.6, 0.6], [0.65, 0.65, 0.65], [0, 0, 0]], dtype=np.float32)
    # When classifying points against actual bounded cuboids.
    covered = covered_by_buildings(points, folder, 0.2)
    # Then a diagonal expanded-box-only point remains available to obstacle reconstruction.
    np.testing.assert_array_equal(covered, [True, False, True])


def test_ground_crossing_output_tiles_is_one_smooth_collidable_mesh(tmp_path):
    from test.test_env_simulation_ground import measured_grid, reconstruct

    # Given a continuous floor spanning several old 30-meter output tiles.
    terrain = reconstruct(measured_grid(-1, 32, 0.25))
    points = np.empty((0, 3), np.float32)
    obstacles = occupied_obstacles(points, np.empty(0, np.int64), ObstacleConfig())
    buildings = stage_base(tmp_path / 'buildings.usda', '/World', METADATA)
    buildings.GetRootLayer().Save()
    # When publishing the simulation scene.
    entries = publish(tmp_path, terrain, obstacles, 0.1, METADATA)
    stage = Usd.Stage.Open(str(tmp_path / 'simulation.usda'))
    meshes = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    # Then all floor triangles share one mesh, smooth normals and exact collision.
    assert len(entries) == len(meshes) == 1
    mesh = UsdGeom.Mesh(meshes[0])
    np.testing.assert_array_equal(mesh.GetPointsAttr().Get(), terrain.positions)
    np.testing.assert_array_equal(
        np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3), terrain.faces
    )
    assert mesh.GetNormalsInterpolation() == UsdGeom.Tokens.vertex
    normals = np.asarray(mesh.GetNormalsAttr().Get())
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-6)
    assert (normals[:, 2] > 0).all()
    assert UsdPhysics.CollisionAPI(mesh.GetPrim()).GetCollisionEnabledAttr().Get()
    assert UsdPhysics.MeshCollisionAPI(mesh.GetPrim()).GetApproximationAttr().Get() == 'none'


def test_voxel_collision_does_not_fill_a_hollow_object_or_space_below_it(tmp_path):
    # Given a hollow elevated shell, not a semantic tree crown or ground column.
    keys = np.array(
        [
            [x, y, z + 10]
            for x in range(3)
            for y in range(3)
            for z in range(3)
            if x in (0, 2) or y in (0, 2) or z in (0, 2)
        ],
        dtype=np.int64,
    )
    positions = ((keys + 0.5) * 0.4).astype(np.float32)
    occupied = occupied_obstacles(positions, np.arange(len(positions)), ObstacleConfig(cell=0.4))
    # When publishing exact static occupied-volume boundary collision.
    buildings = stage_base(tmp_path / 'buildings.usda', '/World', METADATA)
    UsdGeom.Xform.Define(buildings, '/World/BuildingModels')
    buildings.GetRootLayer().Save()
    publish(tmp_path, empty_terrain(), occupied, 0.4, METADATA)
    stage = Usd.Stage.Open(str(tmp_path / 'simulation.usda'))
    # Then the missing center cell and all space below remain absent from the authority.
    assert not (occupied.keys == [1, 1, 11]).all(1).any()
    assert occupied.keys[:, 2].min() == 10
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            assert UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == 'none'


def test_real_generator_accepts_nonpayload_structure_bounds_and_passes_verifier(tmp_path):
    # Given a source-backed ground map whose inferred-structure index contains bounds only.
    from test.env_pipeline_fixtures import write_pcd
    from usim_env_pipeline.ground import author_ground, reconstruct_ground
    from usim_env_pipeline.models import Support
    from usim_env_pipeline.pcd import sha256
    from usim_env_pipeline.simulation_map import ground_support, reconstruct
    from usim_env_pipeline.simulation_verify import verify

    x, y = np.meshgrid(np.arange(-2, 4.01, 0.1), np.arange(-2, 4.01, 0.1))
    floor = np.column_stack((x.ravel(), y.ravel(), np.where(x.ravel() < 0.25, 0, 0.9))).astype(
        np.float32
    )
    points = np.concatenate((floor, [[5.1, 0.1, 2.1]])).astype(np.float32)
    records = np.zeros(len(points), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
    for axis, field in enumerate(('x', 'y', 'z')):
        records[field] = points[:, axis]
    source = tmp_path / 'source.pcd'
    original = write_pcd(source, records)
    write_json(
        tmp_path / 'manifest.json',
        {
            'primary_source': 'map',
            'sources': {
                'map': {
                    'path': 'source.pcd',
                    'sha256': sha256(source),
                    'license': 'CC-BY-4.0',
                    'attribution': 'fixture',
                }
            },
            'source_catalog': '',
            'frame': 'XYZ',
        },
    )
    world = stage_base(tmp_path / 'authored.usda', '/World', METADATA)
    for root in ('Regions', 'PointRegions', 'InferredStructures'):
        UsdGeom.Xform.Define(world, '/World/' + root)
    surface = reconstruct_ground(floor, np.zeros(len(floor)), Support())
    entries = author_ground(world, tmp_path, floor, surface, 30, METADATA)
    world.GetRootLayer().Save()
    write_json(
        tmp_path / 'authored-index.json',
        entries + [{'path': '/World/InferredStructures/post', 'min': [0, 0, 0], 'max': [1, 1, 2]}],
    )
    buildings = stage_base(tmp_path / 'buildings.usda', '/World', METADATA)
    buildings.GetRootLayer().subLayerPaths = ['./authored.usda']
    UsdGeom.Xform.Define(buildings, '/World/BuildingModels')
    buildings.GetRootLayer().Save()
    (tmp_path / 'building-models').mkdir()
    write_json(tmp_path / 'building-models/models.json', [])
    # When invoking the actual generator and independently auditing its outputs.
    report = reconstruct(tmp_path)
    verify(tmp_path)
    # Then an isolated elevated return is recorded as removed noise, without editing archives.
    assert report['source_records'] == len(points)
    assert report['noise_records'] == 1
    with np.load(tmp_path / 'simulation-models/source-partition.npz') as partition:
        assert partition['labels'][-1] == 5
        assert np.isin(partition['labels'][ground_support(tmp_path)], [1, 2]).all()
    assert report['unresolved_records'] == 0
    assert report['ground_cells'] > 0
    assert source.read_bytes() == original
