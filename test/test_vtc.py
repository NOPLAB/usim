"""Offline checks for the VTC asset boundary and portable world generation."""

import hashlib
import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest


SCRIPT = Path(__file__).parents[1] / 'scripts' / 'vtc' / 'fetch_vtc_world.py'
SPEC = importlib.util.spec_from_file_location('fetch_vtc_world', SCRIPT)
assert SPEC is not None and SPEC.loader is not None
vtc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vtc)


def test_manifest_pins_only_audited_geometry_and_license():
    assert len(vtc.COMMIT) == 40
    assert set(vtc.MANIFEST) == {
        'Environment/Terrain/fbx/002-ParkingArea.fbx',
        'Environment/Terrain/fbx/005-ParkingArea.fbx',
        'Environment/Terrain/fbx/006-RoadMark.fbx',
        'Environment/Terrain/fbx/ParkArea.fbx',
        'Environment/Terrain/fbx/ParkingArea.fbx',
        'Environment/Terrain/fbx/ParkingArea2.fbx',
        'Environment/Terrain/fbx/SevenElevenArea.fbx',
        'Environment/Terrain/fbx/Tsukuba-Terrain.fbx',
        'LICENSE',
    }
    for path, (digest, size) in vtc.MANIFEST.items():
        assert vtc.allowed_path(path)
        assert len(bytes.fromhex(digest)) == 32
        assert size > 0
    assert '@sha256:' in vtc.BLENDER_IMAGE


@pytest.mark.parametrize(
    'path',
    [
        'Environment/CityHall/CityHall.blend',
        'cityhall.blend',
        'City Hall.fbx',
        'CITY_HALL.blend',
        'Environment/vtc_environment_assy.blend',
        'Resources/point_cloud/TsukubaChallenge_2018.ply',
        'Textures/Asphalt010_4K-JPG/Asphalt010_4K_Color.jpg',
        'Environment/Terrain/TsukubaTerrain.blend',
        'Environment/Terrain/RoadMark.blend',
        '../LICENSE',
    ],
)
def test_unapproved_assets_are_never_downloaded(path):
    assert not vtc.allowed_path(path)


def test_hash_verification_rejects_mismatches(tmp_path):
    path = tmp_path / 'asset'
    path.write_bytes(b'abc')
    digest = hashlib.sha256(b'abc').hexdigest()
    vtc.verify_file(path, digest, 3)
    with pytest.raises(ValueError, match='mismatch'):
        vtc.verify_file(path, '0' * 64, 3)
    with pytest.raises(ValueError, match='mismatch'):
        vtc.verify_file(path, digest, 4)
    path.write_bytes(b'abd')
    with pytest.raises(ValueError, match='mismatch'):
        vtc.verify_file(path, digest, 3)


def test_cached_file_is_verified_before_use(tmp_path):
    path = tmp_path / 'upstream' / 'Environment' / 'Terrain' / 'fbx' / '002-ParkingArea.fbx'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='mismatch'):
        vtc.fetch_assets(tmp_path)


def test_sdf_has_static_mesh_visual_collision_and_ground(tmp_path):
    path = vtc.write_sdf(tmp_path, [{'name': 'mesh_0000', 'file': 'meshes/mesh_0000.dae'}])
    world = ET.parse(path).getroot().find('world')
    assert world is not None
    terrain = world.find("model[@name='mesh_0000']")
    assert terrain is not None
    assert terrain.findtext('static') == 'true'
    for kind in ('visual', 'collision'):
        mesh = terrain.find(f'link/{kind}/geometry/mesh')
        assert mesh is not None
        assert mesh.findtext('uri') == 'meshes/mesh_0000.dae'
        assert mesh.findtext('scale') == '1 1 1'
    ground = world.find("model[@name='ground']")
    assert ground is not None
    assert ground.findtext('pose') == '0 0 -0.05 0 0 0'
    assert ground.find('link/collision/geometry/box') is not None
