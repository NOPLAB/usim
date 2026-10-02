"""Offline provenance, allowlist and archive boundary regressions."""

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'robots' / 'fetch.py'
spec = importlib.util.spec_from_file_location('robot_sources_fetch', SCRIPT)
assert spec is not None and spec.loader is not None
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)

PINS = {
    'turtlebot3': (
        'ROBOTIS-GIT/turtlebot3',
        '90a68bd2e3c61c12966779da89d8eeaec82730e9',
        '8f110df576bb5f346e04550061f82edfa67fed0cc2d19a79772315adc92606a9',
        8862240,
    ),
    'rosbot': (
        'husarion/rosbot_ros',
        '5a42ca181fbac0b5eb1020795282785561a473f4',
        '7a05eba3ab3628fd2196e413f0dcbaf0799a91a8812c91a88a8fc27739fc0e58',
        5801431,
    ),
    'pmb2': (
        'pal-robotics/pmb2_robot',
        'e6584136f15356b8f4944c25d70fb8689e3355af',
        '4470b4892dcc926473c1d2a14e30ccac1f3f818156c7da14e3e1647f790d1273',
        2977359,
    ),
    'orne-box': (
        'open-rdc/orne-box',
        '4d9815f1f363dbb2ea837cc51f8d96a87b6a60bb',
        '7193f28edd33070cd862220313374e0302f8760939bc130b5e0cf66838169e5b',
        3710896,
    ),
    'clearpath': (
        'clearpathrobotics/clearpath_common',
        '5c5ec97ee0245d8543aeb29115e01d3c0f38900e',
        '40faadf4569a9f25f55c67739f13b1f0d3215d64f9486a2543833310f2cd9b84',
        20252477,
    ),
    'husarion_components_description': (
        'husarion/husarion_components_description',
        '5f783f89961bb16098184f5381b1a76058cec19e',
        '8f855cfa1df42ef0692a5787d7b39c857de5017486a9dcfb99578dee69d59492',
        3572783,
    ),
    'pal_urdf_utils': (
        'pal-robotics/pal_urdf_utils',
        '171a1fe521e42eaf2b6cff71d5339551feaa0191',
        '6edf4a155c325e365dcb64388110bf417f204d8b6c82e686e55c0653fb199193',
        7387273,
    ),
}


def test_fixed_allowlist_and_audited_pins():
    manifest = fetch.load_manifest()
    assert manifest['schema_version'] == 1
    assert manifest['audit_date'] == '2026-10-02'
    assert set(manifest['robots']) == {
        'turtlebot3',
        'rosbot',
        'pmb2',
        'orne-box',
        'jackal',
        'husky',
    }
    assert set(manifest['sources']) == set(PINS)
    for name, (repo, commit, digest, size) in PINS.items():
        source = manifest['sources'][name]
        assert source['repository'] == f'https://github.com/{repo}'
        assert source['archive_url'] == f'https://codeload.github.com/{repo}/tar.gz/{commit}'
        assert source['commit'] == commit
        assert source['sha256'] == digest
        assert source['archive_bytes'] == size
        assert source['archive_root'] == f'{repo.split("/")[1]}-{commit}'
        assert len(source['license_files']) == 1
        license_hash = next(iter(source['license_files'].values()))
        assert len(license_hash) == 64
    assert manifest['robots']['jackal']['source'] == manifest['robots']['husky']['source']
    assert manifest['sources']['clearpath']['non_endorsement'] is True
    assert manifest['sources']['clearpath']['header_files'] == [
        'clearpath_platform_description/urdf/common.urdf.xacro'
    ]
    assert manifest['robots']['rosbot']['dependencies'][0]['source'] == (
        'husarion_components_description'
    )
    assert manifest['robots']['pmb2']['dependencies'][0]['source'] == 'pal_urdf_utils'


def test_geometry_and_skid_only_pairing():
    robots = fetch.load_manifest()['robots']
    expected = {
        'turtlebot3': {
            'burger': (0.033, 0.160),
            'waffle': (0.033, 0.287),
            'waffle_pi': (0.033, 0.287),
        },
        'rosbot': {'2': (0.0425, 0.192), 'xl': (0.048, 0.248)},
        'pmb2': {'base': (0.0985, 0.4044)},
        'orne-box': {'base': (0.145, 0.4615)},
        'jackal': {'j100': (0.098, 0.37559)},
        'husky': {'a200': (0.1651, 0.555)},
    }
    for name, variants in expected.items():
        assert set(robots[name]['variants']) == set(variants)
        for variant, geometry in variants.items():
            actual = robots[name]['variants'][variant]
            assert (actual['wheel_radius_m'], actual['wheel_separation_m']) == geometry
        count = 2 if name in {'rosbot', 'jackal', 'husky'} else 1
        assert len(robots[name]['left_joints']) == len(robots[name]['right_joints']) == count
        assert set(robots[name]['left_joints']).isdisjoint(robots[name]['right_joints'])
    assert robots['rosbot']['xacro_args']['mecanum'] == 'false'
    assert robots['rosbot']['left_joints'] == ['fl_wheel_joint', 'rl_wheel_joint']
    assert robots['rosbot']['right_joints'] == ['fr_wheel_joint', 'rr_wheel_joint']
    for name in ['jackal', 'husky']:
        assert robots[name]['left_joints'] == ['front_left_wheel_joint', 'rear_left_wheel_joint']
        assert robots[name]['right_joints'] == [
            'front_right_wheel_joint',
            'rear_right_wheel_joint',
        ]


def fixture_archive(tmp_path, extra=()):
    archive = tmp_path / 'fixture.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for name, data in [
            ('root/LICENSE', b'fixture license'),
            ('root/NOTICE.md', b'fixture notice'),
            ('root/description/robot.urdf', b'<robot/>'),
            ('root/description/NOTICE', b'fixture notice'),
            ('root/unselected/robot.urdf', b'not selected'),
        ]:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
        for member in extra:
            tar.addfile(member)
    source: dict[str, Any] = {
        'archive_root': 'root',
        'archive_url': 'https://example.invalid/fixture',
        'commit': '0' * 40,
        'sha256': fetch.checksum(archive),
        'archive_bytes': archive.stat().st_size,
        'license_files': {'LICENSE': hashlib.sha256(b'fixture license').hexdigest()},
        'include': ['description/'],
    }
    return archive, source


def test_selected_files_and_license_are_preserved(tmp_path):
    archive, source = fixture_archive(tmp_path)
    destination = tmp_path / 'out'
    destination.mkdir()
    fetch.extract_archive(archive, destination, source)
    assert set(fetch.inventory(destination)) == {
        'LICENSE',
        'NOTICE.md',
        'description/robot.urdf',
        'description/NOTICE',
    }
    assert (destination / 'LICENSE').read_bytes() == b'fixture license'
    assert (destination / 'NOTICE.md').read_bytes() == b'fixture notice'
    assert (destination / 'description/NOTICE').read_bytes() == b'fixture notice'
    clearpath = fetch.load_manifest()['sources']['clearpath']
    assert fetch.included('clearpath_platform_description/urdf/a200/a200.urdf.xacro', clearpath)
    assert fetch.included('clearpath_platform_description/meshes/j100/j100_base.stl', clearpath)
    assert not fetch.included(
        'clearpath_platform_description/urdf/dd100/dd100.urdf.xacro', clearpath
    )
    assert not fetch.included('clearpath_platform_description/meshes/r100/chassis.dae', clearpath)


@pytest.mark.parametrize(
    'name',
    [
        'root/../escape',
        '../escape',
        '/root/escape',
        'other/escape',
        'root/C:/escape',
        r'root\..\escape',
        'root/description/../../escape',
        'root/./escape',
    ],
)
def test_traversal_is_rejected_before_any_extraction(tmp_path, name):
    archive, source = fixture_archive(tmp_path, [tarfile.TarInfo(name)])
    destination = tmp_path / 'out'
    destination.mkdir()
    with pytest.raises(ValueError, match='unsafe archive path'):
        fetch.extract_archive(archive, destination, source)
    assert not list(destination.iterdir())


@pytest.mark.parametrize('kind', [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE])
def test_links_and_special_files_are_rejected_even_when_unselected(tmp_path, kind):
    member = tarfile.TarInfo('root/unselected/link')
    member.type = kind
    member.linkname = '../../escape'
    archive, source = fixture_archive(tmp_path, [member])
    destination = tmp_path / 'out'
    destination.mkdir()
    with pytest.raises(ValueError, match='unsafe archive member type'):
        fetch.extract_archive(archive, destination, source)
    assert not list(destination.iterdir())


def test_duplicate_paths_rejected(tmp_path):
    archive, source = fixture_archive(tmp_path, [tarfile.TarInfo('root/description/robot.urdf')])
    with pytest.raises(ValueError, match='duplicate archive path'):
        fetch.extract_archive(archive, tmp_path / 'out', source)


def test_wrong_license_rejected(tmp_path):
    archive, source = fixture_archive(tmp_path)
    source['license_files']['LICENSE'] = '0' * 64
    with pytest.raises(ValueError, match='license checksum mismatch'):
        fetch.extract_archive(archive, tmp_path / 'out', source)


def test_download_hash_failure_leaves_no_source_or_cache_file(tmp_path):
    archive, source = fixture_archive(tmp_path)
    out = tmp_path / 'out'
    with patch.object(fetch.urllib.request, 'urlopen', return_value=io.BytesIO(b'wrong archive')):
        with pytest.raises(ValueError, match='archive checksum mismatch'):
            fetch.fetch_source('fixture', source, out)
    assert not list((out / 'archives').iterdir())
    assert not (out / 'sources').exists()


def test_cached_fetch_is_idempotent_and_detects_edits(tmp_path):
    archive, source = fixture_archive(tmp_path)
    out = tmp_path / 'out'
    with patch.object(
        fetch.urllib.request, 'urlopen', return_value=io.BytesIO(archive.read_bytes())
    ):
        first = fetch.fetch_source('fixture', source, out)
    with patch.object(fetch.urllib.request, 'urlopen', side_effect=AssertionError('network')):
        assert fetch.fetch_source('fixture', source, out) == first
    assert (
        first['files']['description/robot.urdf']['sha256']
        == hashlib.sha256(b'<robot/>').hexdigest()
    )
    (out / first['path'] / 'description/robot.urdf').write_bytes(b'edited')
    with pytest.raises(ValueError, match='existing source differs'):
        fetch.fetch_source('fixture', source, out)


def test_clearpath_selection_fetches_single_source_and_emits_geometry(tmp_path, capsys):
    with patch.object(fetch, 'fetch_source', return_value={'path': 'sources/pinned'}) as download:
        fetch.main(['jackal', 'husky', '--out', str(tmp_path)])
    assert download.call_count == 1
    assert download.call_args.args[0] == 'clearpath'
    provenance = json.loads((tmp_path / 'provenance.json').read_text())
    assert set(provenance['robots']) == {'jackal', 'husky'}
    assert provenance['import_verified'] is False
    assert json.loads(capsys.readouterr().out) == provenance


def test_rosbot_and_pmb2_selections_fetch_xacro_dependencies(tmp_path):
    with patch.object(fetch, 'fetch_source', return_value={'path': 'sources/pinned'}) as download:
        fetch.main(['rosbot', 'pmb2', '--out', str(tmp_path)])
    assert {call.args[0] for call in download.call_args_list} == {
        'rosbot',
        'pmb2',
        'husarion_components_description',
        'pal_urdf_utils',
    }


def test_cached_source_provenance_cannot_be_changed(tmp_path):
    archive, source = fixture_archive(tmp_path)
    out = tmp_path / 'out'
    with patch.object(
        fetch.urllib.request, 'urlopen', return_value=io.BytesIO(archive.read_bytes())
    ):
        result = fetch.fetch_source('fixture', source, out)
    (out / result['path'] / 'provenance.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='existing source differs'):
        fetch.fetch_source('fixture', source, out)


def test_unknown_robot_fails_without_download():
    with patch.object(fetch, 'fetch_source', side_effect=AssertionError('download')):
        with pytest.raises(SystemExit) as error:
            fetch.main(['dingo'])
    assert error.value.code == 2
