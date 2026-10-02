"""Offline coverage of the audited world boundary, archive safety, and provenance."""

import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import stat
import xml.etree.ElementTree as ET
import zipfile

import pytest


SCRIPT = Path(__file__).parents[1] / 'scripts' / 'worlds' / 'fetch_worlds.py'
SPEC = importlib.util.spec_from_file_location('fetch_worlds', SCRIPT)
assert SPEC is not None and SPEC.loader is not None
worlds = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worlds)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(url):
        raise AssertionError(f'Unexpected network access: {url}')

    monkeypatch.setattr(worlds, 'download', blocked)


def make_zip(files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        for name, data in files:
            if isinstance(name, str):
                info = zipfile.ZipInfo(name)
                # Preserve raw ZIP names: Windows ZipInfo otherwise rewrites backslashes.
                info.filename = name
                info.orig_filename = name
                name = info
            archive.writestr(name, data)
    return buffer.getvalue()


def test_exact_allowlist_and_immutable_source_records():
    expected = {
        'aws-racetrack': 'MIT-0',
        'sonoma-raceway': 'CC0-1.0',
        'aws-small-house': 'MIT-0',
        'aws-bookstore': 'MIT-0',
        'aws-small-warehouse': 'MIT-0',
        'willow-garage': 'CC-BY-3.0',
    }
    assert set(worlds.SOURCES) == set(expected)
    for name, source in worlds.SOURCES.items():
        assert source['license'] == expected[name]
        assert len(bytes.fromhex(source['archive_sha256'])) == 32
        assert len(bytes.fromhex(source['license_sha256'])) == 32
        assert source['archive_size'] > 0
        assert source['attribution']
        assert source['license_url'].startswith('https://')
        assert source['license_path']
        assert source['entrypoints']
        if name != 'sonoma-raceway':
            assert re.fullmatch('[0-9a-f]{40}', source['revision'])
        if 'archive_url' in source:
            assert source['revision'] in source['archive_url']
            assert '/tip/' not in source['archive_url']
            assert '/latest/' not in source['archive_url']
    assert sum(source['archive_size'] for source in worlds.SOURCES.values()) == 265866168


def test_sonoma_version_and_independent_license_evidence():
    source = worlds.SOURCES['sonoma-raceway']
    evidence = source['license_evidence']
    assert source['revision'] == '3'
    assert source['archive_url'].endswith('/3/Sonoma%20Raceway.zip')
    assert evidence['version'] == 3
    assert evidence['license_id'] == 1
    assert evidence['owner'] == 'OpenRobotics'
    assert evidence['license_url'] == source['license_url']
    assert evidence['deed_url'] == source['license_url']
    assert source['license_download_url'] == source['license_url'] + 'legalcode.txt'
    assert source['archive_sha256'] == (
        '36952152637d867b026b5bf12976425773edd85d21c7148f136aa773cfa0445d'
    )


def test_willow_fetch_boundary_is_only_model_and_license():
    source = worlds.SOURCES['willow-garage']
    assert source['archive_kind'] == 'deterministic-subset-zip'
    assert 'archive_url' not in source
    assert len(source['files']) == 15
    assert source['files']['LICENSE']['sha256'] == source['license_sha256']
    assert 'willowgarage/model.config' in source['files']
    assert 'willowgarage/model.sdf' in source['files']
    for name, pin in source['files'].items():
        assert name == 'LICENSE' or name.startswith('willowgarage/')
        assert len(bytes.fromhex(pin['sha256'])) == 32
        assert pin['size'] > 0


@pytest.mark.parametrize('name', ['citysim', 'PLATEAU', 'hospital', '../willow-garage'])
def test_unapproved_source_rejected_before_download(name, tmp_path):
    with pytest.raises(SystemExit) as error:
        worlds.main(['--root', str(tmp_path), name])
    assert error.value.code == 2
    assert not (tmp_path / 'assets').exists()


def test_duplicate_sources_rejected_before_download(tmp_path):
    with pytest.raises(SystemExit) as error:
        worlds.main(['--root', str(tmp_path), 'aws-bookstore', 'aws-bookstore'])
    assert error.value.code == 2
    assert not (tmp_path / 'assets').exists()


def test_list_is_offline(capsys):
    assert worlds.main(['--list']) == 0
    assert len(capsys.readouterr().out.splitlines()) == 6


def test_hash_and_size_verification():
    digest = hashlib.sha256(b'abc').hexdigest()
    assert worlds.verify(b'abc', digest, 3) == b'abc'
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        worlds.verify(b'abd', digest, 3)
    with pytest.raises(ValueError, match='Size mismatch'):
        worlds.verify(b'abc', digest, 4)


def test_corrupted_cache_rejected_without_download(tmp_path):
    source = worlds.SOURCES['aws-racetrack']
    (tmp_path / f'{source["archive_sha256"]}.zip').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='mismatch'):
        worlds.archive_bytes(source, tmp_path)


def test_download_hash_failure_is_not_cached(monkeypatch, tmp_path):
    monkeypatch.setattr(worlds, 'download', lambda url: b'corrupt')
    cache = tmp_path / 'cache'
    with pytest.raises(ValueError, match='mismatch'):
        worlds.archive_bytes(worlds.SOURCES['aws-bookstore'], cache)
    assert not cache.exists()


@pytest.mark.parametrize(
    'name',
    [
        '../LICENSE',
        '/absolute/model.sdf',
        'root/models/../../escape',
        'root/models/../../../escape',
        r'root\models\escape',
        'C:/model.sdf',
        'root/models/C:escape',
    ],
)
def test_archive_traversal_rejected_before_extraction(name, tmp_path):
    source = {'archive_root': 'root', 'include': ['models/']}
    data = make_zip([('root/models/valid.sdf', b'valid'), (name, b'invalid')])
    with pytest.raises(ValueError, match='Unsafe archive path'):
        worlds.extract(data, source, tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('mode', [stat.S_IFLNK | 0o777, stat.S_IFIFO | 0o644])
def test_archive_links_and_special_files_rejected(mode, tmp_path):
    info = zipfile.ZipInfo('models/link')
    info.create_system = 3
    info.external_attr = mode << 16
    data = make_zip([(info, b'../../outside')])
    with pytest.raises(ValueError, match='Non-regular'):
        worlds.extract(data, {'archive_root': '', 'include': ['models/']}, tmp_path)
    assert not list(tmp_path.iterdir())


def test_case_colliding_archive_members_rejected(tmp_path):
    data = make_zip([('models/a', b'a'), ('models/A', b'b')])
    with pytest.raises(ValueError, match='Duplicate'):
        worlds.extract(data, {'archive_root': '', 'include': ['models/']}, tmp_path)
    assert not list(tmp_path.iterdir())


def test_aws_resource_selection_excludes_code_and_unrelated_roots(tmp_path):
    source = worlds.SOURCES['aws-small-house']
    root = source['archive_root']
    data = make_zip(
        [
            (f'{root}/LICENSE', b'license'),
            (f'{root}/README.md', b'readme'),
            (f'{root}/models/chair/model.sdf', b'chair'),
            (f'{root}/models/chair/LICENSE', b'nested notice'),
            (f'{root}/worlds/small_house.world', b'world'),
            (f'{root}/launch/world.launch', b'launch'),
            (f'{root}/photos/house.png', b'photo'),
            (f'{root}/maps/map.pgm', b'map'),
            (f'{root}/models-unreviewed/model.sdf', b'excluded'),
            ('different-root/models/model.sdf', b'excluded'),
        ]
    )
    worlds.extract(data, source, tmp_path)
    assert {
        path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob('*') if path.is_file()
    } == {
        'LICENSE',
        'README.md',
        'models/chair/model.sdf',
        'models/chair/LICENSE',
        'worlds/small_house.world',
    }


def test_model_layout_and_excluded_models(tmp_path):
    source = worlds.SOURCES['willow-garage']
    data = make_zip(
        [
            ('LICENSE', b'license'),
            ('willowgarage/model.sdf', b'office'),
            ('hospital/model.sdf', b'excluded'),
            ('citysim/model.sdf', b'excluded'),
        ]
    )
    worlds.extract(data, source, tmp_path)
    assert (tmp_path / 'models/willowgarage/model.sdf').read_bytes() == b'office'
    assert not (tmp_path / 'hospital').exists()
    assert not (tmp_path / 'citysim').exists()
    sonoma = worlds.SOURCES['sonoma-raceway']
    assert worlds.install_path('model.sdf', sonoma) == Path('models/sonoma_raceway/model.sdf')
    assert worlds.install_path('thumbnails/1.png', sonoma) is None


def test_subset_archive_is_deterministic_and_fetches_only_pinned_files(monkeypatch):
    source = copy.deepcopy(worlds.SOURCES['willow-garage'])
    content = {name: name.encode() for name in source['files']}
    for name, data in content.items():
        source['files'][name] = {'sha256': hashlib.sha256(data).hexdigest(), 'size': len(data)}
    calls = []
    prefix = f'https://raw.githubusercontent.com/osrf/gazebo_models/{source["revision"]}/'

    def fake_download(url):
        assert url.startswith(prefix)
        name = url.removeprefix(prefix)
        calls.append(name)
        return content[name]

    monkeypatch.setattr(worlds, 'download', fake_download)
    data = worlds.subset_archive(source)
    assert data == worlds.subset_archive(source)
    assert calls == sorted(source['files']) * 2
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert archive.namelist() == sorted(source['files'])
        for info in archive.infolist():
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert info.compress_type == zipfile.ZIP_STORED
            assert info.external_attr >> 16 == 0o100644
            assert archive.read(info) == content[info.filename]


def test_subset_file_hash_failure(monkeypatch):
    monkeypatch.setattr(worlds, 'download', lambda url: b'changed')
    with pytest.raises(ValueError, match='mismatch'):
        worlds.subset_archive(worlds.SOURCES['willow-garage'])


@pytest.fixture
def tiny_source():
    source = copy.deepcopy(worlds.SOURCES['aws-bookstore'])
    root = source['archive_root']
    license_data = b'fixture license'
    data = make_zip(
        [
            (f'{root}/LICENSE', license_data),
            (f'{root}/models/shelf/model.sdf', b'model'),
            (f'{root}/worlds/bookstore.world', b'world'),
            (f'{root}/photos/excluded.png', b'excluded'),
        ]
    )
    source['archive_size'] = len(data)
    source['archive_sha256'] = hashlib.sha256(data).hexdigest()
    source['license_sha256'] = hashlib.sha256(license_data).hexdigest()
    return source, data


def test_install_provenance_and_existing_install_protection(monkeypatch, tmp_path, tiny_source):
    source, data = tiny_source
    monkeypatch.setitem(worlds.SOURCES, 'aws-bookstore', source)
    monkeypatch.setattr(worlds, 'download', lambda url: data)
    destination = worlds.install('aws-bookstore', tmp_path)
    assert destination == tmp_path / 'assets/worlds/aws-bookstore'
    provenance = json.loads((destination / 'PROVENANCE.json').read_text())
    assert provenance['source'] == source
    assert provenance['installed_size'] == sum(pin['size'] for pin in provenance['files'].values())
    for name, pin in provenance['files'].items():
        installed = (destination / name).read_bytes()
        assert pin == {'sha256': hashlib.sha256(installed).hexdigest(), 'size': len(installed)}
    assert 'ATTRIBUTION.txt' in provenance['files']
    assert not (destination / 'photos').exists()
    monkeypatch.setattr(worlds, 'download', lambda url: pytest.fail('Reinstall must be offline'))
    assert worlds.install('aws-bookstore', tmp_path) == destination
    assert json.loads((destination / 'PROVENANCE.json').read_text()) == provenance
    (destination / 'worlds/bookstore.world').write_bytes(b'changed')
    with pytest.raises(ValueError, match='mismatch'):
        worlds.install('aws-bookstore', tmp_path)


def test_invalid_install_never_publishes_partial_directory(monkeypatch, tmp_path, tiny_source):
    source, data = tiny_source
    source['entrypoints'] = ['worlds/missing.world']
    monkeypatch.setitem(worlds.SOURCES, 'aws-bookstore', source)
    monkeypatch.setattr(worlds, 'download', lambda url: data)
    with pytest.raises(ValueError, match='Missing world'):
        worlds.install('aws-bookstore', tmp_path)
    assert [path.name for path in (tmp_path / 'assets/worlds').iterdir()] == ['_archives']


@pytest.mark.parametrize('name', ['sonoma-raceway', 'willow-garage'])
def test_model_wrapper_and_rights_provenance(name, monkeypatch, tmp_path):
    source = copy.deepcopy(worlds.SOURCES[name])
    license_data = b'fixture legal text'
    source['license_sha256'] = hashlib.sha256(license_data).hexdigest()
    requested = []
    if 'license_download_url' in source:

        def fake_license_download(url):
            requested.append(url)
            return license_data

        monkeypatch.setattr(worlds, 'download', fake_license_download)
    else:
        (tmp_path / source['license_path']).write_bytes(license_data)
    worlds.finish_install(source, tmp_path)
    world = ET.parse(tmp_path / source['entrypoints'][0]).getroot().find('world')
    assert world is not None
    assert {include.findtext('uri') for include in world.findall('include')} == {
        'model://sun',
        'model://ground_plane',
        f'model://{source["wrapper_model"]}',
    }
    assert (tmp_path / source['license_path']).read_bytes() == license_data
    if 'license_download_url' in source:
        assert requested == [source['license_download_url']]
        evidence = json.loads((tmp_path / 'LICENSE-METADATA.json').read_text())
        assert evidence == source['license_evidence']
