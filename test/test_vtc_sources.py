"""Shared fuRo provenance stays consistent across source adapters."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from test.env_pipeline_fixtures import plane_records, write_pcd
from usim_env_vtc import pcd, sources
from usim_env_vtc.reconstruct import vtc_source


@pytest.mark.parametrize('year', ['2018', '2019'])
def test_official_source_metadata_uses_pinned_registry(year):
    asset = sources.ASSETS[year]
    source = vtc_source(Path(asset['name']), year)

    assert source.key == year
    assert source.expected_sha256 == asset['sha256']
    assert source.strict_xyzinormal
    assert source.license == 'CC-BY-NC-SA-4.0'
    assert source.attribution == sources.ATTRIBUTION
    assert source.catalog == 'https://github.com/tsukubachallenge/tc-datasets'
    assert urlsplit(source.url).path == f'/file/d/{asset["id"]}/view'
    assert parse_qs(urlsplit(sources.download_url(asset['id'])).query) == {
        'export': ['download'],
        'id': [asset['id']],
    }
    assert pcd.PUBLIC_HASHES[asset['name']] == asset['sha256']


def test_mutable_public_hash_export_updates_reconstruction(monkeypatch):
    name = sources.ASSETS['2019']['name']
    digest = 'a' * 64
    monkeypatch.setitem(pcd.PUBLIC_HASHES, name, digest)

    assert sources.PUBLIC_HASHES[name] == digest
    assert vtc_source(Path(name), '2019').expected_sha256 == digest
    assert vtc_source(Path('survey.pcd'), '2019').expected_sha256 is None


def test_registry_provenance_survives_generation_and_independent_verification(
    tmp_path, monkeypatch
):
    pytest.importorskip('open3d')
    pytest.importorskip('scipy')
    pytest.importorskip('pxr')
    from usim_env_vtc.reconstruct import run
    from usim_env_vtc.verify_map import verify

    paths, originals = {}, {}
    for year in sources.ASSETS:
        name = sources.ASSETS[year]['name']
        paths[year] = tmp_path / name
        originals[year] = write_pcd(paths[year], plane_records()[:3])
        monkeypatch.setitem(
            sources.PUBLIC_HASHES, name, hashlib.sha256(originals[year]).hexdigest()
        )
    output = tmp_path / 'map'

    run(
        argparse.Namespace(
            source=paths['2019'],
            reference_2018=paths['2018'],
            output=output,
            tile_size=30.0,
            halo=1.5,
            workers=1,
        )
    )
    assert verify(output)['records'] == 3
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['license'] == sources.LICENSE
    assert manifest['attribution'] == sources.ATTRIBUTION
    assert manifest['source_catalog'] == sources.CATALOG
    for year, item in manifest['sources'].items():
        assert item['official_url'] == sources.SOURCE_URLS[year]
        assert item['license'] == sources.LICENSE
        assert item['sha256'] == sources.PUBLIC_HASHES[paths[year].name]
        assert (output / item['path']).read_bytes() == originals[year]

    monkeypatch.setitem(sources.PUBLIC_HASHES, paths['2018'].name, '0' * 64)
    with pytest.raises(AssertionError, match='checksum differs'):
        verify(output)
