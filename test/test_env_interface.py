"""Offline common-interface checks using real acquisition and generated artifacts."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

from usim_env import DownloadRequest, GenerateRequest, get_environment
from usim_env.cli import main
from usim_env_pipeline import MapConfig


def test_explicit_registry_and_asset_names():
    assert set(get_environment('vtc').assets) == {'2018', '2019'}
    assert get_environment('tsudanuma').assets == {
        'map': 'map_tsudanuma20_cit.pcd',
        'sensors': 'tsudanuma20_cit_compressed.bag',
    }
    with pytest.raises(ValueError, match='Unknown environment'):
        get_environment('not-a-dataset')
    request = GenerateRequest(Path('source'), Path('output'))
    assert request.config == MapConfig()
    assert request.reference_2018 is None


@pytest.mark.parametrize(
    'arguments', [['--help'], ['download', '--help'], ['generate', '--help'], ['list']]
)
def test_help_and_list_without_site_packages(arguments):
    root = Path(__file__).parents[1]
    environment = {**os.environ, 'PYTHONPATH': str(root / 'src')}
    result = subprocess.run(
        [sys.executable, '-S', '-m', 'usim_env', *arguments],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    if arguments == ['list']:
        records = [json.loads(line) for line in result.stdout.splitlines()]
        assert {record['dataset'] for record in records} == {'vtc', 'tsudanuma'}
        assert all(record['assets'] for record in records)


@pytest.mark.parametrize('selected', [(), ('2018',), ('2019',)])
def test_vtc_cli_downloads_verified_selected_or_default_originals(
    tmp_path, monkeypatch, capsys, selected
):
    from test.test_env_acquire import Response
    from usim_env_vtc import download

    payloads = {key: f'offline original {key}'.encode() for key in ('2018', '2019')}
    assets = {
        key: {
            'name': f'{key}.pcd',
            'id': f'offline-{key}',
            'bytes': len(payload),
            'sha256': hashlib.sha256(payload).hexdigest(),
        }
        for key, payload in payloads.items()
    }
    monkeypatch.setattr(download, 'ASSETS', assets)
    keys = selected or tuple(assets)
    opener = Mock()
    opener.open.side_effect = [Response(payloads[key], length=len(payloads[key])) for key in keys]
    monkeypatch.setattr(download, 'build_opener', Mock(return_value=opener))

    assert main(['download', 'vtc', '--output-dir', str(tmp_path), *selected]) == 0

    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(records) == len(keys)
    for key, record in zip(keys, records):
        path = record['path']
        assert isinstance(path, str)
        assert Path(path).read_bytes() == payloads[key]
        assert record['sha256'] == assets[key]['sha256']
        assert record['license'] == 'CC BY-NC-SA 4.0'
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        name for key in keys for name in (f'{key}.pcd', f'{key}.pcd.acquisition.json')
    )
    for key in keys:
        metadata = json.loads((tmp_path / f'{key}.pcd.acquisition.json').read_text())
        assert metadata['sha256'] == assets[key]['sha256']
        assert metadata['bytes'] == len(payloads[key])
    cached = get_environment('vtc').download(DownloadRequest(tmp_path, selected))
    assert all(record['cached'] for record in cached)


@pytest.mark.parametrize('selected', [(), ('map',), ('sensors',)])
def test_tsudanuma_download_preserves_actual_validated_originals(tmp_path, monkeypatch, selected):
    from test.test_env_acquire import Response, http
    from test.test_tsudanuma_download import bag_bytes, pcd_bytes
    from usim_env_tsudanuma import download

    payloads = {'map': pcd_bytes(), 'sensor': bag_bytes()}
    assets = {
        key: {**asset, 'approximate_bytes': len(payloads[key])}
        for key, asset in download.ASSETS.items()
    }
    assets['map']['points'] = 2
    monkeypatch.setattr(download, 'ASSETS', assets)
    keys = tuple('sensor' if key == 'sensors' else key for key in selected) or tuple(assets)
    http(monkeypatch, *(Response(payloads[key], length=len(payloads[key])) for key in keys))

    records = get_environment('tsudanuma').download(DownloadRequest(tmp_path, selected))

    for key, record in zip(keys, records):
        path = record['path']
        assert isinstance(path, str)
        assert Path(path).read_bytes() == payloads[key]
        assert record['sha256'] == hashlib.sha256(payloads[key]).hexdigest()
        assert record['license'] == 'CC-BY-4.0'
        assert record['validation']
    cached = get_environment('tsudanuma').download(DownloadRequest(tmp_path, selected))
    assert len(cached) == len(keys)
    assert all(record['cached'] for record in cached)


@pytest.mark.parametrize('dataset', ['vtc', 'tsudanuma'])
def test_unknown_assets_rejected_before_any_download(tmp_path, dataset):
    with pytest.raises(ValueError, match='Unknown'):
        get_environment(dataset).download(DownloadRequest(tmp_path / 'output', ('bad',)))
    assert not (tmp_path / 'output').exists()


@pytest.mark.parametrize(
    'arguments',
    [
        ['download', 'unknown', '--output-dir', 'output'],
        ['download', 'vtc'],
        ['download', 'vtc', '--output-dir', 'output', '--bad'],
        ['download', 'vtc', '--output-dir', 'output', 'map'],
        ['generate', 'vtc', '--source', 'source'],
        ['generate', 'vtc', '--source', 'source', '--output', 'output', '--workers', '0'],
        ['generate', 'vtc', '--source', 'source', '--output', 'output', '--halo', 'nan'],
        [
            'generate',
            'tsudanuma',
            '--source',
            'source',
            '--output',
            'output',
            '--reference-2018',
            'older',
        ],
    ],
)
def test_cli_argument_boundaries(arguments):
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2


@pytest.mark.parametrize('use_cli', [False, True])
def test_vtc_generates_real_source_preserving_artifacts(tmp_path, capsys, use_cli):
    pytest.importorskip('numpy')
    pytest.importorskip('open3d')
    pytest.importorskip('scipy')
    pytest.importorskip('pxr')
    from pxr import Usd

    from test.env_pipeline_fixtures import plane_records, write_pcd
    from usim_env_vtc.verify_map import verify

    source, reference, output = tmp_path / 'survey.pcd', tmp_path / 'older.pcd', tmp_path / 'map'
    records = plane_records()
    original = write_pcd(source, records)
    older = write_pcd(reference, records[:3])
    if use_cli:
        assert (
            main(
                [
                    'generate',
                    'vtc',
                    '--source',
                    str(source),
                    '--output',
                    str(output),
                    '--tile-size',
                    '25',
                    '--halo',
                    '2',
                    '--workers',
                    '1',
                    '--reference-2018',
                    str(reference),
                ]
            )
            == 0
        )
        summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    else:
        summary = get_environment('vtc').generate(
            GenerateRequest(source, output, MapConfig(tile_size=25, halo=2, workers=1), reference)
        )
    assert summary == json.loads((output / 'summary.json').read_text())
    assert summary['records'] == len(records)
    assert summary['triangles'] > 0
    assert verify(output)['records'] == len(records)
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['tile_size'] == 25
    assert manifest['halo'] == 2
    assert set(manifest['sources']) == {'2018', '2019'}
    assert (output / manifest['sources']['2019']['path']).read_bytes() == original
    assert (output / manifest['sources']['2018']['path']).read_bytes() == older
    assert Usd.Stage.Open(str(output / 'world.usda'))
