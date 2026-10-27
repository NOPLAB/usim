"""Offline checks for the complete published Tsudanuma original registry."""

import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
from unittest.mock import Mock

import pytest

from test.test_env_acquire import Response, http
from usim_env_tsudanuma import download, sources


def pcd_bytes(points=2):
    header = (
        'VERSION .7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\n'
        f'COUNT 1 1 1 1\nWIDTH {points}\nHEIGHT 1\nPOINTS {points}\nDATA binary\n'
    )
    return header.encode('ascii') + bytes(points * 16)


def bag_bytes():
    fields = [
        b'op=\x03',
        b'index_pos=' + struct.pack('<Q', 100),
        b'conn_count=' + struct.pack('<I', 1),
        b'chunk_count=' + struct.pack('<I', 1),
    ]
    body = b''.join(struct.pack('<I', len(field)) + field for field in fields)
    return b'#ROSBAG V2.0\n' + struct.pack('<I', len(body)) + body + bytes(64)


def test_registry_has_all_and_only_published_originals():
    assert set(sources.ASSETS) == {'map', 'sensor'}
    assert sources.ASSETS['map']['id'] == '19HeoDbAmwfxLW40NhZFPQGvRZKuip62B'
    assert sources.ASSETS['map'].get('points') == 13_583_284
    assert sources.ASSETS['sensor']['id'] == '1SBKNJ2NDwQlhbN75WJcMppTSwlJu1s4g'
    assert sources.ASSETS['sensor']['format'] == 'rosbag'
    assert sources.LICENSE == 'CC-BY-4.0'
    assert all('bytes' not in asset and 'sha256' not in asset for asset in sources.ASSETS.values())


@pytest.mark.parametrize('key,payload', [('map', pcd_bytes()), ('sensor', bag_bytes())])
def test_adapter_records_observed_originals_and_provenance(tmp_path, monkeypatch, key, payload):
    asset = dict(sources.ASSETS[key])
    asset['approximate_bytes'] = len(payload)
    if key == 'map':
        asset['points'] = 2
    monkeypatch.setattr(download, 'ASSETS', {key: asset})
    http(monkeypatch, Response(payload, length=len(payload)))
    record = download.download(key, tmp_path)
    assert record['bytes'] == len(payload)
    assert record['sha256'] == hashlib.sha256(payload).hexdigest()
    assert record['license'] == sources.LICENSE
    assert record['provider'] == sources.PROVIDER
    assert record['catalog_commit'] == sources.CATALOG_COMMIT
    assert record['official_source'] == sources.SOURCE_URLS[key]
    assert record['catalog_asset'] == asset
    path = record['path']
    assert isinstance(path, str)
    assert Path(path).read_bytes() == payload
    cached = download.download(key, tmp_path)
    assert cached['cached']
    assert cached['validation'] == record['validation']


def test_pcd_count_and_payload_checked(tmp_path):
    path = tmp_path / 'map.pcd'
    path.write_bytes(pcd_bytes())
    assert download.validate_pcd(path, 2)['points'] == 2
    mismatched_catalog = download.validate_pcd(path, 3)
    assert mismatched_catalog['catalog_points'] == 3
    assert not mismatched_catalog['catalog_points_match']
    path.write_bytes(pcd_bytes().replace(b'WIDTH 2', b'WIDTH 3'))
    with pytest.raises(ValueError, match='point count'):
        download.validate_pcd(path, 2)
    path.write_bytes(pcd_bytes()[:-1])
    with pytest.raises(ValueError, match='payload length'):
        download.validate_pcd(path, 2)


def test_rosbag_header_and_index_checked(tmp_path):
    path = tmp_path / 'sensor.bag'
    path.write_bytes(bag_bytes())
    assert download.validate_rosbag(path) == {
        'magic': '#ROSBAG V2.0',
        'index_pos': 100,
        'conn_count': 1,
        'chunk_count': 1,
    }
    path.write_bytes(b'not a ROS bag')
    with pytest.raises(ValueError, match='bag header'):
        download.validate_rosbag(path)
    path.write_bytes(bag_bytes()[:95])
    with pytest.raises(ValueError, match='index'):
        download.validate_rosbag(path)


def test_cli_checks_combined_disk_requirement_before_download(tmp_path, monkeypatch):
    require = Mock(side_effect=OSError('Insufficient disk space'))
    transfer = Mock()
    monkeypatch.setattr(download, 'require_space', require)
    monkeypatch.setattr(download, 'download', transfer)
    with pytest.raises(OSError, match='Insufficient disk space'):
        download.main(['--output-dir', str(tmp_path)])
    assert require.call_args.args == (tmp_path, 57_490_800_000)
    transfer.assert_not_called()


def test_cli_selected_map_and_stdlib_help(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(download, 'require_space', Mock())
    monkeypatch.setattr(download, 'download', Mock(return_value={'bytes': 1}))
    assert download.main(['--output-dir', str(tmp_path), 'map']) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == 'START map'
    assert json.loads(lines[1].removeprefix('COMPLETE ')) == {'bytes': 1}
    script = Path(download.__file__)
    result = subprocess.run(
        [sys.executable, '-S', str(script), '--help'],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_cli_rejects_unknown_asset():
    with pytest.raises(SystemExit) as error:
        download.main(['unknown'])
    assert error.value.code == 2


def test_ascii_original_verified_against_header_not_stale_catalog(tmp_path):
    path = tmp_path / 'map.pcd'
    header = pcd_bytes().split(b'DATA binary\n')[0] + b'DATA ascii\n'
    path.write_bytes(header + b'1 2 3 4\n5 6 7 8\n')
    record = download.validate_pcd(path, 999)
    assert record['points'] == 2
    assert not record['catalog_points_match']
    path.write_bytes(header + b'1 2 3 4\n')
    with pytest.raises(ValueError, match='ASCII point count'):
        download.validate_pcd(path, 999)
    path.write_bytes(header + b'1 2 3\n5 6 7 8\n')
    with pytest.raises(ValueError, match='ASCII field count'):
        download.validate_pcd(path, 999)
