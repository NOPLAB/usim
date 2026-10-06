"""Offline integration checks for original public PCD acquisition."""

from email.message import Message
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit

import pytest


SCRIPT = Path(__file__).parents[1] / 'src' / 'usim_env_vtc' / 'download.py'
SPEC = importlib.util.spec_from_file_location('vtc_full_download', SCRIPT)
assert SPEC is not None and SPEC.loader is not None
download = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download)


class Response(io.BytesIO):
    def __init__(
        self,
        body,
        *,
        status=200,
        content_type='application/octet-stream',
        length=None,
        url='https://drive.usercontent.google.com/download',
    ):
        super().__init__(body)
        self.status = status
        self.headers = Message()
        self.headers['Content-Type'] = content_type
        if length is not None:
            self.headers['Content-Length'] = str(length)
        self.url = url
        self.read_sizes = []

    def geturl(self):
        return self.url

    def read(self, size: int | None = -1):
        assert size is not None and size > 0
        self.read_sizes.append(size)
        return super().read(size)


@pytest.fixture
def original(monkeypatch):
    payload = b'FIELDS x y z intensity normal_x normal_y normal_z curvature\n' + bytes(range(128))
    asset = {
        'name': 'map_tc18_o085_f-04_t30.pcd',
        'id': '1c7Vd4vkMudAHyxc0ZOZCbTgx8ZFZ_Slx',
        'bytes': len(payload),
        'sha256': hashlib.sha256(payload).hexdigest(),
    }
    monkeypatch.setattr(download, 'ASSETS', {'2018': asset})
    monkeypatch.setattr(download, 'CHUNK_BYTES', 7)
    return payload, asset


def mock_http(monkeypatch, *responses):
    opener = Mock()
    opener.open.side_effect = responses
    monkeypatch.setattr(download, 'build_opener', Mock(return_value=opener))
    return opener


def test_manifest_pins_original_assets():
    assert download.ASSETS == {
        '2018': {
            'name': 'map_tc18_o085_f-04_t30.pcd',
            'id': '1c7Vd4vkMudAHyxc0ZOZCbTgx8ZFZ_Slx',
            'bytes': 544071104,
            'sha256': '6927c661dfcbc50317f3ad61907f4a20088a8bc3111767f0413e02af33a75f4e',
        },
        '2019': {
            'name': 'map_tc19_o085_f-04_t05.pcd',
            'id': '1mH20dXpnBBlQ6hMKJZqdVhphrffsvWK_',
            'bytes': 715418112,
            'sha256': '119250f3b153bd0948b05e441cee30459bed45ce05cdf5749267bc2fc703309d',
        },
    }


def test_confirmation_stream_verified_before_atomic_publish(tmp_path, monkeypatch, original):
    payload, asset = original
    page = Response(
        b'<form method="get" action="https://drive.usercontent.google.com/download?export=download'
        b'&amp;authuser=0"><input value="source-id" name="id">'
        b'<input name="confirm" value="t"><input value="a&amp;b" name="uuid"></form>',
        content_type='text/html',
        url='https://drive.google.com/uc',
    )
    binary = Response(payload, length=len(payload))
    opener = mock_http(monkeypatch, page, binary)
    destination = tmp_path / asset['name']
    replace = download.os.replace
    published = []

    def publish(source, target):
        assert not destination.exists()
        assert source.parent == tmp_path
        assert source.read_bytes() == payload
        download.verify_file(source, asset)
        published.append(target)
        replace(source, target)

    monkeypatch.setattr(download.os, 'replace', publish)
    record = download.download('2018', tmp_path)

    assert destination.read_bytes() == payload
    assert list(tmp_path.iterdir()) == [destination]
    assert published == [destination]
    assert binary.read_sizes == [7] * (len(payload) // 7 + 2)
    assert page.closed and binary.closed
    query = parse_qs(urlsplit(opener.open.call_args_list[1].args[0]).query)
    assert query == {
        'export': ['download'],
        'authuser': ['0'],
        'id': ['source-id'],
        'confirm': ['t'],
        'uuid': ['a&b'],
    }
    assert record['source'].endswith('id=' + asset['id'])
    assert record['official_source'].endswith('/' + asset['id'] + '/view')
    assert record['resolved_url'] == binary.url
    assert record['sha256'] == asset['sha256']
    assert record['bytes'] == len(payload)
    assert record['license'] == 'CC BY-NC-SA 4.0'
    assert not record['cached']


def test_cached_original_is_verified_without_network_or_rewrite(tmp_path, monkeypatch, original):
    payload, asset = original
    destination = tmp_path / asset['name']
    destination.write_bytes(payload)
    before = destination.stat()
    opener = mock_http(monkeypatch)
    record = download.download('2018', tmp_path)
    assert record['cached']
    assert destination.read_bytes() == payload
    assert destination.stat().st_mtime_ns == before.st_mtime_ns
    assert destination.stat().st_ino == before.st_ino
    opener.open.assert_not_called()


def test_corrupt_cache_is_preserved_and_rejected(tmp_path, monkeypatch, original):
    payload, asset = original
    destination = tmp_path / asset['name']
    corrupt = payload[:-1] + b'\xff'
    destination.write_bytes(corrupt)
    opener = mock_http(monkeypatch)
    with pytest.raises(ValueError, match='mismatch'):
        download.download('2018', tmp_path)
    assert destination.read_bytes() == corrupt
    opener.open.assert_not_called()


@pytest.mark.parametrize('kind', ['hash', 'short', 'oversize', 'read-error'])
def test_failed_stream_never_publishes_and_removes_partial(tmp_path, monkeypatch, original, kind):
    payload, _ = original
    if kind == 'hash':
        payload = payload[:-1] + b'\xff'
    elif kind == 'short':
        payload = payload[:-1]
    elif kind == 'oversize':
        payload += b'\x00'
    response = Response(payload)
    if kind == 'read-error':
        response.read = Mock(side_effect=[payload[:7], OSError('connection lost')])
    mock_http(monkeypatch, response)
    with pytest.raises((ValueError, OSError)):
        download.download('2018', tmp_path)
    assert list(tmp_path.iterdir()) == []
    assert response.closed


@pytest.mark.parametrize('status', [204, 206, 404, 500])
def test_http_status_is_checked_before_read(tmp_path, monkeypatch, original, status):
    payload, _ = original
    response = Response(payload, status=status)
    mock_http(monkeypatch, response)
    with pytest.raises(ValueError, match='HTTP 200'):
        download.download('2018', tmp_path)
    assert response.read_sizes == []
    assert response.closed
    assert list(tmp_path.iterdir()) == []


def test_wrong_content_length_is_rejected_before_stream(tmp_path, monkeypatch, original):
    payload, _ = original
    response = Response(payload, length=len(payload) + 1)
    mock_http(monkeypatch, response)
    with pytest.raises(ValueError, match='Content-Length mismatch'):
        download.download('2018', tmp_path)
    assert response.read_sizes == []
    assert response.closed
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('second_status,second_type', [(403, 'text/html'), (200, 'text/html')])
def test_confirmation_error_is_not_published(
    tmp_path, monkeypatch, original, second_status, second_type
):
    page = Response(
        b'<form action="/download"><input name="id" value="id">'
        b'<input name="confirm" value="t"></form>',
        content_type='text/html',
    )
    response = Response(b'quota exceeded', status=second_status, content_type=second_type)
    mock_http(monkeypatch, page, response)
    with pytest.raises(ValueError):
        download.download('2018', tmp_path)
    assert page.closed and response.closed
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    'html',
    [b'<html>quota exceeded</html>', b'x' * (1024 * 1024 + 1)],
    ids=['missing-form', 'oversized-form'],
)
def test_missing_or_oversized_confirmation_is_rejected(tmp_path, monkeypatch, original, html):
    response = Response(html, content_type='text/html')
    opener = mock_http(monkeypatch, response)
    with pytest.raises(ValueError):
        download.download('2018', tmp_path)
    assert opener.open.call_count == 1
    assert response.closed
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('years', [[], ['2018']])
def test_cli_downloads_selected_or_default_years(tmp_path, monkeypatch, original, capsys, years):
    payload, asset = original
    mock_http(monkeypatch, Response(payload, length=len(payload)))
    assert download.main(['--output-dir', str(tmp_path), *years]) == 0
    record = json.loads(capsys.readouterr().out)
    assert Path(record['path']).read_bytes() == payload
    assert record['sha256'] == asset['sha256']


def test_cli_help_uses_only_standard_library():
    result = subprocess.run(
        [sys.executable, '-S', str(SCRIPT), '--help'],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_default_download_destination_belongs_to_working_directory(
    tmp_path, monkeypatch, original
):
    # Given an installed tool invoked from a checkout rather than its package directory.
    payload, asset = original
    monkeypatch.chdir(tmp_path)
    mock_http(monkeypatch, Response(payload, length=len(payload)))
    # When downloading without an explicit output path.
    assert download.main(['2018']) == 0
    # Then no data is written beneath the installed Python package.
    destination = tmp_path / 'assets/vtc-full/upstream/public-maps' / asset['name']
    assert destination.read_bytes() == payload
