"""Offline integration tests for streaming upstream acquisition."""

from email.message import Message
import hashlib
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit
from urllib.request import build_opener

import pytest

from usim_env_pipeline import acquire


class Response(io.BytesIO):
    def __init__(self, body, *, content_type='application/octet-stream', length=None, status=200):
        super().__init__(body)
        self.status = status
        self.headers = Message()
        self.headers['Content-Type'] = content_type
        if length is not None:
            self.headers['Content-Length'] = str(length)

    def geturl(self):
        return 'https://drive.usercontent.google.com/download'


def http(monkeypatch, *responses):
    opener = Mock()
    opener.open.side_effect = responses
    monkeypatch.setattr(acquire, 'build_opener', Mock(return_value=opener))
    return opener


def test_confirmation_atomic_stream_record_and_cache(tmp_path, monkeypatch):
    payload = b'original data' * 100
    page = Response(
        b'<form action="/download" method="get"><input name="id" value="source">'
        b'<input name="confirm" value="t"><input name="uuid" value="a&amp;b"></form>',
        content_type='text/html',
    )
    response = Response(payload, length=len(payload))
    opener = http(monkeypatch, page, response)
    monkeypatch.setattr(acquire, 'CHUNK_BYTES', 7)
    destination = tmp_path / 'original.bin'
    events = []

    def validate(path):
        assert not destination.exists()
        assert path.read_bytes() == payload
        return {'kind': 'original'}

    record = acquire.acquire(
        'https://source.example/original',
        destination,
        metadata={'license': 'caller-license', 'catalog': 'caller-catalog'},
        validate=validate,
        progress=lambda count, length: events.append((count, length)),
    )
    assert destination.read_bytes() == payload
    assert record['sha256'] == hashlib.sha256(payload).hexdigest()
    assert record['bytes'] == len(payload)
    assert record['license'] == 'caller-license'
    assert record['validation'] == {'kind': 'original'}
    assert events[-1] == (len(payload), len(payload))
    assert page.closed and response.closed
    assert parse_qs(urlsplit(opener.open.call_args_list[1].args[0]).query) == {
        'id': ['source'],
        'confirm': ['t'],
        'uuid': ['a&b'],
    }
    saved = json.loads(destination.with_name('original.bin.acquisition.json').read_text())
    assert saved == record
    before = destination.stat().st_mtime_ns
    opener.reset_mock()
    cached = acquire.acquire('https://source.example/original', destination)
    assert cached['cached']
    assert cached['sha256'] == record['sha256']
    assert destination.stat().st_mtime_ns == before
    opener.open.assert_not_called()
    destination.write_bytes(payload[:-1] + b'x')
    with pytest.raises(ValueError, match='mismatch'):
        acquire.acquire('https://source.example/original', destination)
    assert destination.exists()
    opener.open.assert_not_called()


@pytest.mark.parametrize('kind', ['short', 'hash', 'html', 'status', 'validator', 'read'])
def test_invalid_transfer_never_publishes(tmp_path, monkeypatch, kind):
    payload = b'original bytes'
    response = Response(
        b'<html>error</html>' if kind == 'html' else payload,
        length=len(payload) + 1 if kind == 'short' else None,
        status=206 if kind == 'status' else 200,
    )
    if kind == 'read':
        response.read = Mock(side_effect=[payload[:2], OSError('connection lost')])
    http(monkeypatch, response)

    def validate(path):
        raise ValueError('Invalid file header')

    with pytest.raises((ValueError, OSError)):
        acquire.acquire(
            'https://source.example/original',
            tmp_path / 'original.bin',
            expected_sha256='0' * 64 if kind == 'hash' else None,
            validate=validate if kind == 'validator' else None,
        )
    assert not list(tmp_path.iterdir())
    assert response.closed


def test_pinned_original_cache_and_length(tmp_path, monkeypatch):
    payload = b'pinned bytes'
    destination = tmp_path / 'original.bin'
    destination.write_bytes(payload)
    opener = http(monkeypatch)
    record = acquire.acquire(
        'https://source.example/original',
        destination,
        expected_bytes=len(payload),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    )
    assert record['cached']
    opener.open.assert_not_called()
    with pytest.raises(ValueError, match='no acquisition record'):
        acquire.acquire('https://source.example/original', destination)


def test_space_check_before_streaming(tmp_path, monkeypatch):
    response = Response(b'bytes', length=5)
    http(monkeypatch, response)
    monkeypatch.setattr(acquire.shutil, 'disk_usage', Mock(return_value=Mock(free=4)))
    with pytest.raises(OSError, match='Insufficient disk space'):
        acquire.acquire('https://source.example/original', tmp_path / 'original.bin')
    assert response.closed
    assert not list(tmp_path.iterdir())


def test_missing_confirmation_form(tmp_path, monkeypatch):
    response = Response(b'<html>quota exceeded</html>', content_type='text/html')
    http(monkeypatch, response)
    with pytest.raises(ValueError, match='confirmation form'):
        acquire.acquire('https://source.example/original', tmp_path / 'original.bin')
    assert response.closed
    assert not list(tmp_path.iterdir())


def test_injected_stdlib_opener_follows_confirmation(tmp_path, monkeypatch):
    payload = b'unchanged original bytes'
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            if self.path == '/original':
                body = (
                    b'<form action="/confirmed"><input name="id" value="original">'
                    b'<input name="confirm" value="t"></form>'
                )
                content_type = 'text/html'
            else:
                body = payload
                content_type = 'application/octet-stream'
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    server = HTTPServer(('127.0.0.1', 0), Handler)
    server.timeout = 5

    def serve():
        server.handle_request()
        server.handle_request()

    worker = Thread(target=serve, daemon=True)
    worker.start()
    monkeypatch.setattr(acquire, 'build_opener', Mock(side_effect=AssertionError))
    destination = tmp_path / 'original.bin'
    source = f'http://127.0.0.1:{server.server_port}/original'
    try:
        record = acquire.acquire(source, destination, opener=build_opener())
    finally:
        worker.join(timeout=10)
        server.server_close()
    assert not worker.is_alive()
    assert requests == ['/original', '/confirmed?id=original&confirm=t']
    assert destination.read_bytes() == payload
    assert record['sha256'] == hashlib.sha256(payload).hexdigest()
    assert record['resolved_url'] == source.replace(
        '/original', '/confirmed?id=original&confirm=t'
    )


def test_cached_source_and_validator_are_rechecked(tmp_path, monkeypatch):
    payload = b'original bytes'
    http(monkeypatch, Response(payload, length=len(payload)))
    destination = tmp_path / 'original.bin'
    source = 'https://source.example/original'
    acquire.acquire(source, destination)
    with pytest.raises(ValueError, match='source URL mismatch'):
        acquire.acquire('https://source.example/replaced', destination)

    validator = Mock(side_effect=ValueError('Invalid file header'))
    with pytest.raises(ValueError, match='Invalid file header'):
        acquire.acquire(source, destination, validate=validator)
    validator.assert_called_once_with(destination)
    assert destination.read_bytes() == payload
