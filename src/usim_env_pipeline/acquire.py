"""Stream and verify unchanged upstream assets without dataset-specific policy."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
from html.parser import HTMLParser
from http.cookiejar import CookieJar
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import TypeAlias, TypedDict
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.request import HTTPCookieProcessor, OpenerDirector, build_opener

from .models import PipelineInputError

JSONValue: TypeAlias = None | bool | int | float | str | list['JSONValue'] | dict[str, 'JSONValue']

CHUNK_BYTES = 8 * 1024 * 1024
HTML_LIMIT = 1024 * 1024
HEADROOM_BYTES = 1024 * 1024 * 1024


class Form(TypedDict):
    action: str | None
    method: str
    inputs: list[tuple[str, str]]


class ConfirmationForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.forms: list[Form] = []
        self.current: Form | None = None

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == 'form':
            self.current = {
                'action': attributes.get('action'),
                'method': (attributes.get('method') or 'get').lower(),
                'inputs': [],
            }
        elif tag == 'input' and self.current is not None and attributes.get('name'):
            self.current['inputs'].append(
                (attributes['name'] or '', attributes.get('value') or '')
            )

    def handle_endtag(self, tag):
        if tag == 'form' and self.current is not None:
            self.forms.append(self.current)
            self.current = None


def confirmation_url(html: str, base_url: str) -> str:
    parser = ConfirmationForm()
    parser.feed(html)
    for form in parser.forms:
        if form['action'] and form['method'] == 'get':
            parts = urlsplit(urljoin(base_url, form['action']))
            query = dict(parse_qsl(parts.query))
            query.update(form['inputs'])
            if query.get('id') and query.get('confirm'):
                return urlunsplit(parts._replace(query=urlencode(query)))
    raise PipelineInputError('Missing Google Drive download confirmation form')


def download_url(file_id: str) -> str:
    return f'https://drive.google.com/uc?export=download&id={file_id}'


def official_url(file_id: str) -> str:
    return f'https://drive.google.com/file/d/{file_id}/view'


def require_space(directory: Path, required_bytes: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(directory).free
    if free < required_bytes + HEADROOM_BYTES:
        raise OSError(
            f'Insufficient disk space: {free} bytes free; '
            f'{required_bytes + HEADROOM_BYTES} bytes required including headroom'
        )


def verify_file(path: Path, expected_bytes: int, expected_sha256: str) -> None:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(CHUNK_BYTES):
            digest.update(block)
    if path.stat().st_size != expected_bytes or digest.hexdigest() != expected_sha256:
        raise PipelineInputError(f'Original size/SHA256 mismatch: {path}')


def _write_record(path: Path, record: Mapping[str, JSONValue]) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', dir=path.parent, suffix='.partial', delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def acquire(
    source_url: str,
    destination: Path,
    *,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
    validate: Callable[[Path], Mapping[str, JSONValue]] | None = None,
    metadata: Mapping[str, JSONValue] | None = None,
    progress: Callable[[int, int | None], None] | None = None,
    opener: OpenerDirector | None = None,
) -> dict[str, JSONValue]:
    """Publish complete verified bytes; observed digests are not provider-pinned hashes.

    The sibling ``<name>.acquisition.json`` stores observed bytes, SHA256, response
    provenance and caller metadata. Cached bytes must match that record (and any
    supplied provider-pinned expectations) before being returned.
    """
    destination = Path(destination)
    record_path = destination.with_name(destination.name + '.acquisition.json')
    record: dict[str, JSONValue] = {
        **(metadata or {}),
        'name': destination.name,
        'path': str(destination.resolve()),
        'source': source_url,
    }
    if destination.exists():
        if record_path.exists():
            saved: JSONValue = json.loads(record_path.read_text(encoding='utf-8'))
            if not isinstance(saved, dict):
                raise PipelineInputError(f'Invalid acquisition record: {record_path}')
            if saved['source'] != source_url:
                raise PipelineInputError(f'Cached source URL mismatch: {destination}')
            saved_bytes, saved_sha256 = saved['bytes'], saved['sha256']
            if not isinstance(saved_bytes, int) or not isinstance(saved_sha256, str):
                raise PipelineInputError(f'Invalid acquisition size/SHA256: {record_path}')
            verify_file(destination, saved_bytes, saved_sha256)
            if expected_bytes is not None and saved['bytes'] != expected_bytes:
                raise PipelineInputError(f'Original size mismatch: {destination}')
            if expected_sha256 is not None and saved['sha256'] != expected_sha256:
                raise PipelineInputError(f'Original SHA256 mismatch: {destination}')
            record = {**saved, **record}
        elif expected_bytes is not None and expected_sha256 is not None:
            verify_file(destination, expected_bytes, expected_sha256)
            record.update(bytes=expected_bytes, sha256=expected_sha256)
        else:
            raise PipelineInputError(f'Cached original has no acquisition record: {destination}')
        if validate is not None:
            record['validation'] = dict(validate(destination))
        return {**record, 'cached': True}

    destination.parent.mkdir(parents=True, exist_ok=True)
    if opener is None:
        opener = build_opener(HTTPCookieProcessor(CookieJar()))
    response = opener.open(source_url, timeout=60)
    partial = None
    try:
        if response.status != 200:
            raise PipelineInputError(f'Download requires HTTP 200, got {response.status}')
        if response.headers.get_content_type() == 'text/html':
            html = response.read(HTML_LIMIT + 1)
            if len(html) > HTML_LIMIT:
                raise PipelineInputError('Google Drive confirmation page exceeds size limit')
            resolved = confirmation_url(html.decode('utf-8'), response.geturl())
            response.close()
            response = opener.open(resolved, timeout=60)
        if response.status != 200:
            raise PipelineInputError(f'Download requires HTTP 200, got {response.status}')
        if response.headers.get_content_type() == 'text/html':
            raise PipelineInputError('Server returned HTML instead of original bytes')
        declared = response.headers.get('Content-Length')
        length = int(declared) if declared is not None else None
        if expected_bytes is not None and length is not None and length != expected_bytes:
            raise PipelineInputError('HTTP Content-Length mismatch with original size')
        require_space(destination.parent, length or expected_bytes or 0)
        digest = hashlib.sha256()
        count = 0
        with tempfile.NamedTemporaryFile(
            mode='wb',
            dir=destination.parent,
            prefix=destination.name + '.',
            suffix='.partial',
            delete=False,
        ) as stream:
            partial = Path(stream.name)
            while block := response.read(CHUNK_BYTES):
                if count == 0 and block.lstrip().lower().startswith((b'<!doctype html', b'<html')):
                    raise PipelineInputError('Server returned HTML instead of original bytes')
                stream.write(block)
                digest.update(block)
                count += len(block)
                if progress is not None:
                    progress(count, length)
            if length is not None and count != length:
                raise PipelineInputError('HTTP Content-Length mismatch with received bytes')
            if expected_bytes is not None and count != expected_bytes:
                raise PipelineInputError('Original size mismatch')
            if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
                raise PipelineInputError('Original SHA256 mismatch')
            stream.flush()
            os.fsync(stream.fileno())
        if validate is not None:
            record['validation'] = dict(validate(partial))
        record.update(
            bytes=count,
            sha256=digest.hexdigest(),
            resolved_url=response.geturl(),
            response_headers=dict(response.headers.items()),
            cached=False,
        )
        _write_record(record_path, record)
        os.replace(partial, destination)
    finally:
        response.close()
        if partial is not None:
            partial.unlink(missing_ok=True)
    return record
