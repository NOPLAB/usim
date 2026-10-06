"""MIT tooling: acquire unchanged fuRo PCD assets (CC BY-NC-SA 4.0) using stdlib."""

from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
from http.cookiejar import CookieJar
import json
import os
from pathlib import Path
import tempfile
from typing import TypedDict
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.request import HTTPCookieProcessor, build_opener


class Asset(TypedDict):
    name: str
    id: str
    sha256: str
    bytes: int


class Form(TypedDict):
    action: str | None
    method: str
    inputs: list[tuple[str, str]]


ASSETS: dict[str, Asset] = {
    '2018': {
        'name': 'map_tc18_o085_f-04_t30.pcd',
        'id': '1c7Vd4vkMudAHyxc0ZOZCbTgx8ZFZ_Slx',
        'sha256': '6927c661dfcbc50317f3ad61907f4a20088a8bc3111767f0413e02af33a75f4e',
        'bytes': 544071104,
    },
    '2019': {
        'name': 'map_tc19_o085_f-04_t05.pcd',
        'id': '1mH20dXpnBBlQ6hMKJZqdVhphrffsvWK_',
        'sha256': '119250f3b153bd0948b05e441cee30459bed45ce05cdf5749267bc2fc703309d',
        'bytes': 715418112,
    },
}
DEFAULT_OUTPUT = Path('assets/vtc-full/upstream/public-maps')
CHUNK_BYTES = 8 * 1024 * 1024
HTML_LIMIT = 1024 * 1024


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
    raise ValueError('Missing Google Drive download confirmation form')


def verify_file(path: Path, asset: Asset) -> None:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(CHUNK_BYTES):
            digest.update(block)
    if path.stat().st_size != asset['bytes'] or digest.hexdigest() != asset['sha256']:
        raise ValueError(f'Original size/SHA256 mismatch: {path}')


def require_success(response) -> None:
    if response.status != 200:
        raise ValueError(f'Download requires HTTP 200, got {response.status}')


def download(year: str, output_dir: Path) -> dict[str, object]:
    """Verify cached originals or atomically publish a fully verified new original."""
    asset = ASSETS[year]
    destination = output_dir / asset['name']
    source = f'https://drive.google.com/uc?export=download&id={asset["id"]}'
    record: dict[str, object] = {
        'name': asset['name'],
        'path': str(destination.resolve()),
        'source': source,
        'official_source': f'https://drive.google.com/file/d/{asset["id"]}/view',
        'provider': 'fuRo',
        'license': 'CC BY-NC-SA 4.0',
        'license_url': 'https://creativecommons.org/licenses/by-nc-sa/4.0/',
        'bytes': asset['bytes'],
        'sha256': asset['sha256'],
    }
    if destination.exists():
        verify_file(destination, asset)
        return {**record, 'cached': True}

    output_dir.mkdir(parents=True, exist_ok=True)
    opener = build_opener(HTTPCookieProcessor(CookieJar()))
    response = opener.open(source, timeout=60)
    try:
        require_success(response)
        if response.headers.get_content_type() == 'text/html':
            html = response.read(HTML_LIMIT + 1)
            if len(html) > HTML_LIMIT:
                raise ValueError('Google Drive confirmation page exceeds size limit')
            resolved = confirmation_url(html.decode('utf-8'), response.geturl())
            response.close()
            response = opener.open(resolved, timeout=60)
        require_success(response)
        if response.headers.get_content_type() == 'text/html':
            raise ValueError('Google Drive returned HTML instead of the original PCD')
        declared = response.headers.get('Content-Length')
        if declared is not None and int(declared) != asset['bytes']:
            raise ValueError('HTTP Content-Length mismatch with original size')

        partial = None
        try:
            digest = hashlib.sha256()
            count = 0
            with tempfile.NamedTemporaryFile(
                mode='wb',
                dir=output_dir,
                prefix=asset['name'] + '.',
                suffix='.partial',
                delete=False,
            ) as stream:
                partial = Path(stream.name)
                while block := response.read(CHUNK_BYTES):
                    stream.write(block)
                    digest.update(block)
                    count += len(block)
                if count != asset['bytes'] or digest.hexdigest() != asset['sha256']:
                    raise ValueError(f'Original size/SHA256 mismatch: {asset["name"]}')
                stream.flush()
                os.fsync(stream.fileno())
            record.update(
                {
                    'resolved_url': response.geturl(),
                    'response_headers': dict(response.headers.items()),
                    'cached': False,
                }
            )
            os.replace(partial, destination)
        finally:
            if partial is not None:
                partial.unlink(missing_ok=True)
    finally:
        response.close()
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('years', nargs='*', choices=tuple(ASSETS), default=None)
    args = parser.parse_args(argv)
    for year in args.years or ASSETS:
        print(json.dumps(download(year, args.output_dir), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
