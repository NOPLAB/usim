"""MIT tooling: acquire unchanged fuRo PCD assets (CC BY-NC-SA 4.0) using stdlib."""

from __future__ import annotations

import argparse
from http.cookiejar import CookieJar
import json
from pathlib import Path
from urllib.request import HTTPCookieProcessor, build_opener

if not __package__:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'usim_env_vtc'

from usim_env_pipeline.acquire import JSONValue, acquire

from .sources import ASSETS as ASSETS
from .sources import LICENSE_NAME, LICENSE_URL, PROVIDER, download_url, official_url


DEFAULT_OUTPUT = Path('assets/vtc-full/upstream/public-maps')


def download(year: str, output_dir: Path) -> dict[str, JSONValue]:
    """Verify cached originals or atomically publish a fully verified new original."""
    asset = ASSETS[year]
    destination = output_dir / asset['name']
    return acquire(
        download_url(asset['id']),
        destination,
        expected_bytes=asset['bytes'],
        expected_sha256=asset['sha256'],
        metadata={
            'official_source': official_url(asset['id']),
            'provider': PROVIDER,
            'license': LICENSE_NAME,
            'license_url': LICENSE_URL,
        },
        opener=build_opener(HTTPCookieProcessor(CookieJar())),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('years', nargs='*', metavar='{' + ','.join(ASSETS) + '}')
    args = parser.parse_args(argv)
    if any(year not in ASSETS for year in args.years):
        parser.error('years must be selected from ' + ', '.join(ASSETS))
    for year in args.years or ASSETS:
        print(json.dumps(download(year, args.output_dir), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
