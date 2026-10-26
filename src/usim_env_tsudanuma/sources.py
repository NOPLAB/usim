"""Published Tsudanuma originals; the catalogue provides no exact sizes or hashes."""

from __future__ import annotations

from typing import Final, TypedDict

from usim_env_pipeline.acquire import download_url, official_url


class Original(TypedDict):
    name: str
    id: str
    format: str
    approximate_bytes: int


class Asset(Original, total=False):
    points: int
    date: str
    duration: str


PROVIDER: Final = 'Chiba Institute of Technology'
LICENSE: Final = 'CC-BY-4.0'
LICENSE_URL: Final = 'https://creativecommons.org/licenses/by/4.0/'
ATTRIBUTION: Final = 'Chiba Institute of Technology, Tsudanuma 2020 dataset'
CATALOG: Final = 'https://github.com/tsukubachallenge/tc-datasets'
CATALOG_COMMIT: Final = '832026ff389e94eca9f945a3fe07f5905d0646d7'
CATALOG_URL: Final = f'{CATALOG}/blob/{CATALOG_COMMIT}/README.md'
ASSETS: Final[dict[str, Asset]] = {
    'map': {
        'name': 'map_tsudanuma20_cit.pcd',
        'id': '19HeoDbAmwfxLW40NhZFPQGvRZKuip62B',
        'format': 'pcd',
        'approximate_bytes': 490_800_000,
        'points': 13_583_284,
    },
    'sensor': {
        'name': 'tsudanuma20_cit_compressed.bag',
        'id': '1SBKNJ2NDwQlhbN75WJcMppTSwlJu1s4g',
        'format': 'rosbag',
        'approximate_bytes': 57_000_000_000,
        'date': '2020-08-27 17:43:12',
        'duration': '56:12',
    },
}
SOURCE_URLS: Final = {key: official_url(asset['id']) for key, asset in ASSETS.items()}
DOWNLOAD_URLS: Final = {key: download_url(asset['id']) for key, asset in ASSETS.items()}
