"""Canonical provenance for the unchanged public fuRo VTC point clouds."""

from __future__ import annotations

from typing import Final, TypedDict


class Asset(TypedDict):
    name: str
    id: str
    sha256: str
    bytes: int


PROVIDER: Final = 'fuRo'
LICENSE: Final = 'CC-BY-NC-SA-4.0'
LICENSE_NAME: Final = 'CC BY-NC-SA 4.0'
LICENSE_URL: Final = 'https://creativecommons.org/licenses/by-nc-sa/4.0/'
ATTRIBUTION: Final = (
    'fuRo; Yoshitaka Hara and Masahiro Tomono, Moving Object Removal and Surface Mesh '
    'Mapping for Path Planning on 3D Terrain, Advanced Robotics 34(6), 375-387, 2020; '
    'https://doi.org/10.1080/01691864.2020.1717375'
)
CATALOG: Final = 'https://github.com/tsukubachallenge/tc-datasets'
ASSETS: Final[dict[str, Asset]] = {
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


def official_url(file_id: str) -> str:
    return f'https://drive.google.com/file/d/{file_id}/view'


def download_url(file_id: str) -> str:
    return f'https://drive.google.com/uc?export=download&id={file_id}'


SOURCE_URLS: Final = {year: official_url(asset['id']) for year, asset in ASSETS.items()}
PUBLIC_HASHES: Final = {asset['name']: asset['sha256'] for asset in ASSETS.values()}
