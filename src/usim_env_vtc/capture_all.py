"""Render every map region in fresh Isaac processes and audit payload coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', type=Path, required=True, help='Isaac environment interpreter')
    parser.add_argument('--world', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--grid', type=int, default=8)
    args = parser.parse_args()
    if args.grid < 1:
        parser.error('grid must be positive')
    args.out.mkdir(parents=True, exist_ok=True)
    entries = json.loads(args.index.read_text(encoding='utf-8'))
    expected = {entry['path'] for entry in entries}
    covered: set[str] = set()
    evidence = []
    viewer = Path(__file__).with_name('native_view.py')
    for y in range(args.grid):
        for x in range(args.grid):
            command = [
                str(args.python),
                '-u',
                str(viewer),
                '--world',
                str(args.world),
                '--index',
                str(args.index),
                '--out',
                str(args.out),
                '--grid',
                str(args.grid),
                '--region',
                str(x),
                str(y),
            ]
            with (args.out / f'region-{x}-{y}.log').open('w', encoding='utf-8') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            image = args.out / f'region-{x}-{y}.png'
            item = json.loads(image.with_suffix('.png.json').read_text(encoding='utf-8'))
            if (
                item['stageStreamingBusyAfterReadback']
                or item['geometryDecimation']
                or item['world'] != args.world.resolve().as_posix()
                or item['region'] != [x, y]
                or not image.is_file()
            ):
                raise AssertionError(f'Incomplete region capture: {x},{y}')
            covered.update(item['loadedPayloads'])
            evidence.append(
                {
                    'image': image.name,
                    'sha256': hashlib.sha256(image.read_bytes()).hexdigest(),
                    'loadedPayloads': item['loadedPayloads'],
                    'regionBounds': item['regionBounds'],
                }
            )
            print(f'CAPTURE_PROGRESS regions={len(evidence)}/{args.grid**2}', flush=True)
    if covered != expected:
        raise AssertionError(f'Payload coverage mismatch: missing={expected - covered}')
    from PIL import Image, ImageDraw

    sheet = Image.new('RGB', (320 * args.grid, 220 * args.grid), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for y in range(args.grid):
        for x in range(args.grid):
            with Image.open(args.out / f'region-{x}-{y}.png') as image:
                image.thumbnail((320, 200))
                location = (320 * x, 220 * (args.grid - y - 1))
                sheet.paste(image, location)
                draw.text((location[0] + 4, location[1] + 202), f'{x},{y}', fill='white')
    sheet.save(args.out / 'all-regions.png')
    report = {
        'status': 'passed',
        'world': args.world.resolve().as_posix(),
        'regions': len(evidence),
        'payloadsCovered': len(covered),
        'allPayloadsCovered': True,
        'allRegionsStreamingIdle': True,
        'geometryDecimation': False,
        'evidence': evidence,
    }
    (args.out / 'coverage.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'ALL_REGIONS_PASSED regions={len(evidence)} payloads={len(covered)}', flush=True)


if __name__ == '__main__':
    main()
