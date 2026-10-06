"""Extract dimensional envelopes for placement, without retaining source artwork arrays."""

import argparse
import json
from pathlib import Path

from pxr import Usd, UsdGeom


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assembly-report', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.assembly_report.read_text(encoding='utf-8'))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    models = []
    for name, entry in report['meshes'].items():
        filename = args.assembly_report.parent / 'meshes' / Path(entry['usd']).name
        stage = Usd.Stage.Open(str(filename.resolve()))
        bounds = cache.ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange()
        models.append(
            {
                'name': name,
                'min': list(bounds.GetMin()),
                'max': list(bounds.GetMax()),
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'models': models}, indent=2), encoding='utf-8')
    print(f'MODEL_ENVELOPES_COMPLETE models={len(models)}', flush=True)


if __name__ == '__main__':
    main()
