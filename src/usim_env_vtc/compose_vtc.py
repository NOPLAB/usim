"""Compose a LOCAL mixed-license VTC inspection stage without copying its artwork."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vtc-world', type=Path, required=True)
    parser.add_argument('--vtc-index', type=Path, required=True)
    parser.add_argument('--survey-world', type=Path, required=True)
    parser.add_argument('--survey-index', type=Path, required=True)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    registration = json.loads(args.registration.read_text(encoding='utf-8'))
    matrix = np.asarray(registration['source19_to_vtc'], dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('Require the validated 2019-to-VTC homogeneous transform')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(args.out.resolve()))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, '/World').GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    stage.GetRootLayer().subLayerPaths = [
        Path(os.path.relpath(args.vtc_world.resolve(), args.out.parent.resolve())).as_posix()
    ]
    survey = UsdGeom.Xform.Define(stage, '/World/PublicSurvey')
    survey.GetPrim().GetReferences().AddReference(
        Path(os.path.relpath(args.survey_world.resolve(), args.out.parent.resolve())).as_posix(),
        '/World',
    )
    survey.MakeMatrixXform().Set(Gf.Matrix4d(*matrix.T.ravel().tolist()))
    stage.GetRootLayer().customLayerData = {
        'distribution': 'LOCAL inspection only; mixed VTC artwork licenses are not cleared',
        'licenseAudit': 'docs/vtc/tsukuba-full-map/LICENSES.md',
        'publicSurveyLicense': 'CC-BY-NC-SA-4.0',
        'originalAssetsModified': False,
        'surveyVerticesModified': False,
    }
    stage.GetRootLayer().Save()
    original = json.loads(args.vtc_index.read_text(encoding='utf-8'))
    public = json.loads(args.survey_index.read_text(encoding='utf-8'))
    entries = list(original)
    for entry in public:
        corners = np.array(list(itertools.product(*zip(entry['min'], entry['max']))))
        transformed = corners @ matrix[:3, :3].T + matrix[:3, 3]
        entries.append(
            {
                **entry,
                'path': entry['path'].replace('/World/', '/World/PublicSurvey/', 1),
                'min': transformed.min(axis=0).tolist(),
                'max': transformed.max(axis=0).tolist(),
            }
        )
    index_path = args.out.with_suffix('.index.json')
    index_path.write_text(json.dumps(entries, indent=2), encoding='utf-8')
    # Reopen the composed root without loading native geometry and validate paths.
    reopened = Usd.Stage.Open(str(args.out.resolve()), load=Usd.Stage.LoadNone)
    for entry in entries:
        prim = reopened.GetPrimAtPath(entry['path'])
        if not prim or not prim.HasAuthoredPayloads():
            raise AssertionError(f'Composed payload missing: {entry["path"]}')
    actual = np.asarray(
        UsdGeom.Xformable(reopened.GetPrimAtPath('/World/PublicSurvey')).GetLocalTransformation()
    )
    if not np.array_equal(actual, matrix.T):
        raise AssertionError('Composed coordinate transform changed')
    print(f'COMPOSITION_PASSED original={len(original)} public={len(public)} index={index_path}')


if __name__ == '__main__':
    main()
