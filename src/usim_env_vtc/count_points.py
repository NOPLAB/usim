"""Count actual USD point primitives and records without starting a GPU renderer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--inspection', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    stage = Usd.Stage.Open(str(args.world.resolve()), load=Usd.Stage.LoadNone)
    entries = json.loads(args.index.read_text(encoding='utf-8'))
    clouds = []
    seen: set[str] = set()

    def count(root: Usd.Prim) -> None:
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdGeom.Points) and str(prim.GetPath()) not in seen:
                path = str(prim.GetPath())
                seen.add(path)
                points = UsdGeom.Points(prim).GetPointsAttr().Get()
                clouds.append({'path': path, 'points': len(points)})

    # References without payloads are present already; payload arrays load one at a time.
    count(stage.GetPseudoRoot())
    for entry in entries:
        stage.Load(entry['path'])
        count(stage.GetPrimAtPath(entry['path']))
        stage.Unload(entry['path'])
    groups = {
        'vtc_2019_unsplit': '/World/Survey2019/Unsplit/',
        'vtc_2019_split': '/World/Survey2019/',
        'public_2019_regions': '/World/PublicSurvey/PointRegions/',
    }
    totals = {}
    assigned: set[str] = set()
    for label, prefix in groups.items():
        selected = [
            row for row in clouds if row['path'].startswith(prefix) and row['path'] not in assigned
        ]
        assigned.update(row['path'] for row in selected)
        totals[label] = {
            'pointPrimitives': len(selected),
            'points': sum(row['points'] for row in selected),
        }
    other = [row for row in clouds if row['path'] not in assigned]
    totals['other'] = {
        'pointPrimitives': len(other),
        'points': sum(row['points'] for row in other),
    }
    inspection = Usd.Stage.Open(str(args.inspection.resolve()), load=Usd.Stage.LoadAll)
    active_points = [prim for prim in inspection.Traverse() if prim.IsA(UsdGeom.Points)]
    mesh_count, mesh_vertices, mesh_triangles = 0, 0, 0
    for prim in inspection.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            mesh_count += 1
            mesh_vertices += len(mesh.GetPointsAttr().Get())
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
            mesh_triangles += int(np.maximum(counts - 2, 0).sum(dtype=np.int64))
    result = {
        'sourceWorld': args.world.resolve().as_posix(),
        'groups': totals,
        'totalPointPrimitives': len(clouds),
        'totalStoredPoints': sum(row['points'] for row in clouds),
        'activeInspectionPointPrimitives': len(active_points),
        'activeInspectionPoints': sum(
            len(UsdGeom.Points(prim).GetPointsAttr().Get()) for prim in active_points
        ),
        'activeMeshPrimitives': mesh_count,
        'activeMeshVertices': mesh_vertices,
        'activeMeshTriangleEquivalent': mesh_triangles,
        'meshCountScope': 'active primitive traversal; shared instance proxies not expanded',
        'sourceArraysModified': False,
        'uniquePhysicalPointsClaimed': False,
        'duplicateNotice': '2019 split, unsplit and public copies overlap; storage sum is not a unique spatial count',
        'clouds': clouds,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('POINT_COUNT_PASSED ' + json.dumps({k: v for k, v in result.items() if k != 'clouds'}))


if __name__ == '__main__':
    main()
