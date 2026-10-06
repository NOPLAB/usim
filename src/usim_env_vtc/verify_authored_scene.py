"""Verify independent artwork dependencies and exact retained map-data geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils


def main() -> None:
    """Check the actual reopened world, including dependencies of unloaded payloads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    folder = args.world.resolve().parent
    report = json.loads((folder / 'authorship.json').read_text(encoding='utf-8'))
    entries = json.loads((folder / 'world.index.json').read_text(encoding='utf-8'))
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(
        Sdf.AssetPath(str(args.world.resolve()))
    )
    if (
        unresolved
        or assets
        or any(not Path(layer.realPath).resolve().is_relative_to(folder) for layer in layers)
    ):
        raise AssertionError(f'External asset dependencies remain: {assets}, {unresolved}')
    for layer in layers:
        stage = Usd.Stage.Open(layer, Usd.Stage.LoadNone)
        for prim in stage.Traverse():
            for attr in prim.GetAttributes():
                if attr.GetTypeName() in (Sdf.ValueTypeNames.Asset, Sdf.ValueTypeNames.AssetArray):
                    raise AssertionError(f'Unexpected texture/MDL asset: {attr.GetPath()}')
    source = Usd.Stage.Open(str(args.source.resolve()), Usd.Stage.LoadNone)
    for entry in entries:
        if entry['path'].startswith('/World/Vegetation/'):
            source.Load(str(Sdf.Path(entry['path']).GetParentPath()))
    actual = Usd.Stage.Open(str(args.world.resolve()), Usd.Stage.LoadNone)
    source_cache = UsdGeom.XformCache()
    actual_cache = UsdGeom.XformCache()
    for placement in report['placements']:
        path = placement['path']
        original = source.GetPrimAtPath(path)
        replaced = actual.GetPrimAtPath(path)
        if not replaced or not replaced.HasAuthoredPayloads() or not replaced.IsInstanceable():
            raise AssertionError(f'Missing replacement payload: {path}')
        if source_cache.GetLocalToWorldTransform(
            original
        ) != actual_cache.GetLocalToWorldTransform(replaced):
            raise AssertionError(f'Placement changed: {path}')
    vertices = triangles = components = 0
    bounds_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default'])
    posts = 0
    facade_buildings = 0
    facade_models: set[str] = set()
    for entry in entries:
        path = entry['path']
        actual.Load(path)
        replacement = actual.GetPrimAtPath(path)
        shape = actual.GetPrimAtPath(path + '/Shape')
        if shape and shape.GetCustomDataByKey('genericFacadeNotSurveyed'):
            facade_buildings += 1
            facade_models.add(entry['asset'])
        bounds = bounds_cache.ComputeWorldBound(replacement).ComputeAlignedRange()
        if not all(
            entry['min'][i] <= bounds.GetMin()[i] + 1e-5
            and entry['max'][i] >= bounds.GetMax()[i] - 1e-5
            for i in range(3)
        ):
            raise AssertionError(f'Payload bounds do not enclose geometry: {path}')
        if path.startswith('/World/InferredStructures/'):
            posts += sum(prim.IsA(UsdGeom.Cylinder) for prim in replacement.GetChildren())
            if any(not prim.HasAPI(UsdPhysics.CollisionAPI) for prim in replacement.GetChildren()):
                raise AssertionError(f'Measured post collision missing: {path}')
        if path.startswith('/World/Terrain/'):
            source.Load(path)
            for prim in replacement.GetChildren():
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                mesh = UsdGeom.Mesh(prim)
                old = UsdGeom.Mesh(source.GetPrimAtPath(prim.GetPath()))
                for getter in (
                    'GetPointsAttr',
                    'GetFaceVertexCountsAttr',
                    'GetFaceVertexIndicesAttr',
                ):
                    if not np.array_equal(
                        np.asarray(getattr(mesh, getter)().Get()),
                        np.asarray(getattr(old, getter)().Get()),
                    ):
                        raise AssertionError(
                            f'Repaired geometry changed: {prim.GetPath()}, {getter}'
                        )
                if source_cache.GetLocalToWorldTransform(old.GetPrim()) != (
                    actual_cache.GetLocalToWorldTransform(prim)
                ):
                    raise AssertionError(f'Terrain transform changed: {prim.GetPath()}')
                if not prim.HasAPI(UsdPhysics.CollisionAPI) or (
                    UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() != 'none'
                ):
                    raise AssertionError(f'Exact terrain collider missing: {prim.GetPath()}')
                vertices += len(mesh.GetPointsAttr().Get())
                triangles += len(mesh.GetFaceVertexIndicesAttr().Get()) // 3
                components += 1
            source.Unload(path)
            print(f'AUTHORED_VERIFIED_TILE {path}', flush=True)
        actual.Unload(path)
        bounds_cache.Clear()
    for relative, expected in report['hashes'].items():
        if hashlib.sha256((folder / relative).read_bytes()).hexdigest() != expected:
            raise AssertionError(f'Generated output hash mismatch: {relative}')
    result = {
        'status': 'passed',
        'world': str(args.world.resolve()),
        'placementTransformsExact': len(report['placements']),
        'payloadBoundsConservative': len(entries),
        'allDependenciesInsideOutput': True,
        'externalTextureOrMdlAssets': len(assets),
        'unresolvedAssets': unresolved,
        'terrainVerticesExact': vertices,
        'terrainTrianglesExact': triangles,
        'terrainComponentsExact': components,
        'measuredPosts': posts,
        'genericFacadeBuildings': facade_buildings,
        'genericFacadePrototypes': len(facade_models),
        'sourceArtworkReferences': 0,
        'outputHashesVerified': len(report['hashes']),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('AUTHORED_VERIFICATION_PASSED ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
