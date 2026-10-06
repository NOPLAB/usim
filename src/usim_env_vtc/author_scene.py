"""Build a standalone VTC world with original artwork and retained map-data placement."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Literal, TypedDict

from pxr import Gf, Sdf, Usd, UsdGeom

if not __package__:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'usim_env_vtc'
from .authored_architecture import author_architecture
from .authored_buildings import author_building, building_dimensions
from .authored_map_geometry import retain_bsp, retain_posts
from .authored_notice import write_notice
from .authored_terrain import TerrainEntry, author_tile
from .authored_trees import author_tree


class ModelInfo(TypedDict):
    name: str
    min: list[float]
    max: list[float]


class Placement(TypedDict):
    path: str
    sourceModel: str
    authoredModel: str
    transform: list[list[float]]


def model_reference(prim: Usd.Prim) -> str | None:
    """Locate the direct prototype arc without inheriting ancestor or material references."""
    for spec in prim.GetPrimStack():
        for reference in spec.referenceList.GetAppliedItems():
            if reference.primPath == Sdf.Path('/Model'):
                return Path(reference.assetPath).name
    return None


def author_prototype(
    filename: Path,
    info: ModelInfo,
    seed: int,
    physical_size: tuple[float, float, float] | None = None,
) -> None:
    """Create independent geometry using only dimensional envelopes, never source mesh data."""
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, '/Model')
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    name = info['name']
    low, high = Gf.Vec3d(*info['min']), Gf.Vec3d(*info['max'])
    tree = any(term in name for term in ('Tree', 'Pine', 'Bush'))
    if tree:
        kind: Literal['broadleaf', 'pine', 'bush', 'stump'] = 'broadleaf'
        if 'Pine' in name:
            kind = 'pine'
        if 'Bush' in name:
            kind = 'bush'
        if 'Stump' in name:
            kind = 'stump'
        author_tree(stage, '/Model/Shape', kind, seed)
        shape = UsdGeom.Xformable(stage.GetPrimAtPath('/Model/Shape'))
        shape.AddTranslateOp().Set(((low[0] + high[0]) / 2, (low[1] + high[1]) / 2, low[2]))
        shape.AddScaleOp().Set(((high[0] - low[0]) / 2, (high[1] - low[1]) / 2, high[2] - low[2]))
    elif physical_size is not None:
        author_building(stage, '/Model/Shape', physical_size)
    else:
        author_architecture(stage, '/Model/Shape', name, tuple(low), tuple(high))
    stage.GetRootLayer().customLayerData = {
        'artworkAuthorship': 'original procedural usim model',
        'sourceMeshCopied': False,
        'sourceTexturesCopied': False,
        'dimensionalEnvelopeSource': name,
        'seed': seed,
    }
    stage.GetRootLayer().Export(str(filename.resolve()))


def ancestor_transforms(source: Usd.Stage, world: Usd.Stage, path: str) -> None:
    """Retain source parent coordinate frames while discarding all ancestor asset arcs."""
    for ancestor in Sdf.Path(path).GetPrefixes()[:-1]:
        if world.GetPrimAtPath(ancestor):
            continue
        new = UsdGeom.Xform.Define(world, ancestor)
        old = source.GetPrimAtPath(ancestor)
        if old and old.IsA(UsdGeom.Xformable):
            new.MakeMatrixXform().Set(UsdGeom.Xformable(old).GetLocalTransformation())


def place_model(source: Usd.Stage, world: Usd.Stage, path: str, asset: str) -> TerrainEntry:
    """Preserve an actor matrix and visibility, with a payload pointing only to original art."""
    ancestor_transforms(source, world, path)
    old = source.GetPrimAtPath(path)
    new = UsdGeom.Xform.Define(world, path)
    new.MakeMatrixXform().Set(UsdGeom.Xformable(old).GetLocalTransformation())
    new.CreateVisibilityAttr(UsdGeom.Imageable(old).GetVisibilityAttr().Get())
    new.GetPrim().GetPayloads().AddPayload(asset, '/Model')
    new.GetPrim().SetInstanceable(True)
    world.Load(path)
    bounds = (
        UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        .ComputeWorldBound(new.GetPrim())
        .ComputeAlignedRange()
    )
    entry: TerrainEntry = {
        'path': path,
        'asset': asset,
        'prim': '/Model',
        'min': list(bounds.GetMin()),
        'max': list(bounds.GetMax()),
    }
    world.Unload(path)
    return entry


def main() -> None:
    """Generate a reproducible independent world; map-data restrictions remain explicit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--masks', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--software-license', type=Path, default=Path('LICENSE'))
    parser.add_argument(
        '--vtc-license', type=Path, default=Path('assets/vtc-full/upstream/VTC/LICENSE')
    )
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text(encoding='utf-8'))
    source = Usd.Stage.Open(str(args.world.resolve()), Usd.Stage.LoadNone)
    source_index = json.loads(args.index.read_text(encoding='utf-8'))
    for entry in source_index:
        if entry['path'].startswith(('/World/Vegetation/', '/World/InferredStructures/')):
            source.Load(entry['path'])
    models: dict[str, ModelInfo] = {
        info['name'].replace('/', '_') + '.usdc': info for info in inventory['models']
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for directory in ('models', 'terrain'):
        (args.out / directory).mkdir(exist_ok=True)
    world = Usd.Stage.CreateNew(str((args.out / 'world.usda').resolve()), Usd.Stage.LoadNone)
    world.SetDefaultPrim(UsdGeom.Xform.Define(world, '/World').GetPrim())
    UsdGeom.SetStageMetersPerUnit(world, 1)
    UsdGeom.SetStageUpAxis(world, UsdGeom.Tokens.z)
    ledger: list[Placement] = []
    entries: list[TerrainEntry] = []
    authored: dict[tuple[str, tuple[float, float, float] | None], str] = {}
    omitted = []
    for prim in source.Traverse():
        if not prim.IsA(UsdGeom.Xform):
            continue
        reference = model_reference(prim)
        if reference is None:
            continue
        path = str(prim.GetPath())
        if path.startswith('/World/Structures/Reference/') or any(
            token in reference for token in ('SM_SkySphere', 'ReferenceMark')
        ):
            omitted.append(path)
            continue
        info = models[reference]
        physical_size = (
            building_dimensions(prim) if info['name'] == 'Engine/BasicShapes/Cube' else None
        )
        key = reference, physical_size
        if key not in authored:
            asset = f'models/model_{len(authored):03}.usdc'
            author_prototype(args.out / asset, info, len(authored), physical_size)
            authored[key] = asset
        asset = authored[key]
        entries.append(place_model(source, world, path, asset))
        ledger.append(
            {
                'path': path,
                'sourceModel': info['name'],
                'authoredModel': asset,
                'transform': [
                    list(row) for row in UsdGeom.Xformable(prim).GetLocalTransformation()
                ],
            }
        )
    entries.extend(retain_posts(source, world, args.out))
    bsp = retain_bsp(source, world, args.out)
    if bsp is not None:
        entries.append(bsp)
    for number, entry in enumerate(source_index):
        if entry['path'].startswith('/World/Terrain/'):
            new_entry = author_tile(source, entry, args.masks, args.out / 'terrain', number)
            ancestor_transforms(source, world, entry['path'])
            tile = world.DefinePrim(entry['path'], 'Xform')
            tile.GetPayloads().AddPayload(new_entry['asset'], '/Tile')
            entries.append(new_entry)
    world.GetRootLayer().customLayerData = {
        'artwork': 'original procedural meshes/materials; no external artwork references',
        'mapData': 'VTC terrain, masks and placements; LiDAR-derived vegetation placements',
        'licenseAudit': 'docs/vtc/authored-models.md',
        'originalAssetsModified': False,
        'geographicSurveyAccuracyEstablished': False,
    }
    world.GetRootLayer().Save()
    (args.out / 'world.index.json').write_text(json.dumps(entries, indent=2), encoding='utf-8')
    hashes = {
        file.relative_to(args.out).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest()
        for file in sorted(args.out.rglob('*.usdc'))
    }
    report = {
        'placements': ledger,
        'placementCount': len(ledger),
        'prototypeCount': len(authored),
        'omittedNonDisplayHelpers': omitted,
        'importedArtworkReferenced': False,
        'originalAssetsModified': False,
        'terrainGeometryRetained': True,
        'roadRepairRetained': True,
        'artworkRecipeLicense': 'MIT',
        'wholeMapLicense': 'NOT blanket MIT; see audit',
        'sourceWorld': str(args.world.resolve()),
        'hashes': hashes,
    }
    (args.out / 'authorship.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    write_notice(args.out, args.software_license, args.vtc_license)
    print(f'AUTHORED_SCENE_COMPLETE placements={len(ledger)} models={len(authored)}', flush=True)


if __name__ == '__main__':
    main()
