"""Fetch an audited VTC subset and convert it locally; no assets ship with usim."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
from typing import Any
import urllib.request
import xml.etree.ElementTree as ET


REPOSITORY = 'Field-Robotics-Japan/vtc_world_blender'
COMMIT = '2e6087aea800876e673e1e713a720728481e3f84'
# linuxserver/blender:4.5.3, Linux amd64 manifest, not a floating tag.
BLENDER_IMAGE = (
    'linuxserver/blender:4.5.3@'
    'sha256:16b3e174154bc89b9ae56edcfb8f5076cb3b05a6bea78156eced17073b851b7d'
)
# LFS SHA256/size from the pinned Git pointers; LICENSE is a regular Git blob.
MANIFEST = {
    'Environment/Terrain/fbx/002-ParkingArea.fbx': (
        'a27d9c0421e46680196d14655fa10d11070c5375899a5c91949b4923a5a10399',
        17356,
    ),
    'Environment/Terrain/fbx/005-ParkingArea.fbx': (
        '556a5d7b99922e340d22d3cbd4f2af99909cd12a6c6efe57601a1c613ea604d6',
        14316,
    ),
    'Environment/Terrain/fbx/006-RoadMark.fbx': (
        'e4699fcc8ab4ae8f4f7ba38b9fc9c06fbb518e0aa0b944a3135cbea8998f8823',
        238252,
    ),
    'Environment/Terrain/fbx/ParkArea.fbx': (
        'dce3d56649c6117d8aa8cf482f94d2a73057cf4896de81046a4788c90997c701',
        1492316,
    ),
    'Environment/Terrain/fbx/ParkingArea.fbx': (
        '6216cde3c8cbf357bd1b876d80bf925e204a1dbf940d48f460fc3d904c600fe9',
        61340,
    ),
    'Environment/Terrain/fbx/ParkingArea2.fbx': (
        '7474ed8bec236e0a7ebfbad3c274f54717f43819c4bf7aed02a8443d7f76a8e9',
        34172,
    ),
    'Environment/Terrain/fbx/SevenElevenArea.fbx': (
        '51a189886c9a0912830b3034c23be4bad7ad8d87aaf76c603e5a4381ab42d483',
        18668,
    ),
    'Environment/Terrain/fbx/Tsukuba-Terrain.fbx': (
        '8087186ef651cd3fc27ecbe6f997aa05ba4ad23b62310e2aed5184b66169ba9d',
        89916,
    ),
    'LICENSE': (
        'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4',
        11357,
    ),
}
NOTICE = f"""Virtual Tsukuba Challenge geometry
Source: https://github.com/{REPOSITORY}
Commit: {COMMIT}
Copyright [2020] Ryodo Tanaka groadpg@gmail.com
Licensed under the Apache License, Version 2.0; see LICENSE.

usim modifications: baked mesh transforms; triangulated geometry; translated
world origin to an interior parking surface for the default robot spawn;
untextured material colors; SDF and optional USD export; added ground and light.
Only eight allowlisted terrain/road-mark FBX exports are included. No Blender
scenes, assembly, City Hall, point clouds, photographs, or textures are fetched.
City Hall is excluded because furo-org/VTC identifies it as CC-BY-NC-SA-4.0,
despite this Blender repository's conflicting Apache attribution.
No tsukubachallenge/tc-datasets or furo-org/VTC terrain/voxel data is used.
"""


def allowed_path(path: str) -> bool:
    """An exact allowlist, also rejecting case/space variations of City Hall."""
    normalized = path.lower().replace(' ', '').replace('_', '').replace('-', '')
    return 'cityhall' not in normalized and path in MANIFEST


def verify_file(path: Path, expected_hash: str, expected_size: int) -> None:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if path.stat().st_size != expected_size or digest.hexdigest() != expected_hash:
        raise ValueError(f'Upstream size/SHA256 mismatch: {path}')


def fetch_assets(out: Path) -> None:
    for relative, (digest, size) in MANIFEST.items():
        if not allowed_path(relative):
            raise ValueError(f'Excluded asset: {relative}')
        target = out / 'upstream' / PurePosixPath(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            verify_file(target, digest, size)
            continue
        host = (
            'raw.githubusercontent.com' if relative == 'LICENSE' else 'media.githubusercontent.com'
        )
        prefix = '' if relative == 'LICENSE' else 'media/'
        url = f'https://{host}/{prefix}{REPOSITORY}/{COMMIT}/{relative}'
        partial = target.with_suffix(target.suffix + '.part')
        try:
            with (
                urllib.request.urlopen(url, timeout=120) as response,
                partial.open('wb') as stream,
            ):
                while block := response.read(1024 * 1024):
                    stream.write(block)
            verify_file(partial, digest, size)
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)
        print(f'Fetched {relative}: {size} bytes, sha256:{digest}', flush=True)
    (out / 'LICENSE').write_bytes((out / 'upstream' / 'LICENSE').read_bytes())
    (out / 'NOTICE').write_text(NOTICE, encoding='utf-8')


def write_sdf(out: Path, meshes: list[dict[str, Any]], ground_z: float = 0.0) -> Path:
    sdf = ET.Element('sdf', version='1.6')
    world = ET.SubElement(sdf, 'world', name='vtc')
    physics = ET.SubElement(world, 'physics', type='ode')
    ET.SubElement(physics, 'max_step_size').text = '0.001'
    ET.SubElement(physics, 'real_time_update_rate').text = '1000'
    scene = ET.SubElement(world, 'scene')
    ET.SubElement(scene, 'ambient').text = '0.6 0.6 0.6 1'
    light = ET.SubElement(world, 'light', name='sun', type='directional')
    ET.SubElement(light, 'direction').text = '-0.5 0.1 -1'
    ET.SubElement(light, 'diffuse').text = '0.8 0.8 0.8 1'
    for name, uri in [('ground', None), *((m['name'], m['file']) for m in meshes)]:
        model = ET.SubElement(world, 'model', name=name)
        ET.SubElement(model, 'static').text = 'true'
        if uri is None:
            ET.SubElement(model, 'pose').text = f'0 0 {ground_z - 0.05:g} 0 0 0'
        link = ET.SubElement(model, 'link', name='geometry')
        for kind in ('visual', 'collision'):
            node = ET.SubElement(link, kind, name=kind)
            geometry = ET.SubElement(node, 'geometry')
            if uri is None:
                box = ET.SubElement(geometry, 'box')
                ET.SubElement(box, 'size').text = '2000 2000 0.1'
            else:
                mesh = ET.SubElement(geometry, 'mesh')
                ET.SubElement(mesh, 'uri').text = uri
                ET.SubElement(mesh, 'scale').text = '1 1 1'
    ET.indent(sdf)
    path = out / 'world.sdf'
    ET.ElementTree(sdf).write(path, encoding='utf-8', xml_declaration=True)
    return path


def write_usd(out: Path, meshes: list[dict[str, Any]], ground_z: float = 0.0) -> Path:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    path = out / 'world.usdc'
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, '/World')
    stage.SetDefaultPrim(root.GetPrim())
    physics = UsdPhysics.Scene.Define(stage, '/World/Physics')
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    physics.CreateGravityMagnitudeAttr(9.81)
    ground = UsdGeom.Cube.Define(stage, '/World/Ground')
    ground.CreateSizeAttr(1.0)
    ground.AddTranslateOp().Set(Gf.Vec3d(0, 0, ground_z - 0.05))
    ground.AddScaleOp().Set(Gf.Vec3d(2000, 2000, 0.1))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    ground.CreateDisplayColorAttr([(0.35, 0.35, 0.35)])
    for item in meshes:
        mesh = UsdGeom.Mesh.Define(stage, '/World/' + item['name'])
        mesh.CreatePointsAttr(item['vertices'])
        mesh.CreateFaceVertexCountsAttr([3] * len(item['triangles']))
        mesh.CreateFaceVertexIndicesAttr([index for tri in item['triangles'] for index in tri])
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateDisplayColorAttr([item['color'][:3]])
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('none')
        material = UsdShade.Material.Define(stage, '/World/Materials/' + item['name'])
        shader = UsdShade.Shader.Define(stage, material.GetPath().AppendChild('Shader'))
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*item['color'][:3])
        )
        shader.CreateOutput('surface', Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('assets/vtc'))
    parser.add_argument('--usd', action='store_true', help='Also write USD; requires usim[usd]')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    fetch_assets(out)
    (out / 'manifest.json').write_text(json.dumps(MANIFEST, indent=2), encoding='utf-8')
    script = Path(__file__).with_name('convert_blender.py').resolve()
    subprocess.run(
        [
            'docker',
            'run',
            '--rm',
            '--network',
            'none',
            '--entrypoint',
            '/usr/bin/blender',
            '--mount',
            f'type=bind,source={out},target=/vtc',
            '--mount',
            f'type=bind,source={script},target=/convert.py,readonly',
            BLENDER_IMAGE,
            '--background',
            '--factory-startup',
            '--disable-autoexec',
            '--python-exit-code',
            '1',
            '--python',
            '/convert.py',
        ],
        check=True,
    )
    meshes = json.loads((out / 'geometry.json').read_text(encoding='utf-8'))
    origin = json.loads((out / 'origin.json').read_text(encoding='utf-8'))
    outputs = [write_sdf(out, meshes, origin['ground_z_m'])]
    if args.usd:
        outputs.append(write_usd(out, meshes, origin['ground_z_m']))
    evidence = {
        'repository': REPOSITORY,
        'commit': COMMIT,
        'blender_image': BLENDER_IMAGE,
        **origin,
        'sources': {p: {'sha256': h, 'bytes': n} for p, (h, n) in MANIFEST.items()},
        'meshes': len(meshes),
        'triangles': sum(len(m['triangles']) for m in meshes),
    }
    (out / 'provenance.json').write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    for path in outputs:
        print(f'Generated {path}', flush=True)


if __name__ == '__main__':
    main()
