"""Run inside pinned Blender; import allowlisted FBX and export portable meshes."""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import bpy
from mathutils import Vector


OUT = Path('/vtc')
NS = 'http://www.collada.org/2005/11/COLLADASchema'
ET.register_namespace('', NS)


def element(parent, tag, text=None, **attributes):
    node = ET.SubElement(parent, '{' + NS + '}' + tag, attributes)
    node.text = text
    return node


def write_collada(path, vertices, triangles, color):
    root = ET.Element('{' + NS + '}COLLADA', version='1.4.1')
    asset = element(root, 'asset')
    element(asset, 'created', '2020-01-01T00:00:00Z')
    element(asset, 'modified', '2020-01-01T00:00:00Z')
    element(asset, 'unit', name='meter', meter='1')
    element(asset, 'up_axis', 'Z_UP')
    effects = element(root, 'library_effects')
    effect = element(effects, 'effect', id='surface')
    profile = element(effect, 'profile_COMMON')
    technique = element(profile, 'technique', sid='common')
    diffuse = element(element(technique, 'lambert'), 'diffuse')
    element(diffuse, 'color', ' '.join(map(str, color)))
    material = element(element(root, 'library_materials'), 'material', id='material')
    element(material, 'instance_effect', url='#surface')
    geometry = element(element(root, 'library_geometries'), 'geometry', id='geometry')
    mesh = element(geometry, 'mesh')
    source = element(mesh, 'source', id='positions')
    element(
        source,
        'float_array',
        ' '.join(str(value) for point in vertices for value in point),
        id='position_array',
        count=str(3 * len(vertices)),
    )
    accessor = element(
        element(source, 'technique_common'),
        'accessor',
        source='#position_array',
        count=str(len(vertices)),
        stride='3',
    )
    for axis in 'XYZ':
        element(accessor, 'param', name=axis, type='float')
    element(
        element(mesh, 'vertices', id='vertices'), 'input', semantic='POSITION', source='#positions'
    )
    faces = element(mesh, 'triangles', count=str(len(triangles)), material='material')
    element(faces, 'input', semantic='VERTEX', source='#vertices', offset='0')
    element(faces, 'p', ' '.join(str(i) for tri in triangles for i in tri))
    scene = element(element(root, 'library_visual_scenes'), 'visual_scene', id='scene')
    instance = element(element(scene, 'node', id='object'), 'instance_geometry', url='#geometry')
    common = element(element(instance, 'bind_material'), 'technique_common')
    element(common, 'instance_material', symbol='material', target='#material')
    element(element(root, 'scene'), 'instance_visual_scene', url='#scene')
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)


def main():
    results = []
    (OUT / 'meshes').mkdir(exist_ok=True)
    manifest = json.loads((OUT / 'manifest.json').read_text())
    for relative in manifest:
        if not relative.endswith('.fbx'):
            continue
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.fbx(
            filepath=str(OUT / 'upstream' / relative),
            use_anim=False,
            use_image_search=False,
        )
        objects = list(bpy.context.scene.objects)
        for obj in objects:
            normalized = obj.name.lower().replace(' ', '').replace('_', '').replace('-', '')
            if 'cityhall' in normalized:
                raise ValueError(f'Excluded object in {relative}: {obj.name}')
        bpy.context.view_layer.update()
        dependency_graph = bpy.context.evaluated_depsgraph_get()
        for obj in objects:
            if obj.type != 'MESH':
                continue
            evaluated = obj.evaluated_get(dependency_graph)
            mesh = evaluated.to_mesh()
            try:
                mesh.calc_loop_triangles()
                if not mesh.loop_triangles:
                    continue
                name = f'mesh_{len(results):04d}'
                # Blender's FBX importer converts source units and axes to metres/Z-up.
                vertices = [list(evaluated.matrix_world @ v.co) for v in mesh.vertices]
                triangles = [list(tri.vertices) for tri in mesh.loop_triangles]
                color = (
                    list(mesh.materials[0].diffuse_color) if mesh.materials else [0.5] * 3 + [1]
                )
                filename = f'meshes/{name}.dae'
                results.append(
                    {
                        'name': name,
                        'source': relative,
                        'source_object': obj.name,
                        'file': filename,
                        'vertices': vertices,
                        'triangles': triangles,
                        'color': color,
                    }
                )
            finally:
                evaluated.to_mesh_clear()
    if not results:
        raise ValueError('No terrain geometry was exported')
    # Anchor the default robot spawn on an interior parking-surface triangle.
    # All FBX exports share the authored coordinates, so translate them together.
    candidates = []
    first_source = next(path for path in manifest if path.endswith('.fbx'))
    for item in results:
        if item['source'] != first_source:
            continue
        for triangle in item['triangles']:
            a, b, c = (Vector(item['vertices'][index]) for index in triangle)
            normal = (b - a).cross(c - a)
            if normal.z > 0 and normal.z >= 0.99 * normal.length:
                candidates.append((normal.z, (a + b + c) / 3))
    if not candidates:
        raise ValueError('The pinned parking export has no horizontal spawn surface')
    origin = max(candidates, key=lambda candidate: candidate[0])[1]
    for item in results:
        item['vertices'] = [list(Vector(point) - origin) for point in item['vertices']]
        write_collada(OUT / item['file'], item['vertices'], item['triangles'], item['color'])
    ground_z = min(point[2] for item in results for point in item['vertices']) - 0.1
    (OUT / 'origin.json').write_text(
        json.dumps({'source_origin_m': list(origin), 'ground_z_m': ground_z}, indent=2)
    )
    (OUT / 'geometry.json').write_text(json.dumps(results), encoding='utf-8')
    print(
        f'Exported {len(results)} meshes, '
        f'{sum(len(item["triangles"]) for item in results)} triangles',
        flush=True,
    )


if __name__ == '__main__':
    main()
