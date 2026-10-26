"""Replace planar raster presentations with authoritative OpenCASCADE B-Rep surfaces."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .cad_contours import world_loops
from .cad_exchange import write_brep
from .cad_kernel import compound, make_planar_face, make_planar_faces, surface_area
from .cad_merge import consolidate
from .cad_output import publish
from .cad_types import CadArchive, CadBoard, CadConfig, CadReport
from .models import PipelineInputError
from .pcd import open_records, sha256, xyz
from .storage import write_json
from .usd import LayerMetadata


def reconstruct(output: Path, config: CadConfig = CadConfig()) -> CadReport:
    """Retain measured archives; heal CAD inner wires and consolidate duplicate surfaces."""
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    source = manifest['sources'][manifest['primary_source']]
    if sha256(output / source['path']) != source['sha256']:
        raise PipelineInputError('CAD source archive hash mismatch')
    records, _ = open_records(output / source['path'])
    planar = json.loads((output / 'planar-report.json').read_text(encoding='utf-8'))
    plane_config = json.loads((output / 'optimized/config.json').read_text(encoding='utf-8'))

    def build(number: int) -> tuple[CadBoard | None, int, float, float]:
        with np.load(output / 'optimized/planes' / f'plane_{number:04}.npz') as archive:
            data: CadArchive = {
                'equation': archive['equation'],
                'u': archive['u'],
                'v': archive['v'],
                'lower': archive['lower'],
                'mask': archive['mask'],
            }
            ids = np.unique(archive['ids'])
        loops = world_loops(data, plane_config['cell'], config.boundary_tolerance)
        normal = data['equation'][:3]
        removed_count, removed_area = 0, 0.0
        for loop in loops:
            signed = float(np.cross(loop, np.roll(loop, -1, axis=0)).sum(0) @ normal)
            if signed < 0:
                area = surface_area(make_planar_face(loop, normal=normal))
                if area < config.hole_area:
                    removed_count += 1
                    removed_area += area
        built = make_planar_faces(loops, normal, hole_threshold_m2=config.hole_area)
        retained, omitted = [], 0.0
        for face in built:
            area = surface_area(face)
            if area >= config.minimum_face_area:
                retained.append(face)
            else:
                omitted += area
        print(
            f'CAD_INPUT_CONVERTED plane={number:04} faces={len(retained)} '
            f'holes_removed={removed_count}',
            flush=True,
        )
        board = (
            CadBoard(compound(retained), data['equation'], (number,), ids) if retained else None
        )
        if board is not None:
            write_brep(board.shape, output / 'cad/inputs' / f'plane_{number:04}.brep')
        return board, removed_count, removed_area, omitted

    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        converted = list(pool.map(build, range(planar['planes'])))
    boards = [item[0] for item in converted if item[0] is not None]
    boards, decisions = consolidate(boards, config, xyz(records))
    folder = output / 'cad'
    folder.mkdir(exist_ok=True)
    write_json(folder / 'config.json', asdict(config))
    write_json(folder / 'joins.json', decisions)
    write_json(
        folder / 'metadata.json',
        {
            'license': source['license'],
            'attribution': source['attribution'],
            'source_catalog': manifest['source_catalog'],
            'frame': manifest['frame'],
            'brep_units': 'meters',
            'step_units': 'millimeters',
        },
    )
    metadata = LayerMetadata(
        source['license'],
        source['attribution'],
        manifest['source_catalog'],
        manifest['frame'],
        'authoritative OpenCASCADE B-Rep; CAD internal-wire repair; kernel-only tessellation',
    )
    rows = publish(output, boards, metadata, config)
    report: CadReport = {
        'kernel': 'OpenCASCADE 7.9.3',
        'authority': 'cad/model.brep',
        'step_units': 'millimeters',
        'brep_units': 'meters',
        'source_sha256': source['sha256'],
        'source_records': len(records),
        'input_planes': planar['planes'],
        'output_boards': len(boards),
        'output_faces': sum(row['faces'] for row in rows),
        'output_holes': sum(row['holes'] for row in rows),
        'small_holes_removed': sum(item[1] for item in converted),
        'small_hole_area': sum(item[2] for item in converted),
        'contained_boards_removed': sum(item['kind'] == 'contained' for item in decisions),
        'overlapping_boards_fused': sum(item['kind'] == 'fused' for item in decisions),
        'omitted_small_face_area': sum(item[3] for item in converted),
        'surface_area': sum(row['area'] for row in rows),
        'display_triangles': sum(row['triangles'] for row in rows),
        'world': 'cad.usda',
    }
    write_json(output / 'cad-report.json', report)
    print('CAD_RECONSTRUCTION_PASSED ' + json.dumps(report), flush=True)
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        raise PipelineInputError('CAD workers must be positive')
    reconstruct(args.map, CadConfig(workers=args.workers))


if __name__ == '__main__':
    main()
