"""The map launcher must not recreate its native window on residency changes."""

import importlib.util
import json
from pathlib import Path
import sys


SCRIPT = Path(__file__).parents[1] / 'src' / 'usim_env_vtc' / 'view.py'
SPEC = importlib.util.spec_from_file_location('vtc_full_view', SCRIPT)
assert SPEC is not None and SPEC.loader is not None
viewer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(viewer)
NATIVE_SPEC = importlib.util.spec_from_file_location(
    'vtc_full_native_view', SCRIPT.with_name('native_view.py')
)
assert NATIVE_SPEC is not None and NATIVE_SPEC.loader is not None
native_view = importlib.util.module_from_spec(NATIVE_SPEC)
NATIVE_SPEC.loader.exec_module(native_view)


def test_inspection_payloads_exclude_survey_fragments_but_retain_original_mesh_detail():
    # Given original meshes, measured fragments and duplicate point layers.
    paths = [
        '/World/Terrain/TC_x0_y0',
        '/World/Structures/OriginalBuilding',
        '/World/PublicSurvey/Regions/t_9985_10007',
        '/World/Survey2019/Unsplit/Part_14',
        '/World/PublicSurvey/PointRegions/t_9985_10007',
        '/World/PointRegions/t_9985_10007',
        '/World/PointCompletion',
    ]
    entries = [{'path': path, 'min': [0, 0, 0], 'max': [1, 1, 1]} for path in paths]
    # When selecting GUI payloads, without altering the original map index.
    displayed = native_view.visible_payloads(entries)
    # Then both complete mesh payloads remain, and no survey layer is loaded.
    assert [entry['path'] for entry in displayed] == paths[:2]
    assert [entry['path'] for entry in entries] == paths


def test_region_request_does_not_restart_native_process(tmp_path, monkeypatch):
    # Given a real native child that reports the observed residency transition.
    counter = tmp_path / 'starts.json'
    native = tmp_path / 'native_view.py'
    native.write_text(
        'import json\nfrom pathlib import Path\n'
        f'counter = Path({str(counter)!r})\n'
        'count = json.loads(counter.read_text()) if counter.exists() else 0\n'
        'counter.write_text(json.dumps(count + 1))\n'
        'if count == 0:\n'
        '    print(\'REGION_REQUEST {"center": [20, 20], "pose": '
        "[[1,0,0,0],[0,1,0,0],[0,0,1,0],[20,20,1000,1]]}', flush=True)\n",
        encoding='utf-8',
    )
    index = tmp_path / 'index.json'
    index.write_text(
        json.dumps([{'path': '/World/Tile', 'min': [0, 0, 0], 'max': [100, 100, 10]}]),
        encoding='utf-8',
    )
    monkeypatch.setattr(viewer, '__file__', str(tmp_path / 'view.py'))
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'view.py',
            '--world',
            str(tmp_path / 'world.usda'),
            '--index',
            str(index),
            '--out',
            str(tmp_path / 'out'),
        ],
    )
    # When a region transition is reported to the GUI launcher.
    viewer.main()
    # Then the same native process owns the window until it exits.
    assert json.loads(counter.read_text()) == 1
