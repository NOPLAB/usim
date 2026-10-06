"""Render one exact map region per Isaac process without changing source geometry."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Final, TypedDict

HIDDEN_SURVEY_PATHS: Final = (
    '/World/Survey2019',
    '/World/PointCompletion',
    '/World/PublicSurvey',
    '/World/PointRegions',
)


class Payload(TypedDict):
    path: str
    min: list[float]
    max: list[float]


def visible_payloads(entries: list[Payload]) -> list[Payload]:
    """Keep full-detail mesh payloads without loading point-cloud layers."""
    return [
        entry
        for entry in entries
        if not any(
            entry['path'] == root or entry['path'].startswith(root + '/')
            for root in HIDDEN_SURVEY_PATHS
        )
    ]


def main() -> None:
    """Initialize SDK imports only after creating the native application."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--world', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--region', type=int, nargs=2, default=(4, 4))
    parser.add_argument('--grid', type=int, default=8)
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--oblique', action='store_true')
    parser.add_argument('--pose')
    parser.add_argument('--pan-to', type=int, nargs=2, help='Headless camera residency probe')
    parser.add_argument('--mesh-only', action='store_true', help='Exclude survey point layers')
    parser.add_argument('--all-regions', action='store_true', help='Preload visible mesh payloads')
    args = parser.parse_args()
    if args.grid < 1 or any(value < 0 or value >= args.grid for value in args.region):
        parser.error('region must be inside the requested grid')
    if args.pan_to and any(value < 0 or value >= args.grid for value in args.pan_to):
        parser.error('pan destination must be inside the requested grid')
    sys.argv = [sys.argv[0]]

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            'headless': not args.interactive,
            'renderer': 'RayTracedLighting',
            'width': 1600,
            'height': 1000,
        }
    )

    import anyio
    import carb
    import carb.eventdispatcher
    import omni.usd
    from omni.kit.async_engine import run_coroutine
    from omni.kit.viewport.utility import (
        capture_viewport_to_file,
        get_active_viewport,
        next_viewport_frame_async,
    )
    from omni.kit.viewport.actions.actions import toggle_grid_visibility
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

    entries: list[Payload] = json.loads(args.index.read_text())
    low = [min(entry['min'][axis] for entry in entries) for axis in (0, 1)]
    high = [max(entry['max'][axis] for entry in entries) for axis in (0, 1)]
    size = [(high[axis] - low[axis]) / args.grid for axis in (0, 1)]
    mesh_only = args.mesh_only or args.interactive or args.pan_to is not None
    preload = args.all_regions
    display_world = args.world.resolve()
    if mesh_only:
        entries = visible_payloads(entries)
        args.out.mkdir(parents=True, exist_ok=True)
        display_world = (args.out / 'mesh-inspection.usda').resolve()
        inspection = Usd.Stage.CreateNew(str(display_world))
        inspection.GetRootLayer().subLayerPaths = [
            Path(os.path.relpath(args.world.resolve(), display_world.parent)).as_posix()
        ]
        inspection.SetDefaultPrim(inspection.GetPrimAtPath('/World'))
        UsdGeom.SetStageMetersPerUnit(inspection, 1.0)
        UsdGeom.SetStageUpAxis(inspection, UsdGeom.Tokens.z)
        for path in HIDDEN_SURVEY_PATHS:
            inspection.OverridePrim(path).SetActive(False)
        for entry in entries:
            inspection.OverridePrim(entry['path']).SetActive(False)
        inspection.GetRootLayer().Save()
    center = [low[axis] + (args.region[axis] + 0.5) * size[axis] for axis in (0, 1)]
    radius = max(size) * 0.75
    half_width = max(size[0], size[1] * 1.6) / 2
    pan_limit = (
        [radius * 0.6, radius * 0.6]
        if args.oblique
        else [radius - half_width, radius - half_width / 1.6]
    )
    settings = carb.settings.get_settings()
    settings.set('/rtx/materialDb/syncLoads', True)
    settings.set('/rtx/hydra/materialSyncLoads', True)
    settings.set('/rtx/hydra/points/renderMode', 'spheres')
    grid_was_visible = bool(
        (settings.get('/persistent/app/viewport/displayOptions') or 0) & (1 << 6)
    )
    resident: set[str] = set()

    def load_region(point: tuple[float, float]) -> set[str]:
        """Cull off-view meshes and keep the GPU residency bounded without changing files."""
        stage = omni.usd.get_context().get_stage()
        frustum = (
            UsdGeom.Camera(stage.GetPrimAtPath(get_active_viewport().camera_path))
            .GetCamera()
            .frustum
        )
        desired = {
            entry['path']
            for entry in entries
            if preload
            or (
                all(
                    entry['min'][axis] <= point[axis] + radius
                    and entry['max'][axis] >= point[axis] - radius
                    for axis in (0, 1)
                )
                and (
                    not mesh_only
                    or frustum.Intersects(
                        Gf.BBox3d(Gf.Range3d(Gf.Vec3d(*entry['min']), Gf.Vec3d(*entry['max'])))
                    )
                )
            )
        }
        culled = resident - desired if mesh_only and not preload else set()
        for path in culled:
            stage.GetPrimAtPath(path).SetActive(False)
        resident.difference_update(culled)
        new = desired - resident
        if new:
            if mesh_only:
                for path in new:
                    stage.GetPrimAtPath(path).SetActive(True)
            stage.LoadAndUnload({Sdf.Path(path) for path in new}, set())
            resident.update(new)
        center[:] = point
        if new or culled:
            print(
                f'REGION_LOADED pid={os.getpid()} added={len(new)} '
                f'culled={len(culled)} resident={len(resident)} center={point}',
                flush=True,
            )
        return desired

    def view_center(matrix: Gf.Matrix4d) -> tuple[float, float]:
        """Project the rendered viewing direction to the source Z=0 reference plane."""
        eye = matrix.ExtractTranslation()
        forward = matrix.TransformDir(Gf.Vec3d(0, 0, -1))
        distance = -eye[2] / forward[2] if forward[2] < -1e-6 else radius
        point = eye + forward * distance
        return float(point[0]), float(point[1])

    async def render() -> None:
        """Wait on native frame/streaming events before reading the actual GPU frame."""
        context = omni.usd.get_context()
        idle = anyio.Event()
        armed = False

        def frame_complete(event: carb.eventdispatcher.Event) -> None:
            if (
                armed
                and not context.get_stage_streaming_status()
                and context.get_stage_loading_status()[2] == 0
            ):
                idle.set()

        subscription = carb.eventdispatcher.get_eventdispatcher().observe_event(
            event_name=context.stage_rendering_event_name(
                omni.usd.StageRenderingEventType.NEW_FRAME, immediate=True
            ),
            on_event=frame_complete,
            observer_name='usim full-map streaming completion',
        )
        with anyio.fail_after(300):
            opened, error = await context.open_stage_async(
                str(display_world), load_set=omni.usd.UsdContextInitialLoadSet.LOAD_NONE
            )
            if not opened:
                raise RuntimeError(error)
            stage = context.get_stage()
            stage.SetEditTarget(stage.GetSessionLayer())
            camera = UsdGeom.Camera.Define(stage, '/World/Cameras/UsimMapInspector')
            camera.CreateClippingRangeAttr((0.05, 4000))
            if not args.oblique:
                camera.CreateProjectionAttr('orthographic')
                aperture = max(size[0], size[1] * 1.6) * 10
                camera.CreateHorizontalApertureAttr(aperture)
                camera.CreateVerticalApertureAttr(aperture / 1.6)
            if args.pose:
                rows = json.loads(args.pose)
                pose = Gf.Matrix4d(*[value for row in rows for value in row])
                center[:] = view_center(pose)
            elif args.oblique:
                pose = (
                    Gf.Matrix4d()
                    .SetLookAt(
                        (center[0] + radius, center[1] - radius, radius),
                        (center[0], center[1], 0),
                        (0, 0, 1),
                    )
                    .GetInverse()
                )
            else:
                pose = (
                    Gf.Matrix4d()
                    .SetLookAt((center[0], center[1], 1000), (center[0], center[1], 0), (0, 1, 0))
                    .GetInverse()
                )
            camera.MakeMatrixXform().Set(pose)
            sun = UsdLux.DistantLight.Define(stage, '/World/ViewerSun')
            sun.CreateIntensityAttr(2000)
            UsdGeom.Xformable(sun).AddRotateXYZOp().Set((35, -25, 0))
            sky = UsdLux.DomeLight.Define(stage, '/World/ViewerSky')
            sky.CreateIntensityAttr(300)
            sky.CreateColorAttr((0.65, 0.8, 1))
            viewport = get_active_viewport()
            toggle_grid_visibility(viewport, visible=False)
            viewport.camera_path = camera.GetPath()
            viewport.resolution = (1600, 1000)
            desired = load_region((center[0], center[1]))
            armed = True
            args.out.mkdir(parents=True, exist_ok=True)
            suffix = '-oblique' if args.oblique else ''
            filename = f'region-{args.region[0]}-{args.region[1]}{suffix}.png'
            while True:
                idle = anyio.Event()
                await idle.wait()
                await next_viewport_frame_async(viewport)
                expected = viewport.view
                result = capture_viewport_to_file(viewport, str(args.out / filename))
                await result.wait_for_result()
                if (
                    not context.get_stage_streaming_status()
                    and context.get_stage_loading_status()[2] == 0
                    and Gf.IsClose(viewport.view, expected, 1e-6)
                ):
                    break
            (args.out / (filename + '.json')).write_text(
                json.dumps(
                    {
                        'world': args.world.resolve().as_posix(),
                        'region': args.region,
                        'grid': args.grid,
                        'center': center,
                        'radius': radius,
                        'regionBounds': [
                            [center[axis] - size[axis] / 2 for axis in (0, 1)],
                            [center[axis] + size[axis] / 2 for axis in (0, 1)],
                        ],
                        'loadedPayloads': sorted(desired),
                        'stageStreamingBusyAfterReadback': False,
                        'renderedView': [list(row) for row in viewport.view],
                        'geometryDecimation': False,
                        'surveyPointsDisplayed': not mesh_only,
                        'allVisiblePayloadsPreloaded': preload,
                        'meshCullingEnabled': mesh_only and not preload,
                        'meshResidencyRadiusMeters': radius,
                    },
                    indent=2,
                ),
                encoding='utf-8',
            )
            del subscription
            print(
                f'REGION_CAPTURED {args.region[0]},{args.region[1]} '
                f'payloads={len(desired)} file={filename}',
                flush=True,
            )

    def interact() -> None:
        """Keep the native window while loading newly visited map regions."""
        pending = run_coroutine(render())
        while not pending.done():
            app.update()
        pending.result()
        if args.pan_to:

            async def pan_probe() -> None:
                viewport = get_active_viewport()
                target = [low[axis] + (args.pan_to[axis] + 0.5) * size[axis] for axis in (0, 1)]
                pose = viewport.view.GetInverse()
                translation = pose.ExtractTranslation()
                translation[0] += target[0] - center[0]
                translation[1] += target[1] - center[1]
                pose.SetTranslateOnly(translation)
                settled = anyio.Event()

                def changed(event: carb.eventdispatcher.Event) -> None:
                    context = omni.usd.get_context()
                    if (
                        Gf.IsClose(viewport.view.GetInverse(), pose, 1e-6)
                        and not context.get_stage_streaming_status()
                        and context.get_stage_loading_status()[2] == 0
                    ):
                        settled.set()

                observer = carb.eventdispatcher.get_eventdispatcher().observe_event(
                    event_name=omni.usd.get_context().stage_rendering_event_name(
                        omni.usd.StageRenderingEventType.NEW_FRAME, immediate=True
                    ),
                    on_event=changed,
                    observer_name='usim camera residency probe',
                )
                camera = UsdGeom.Xformable(
                    omni.usd.get_context().get_stage().GetPrimAtPath(viewport.camera_path)
                )
                camera.GetOrderedXformOps()[0].Set(pose)
                with anyio.fail_after(300):
                    await settled.wait()
                actual = viewport.view.GetInverse()
                point = view_center(actual)
                if not any(abs(point[axis] - center[axis]) > pan_limit[axis] for axis in (0, 1)):
                    raise AssertionError('Probe did not leave resident geometry coverage')
                settled = anyio.Event()
                desired = load_region(point)
                with anyio.fail_after(300):
                    await settled.wait()
                await next_viewport_frame_async(viewport)
                result = capture_viewport_to_file(viewport, str(args.out / 'pan-probe.png'))
                await result.wait_for_result()
                if not Gf.IsClose(viewport.view.GetInverse(), actual, 1e-6):
                    raise AssertionError('Residency update changed the user camera')
                (args.out / 'pan-probe.json').write_text(
                    json.dumps(
                        {
                            'pid': os.getpid(),
                            'center': point,
                            'renderedPose': [list(row) for row in actual],
                            'loadedPayloads': sorted(desired),
                            'residentPayloads': sorted(resident),
                            'nativeProcessRestarted': False,
                            'surveyPointsDisplayed': not mesh_only,
                        },
                        indent=2,
                    ),
                    encoding='utf-8',
                )
                del observer
                print(f'REGION_PROBE_PASSED pid={os.getpid()} payloads={len(desired)}', flush=True)

            pending = run_coroutine(pan_probe())
            while not pending.done():
                app.update()
            pending.result()
        if args.interactive:
            viewport = get_active_viewport()
            previous_view = Gf.Matrix4d(viewport.view)
            camera = UsdGeom.Camera(
                omni.usd.get_context().get_stage().GetPrimAtPath(viewport.camera_path)
            )
            previous_projection = camera.GetCamera().frustum.ComputeProjectionMatrix()
            while app.is_running():
                app.update()
                matrix = viewport.view.GetInverse()
                point = view_center(matrix)
                camera = UsdGeom.Camera(
                    omni.usd.get_context().get_stage().GetPrimAtPath(viewport.camera_path)
                )
                projection = camera.GetCamera().frustum.ComputeProjectionMatrix()
                if not Gf.IsClose(viewport.view, previous_view, 1e-5) or not Gf.IsClose(
                    projection, previous_projection, 1e-5
                ):
                    load_region(point)
                    previous_view = Gf.Matrix4d(viewport.view)
                    previous_projection = projection

    try:
        interact()
    finally:
        toggle_grid_visibility(get_active_viewport(), visible=grid_was_visible)
        app.close()


if __name__ == '__main__':
    main()
