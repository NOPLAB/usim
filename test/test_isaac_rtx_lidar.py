"""Test native RTX boundaries without importing Isaac Sim or requiring CUDA."""

import builtins
import importlib
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from usim.simulation import SimulationConfig
from usim_isaacsim.rtx_lidar import RtxLidar


@pytest.fixture
def configuration():
    return SimulationConfig(
        world=Path('world.usda'),
        robot_urdf=Path('robot.urdf'),
        lidar_samples=720,
        lidar_hz=10,
        lidar_min_range=0.1,
        lidar_max_range=70,
        lidar_vertical_angles=tuple(np.radians(np.linspace(-7, 52, 28))),
        lidar_offset=(1, 2, 3),
    )


@pytest.fixture
def native(monkeypatch):
    events = []
    stage = Mock(spec=['GetPrimAtPath', 'RemovePrim'])
    stage.GetPrimAtPath.return_value.IsValid.return_value = False
    attributes = {}

    def attribute(name):
        return attributes.setdefault(name, Mock(spec=['Set']))

    prim = Mock(spec=['GetAttribute'])
    prim.GetAttribute.side_effect = attribute
    lidar = SimpleNamespace(
        prims=[prim],
        set_world_poses=Mock(side_effect=lambda **kwargs: events.append('pose')),
    )
    sensor = Mock(spec=['get_data', 'detach_annotators', 'annotators'])
    sensor.annotators = ['generic-model-output']
    sensor.get_data.side_effect = lambda annotator: (None, {})
    world = Mock(spec=['render'])
    world.render.side_effect = lambda: events.append('render')
    api = SimpleNamespace(
        Lidar=Mock(return_value=lidar),
        LidarSensor=Mock(return_value=sensor),
        parse_generic_model_output_data=Mock(),
    )
    settings = Mock(spec=['set_bool'])
    modules = {
        'carb': {'settings': SimpleNamespace(get_settings=lambda: settings)},
        'omni': {},
        'omni.usd': {'get_context': lambda: SimpleNamespace(get_stage=lambda: stage)},
        'isaacsim': {},
        'isaacsim.core': {},
        'isaacsim.core.utils': {},
        'isaacsim.core.utils.extensions': {'enable_extension': Mock()},
        'isaacsim.sensors': {},
        'isaacsim.sensors.experimental': {},
        'isaacsim.sensors.experimental.rtx': vars(api),
        'pxr': {'Vt': SimpleNamespace(FloatArray=tuple, UIntArray=tuple)},
    }
    for name, members in modules.items():
        module = ModuleType(name)
        vars(module).update(members)
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, child = name.rpartition('.')
        if parent:
            monkeypatch.setattr(sys.modules[parent], child, module, raising=False)
    return SimpleNamespace(
        api=api,
        world=world,
        lidar=lidar,
        sensor=sensor,
        stage=stage,
        attributes=attributes,
        events=events,
        settings=settings,
    )


def payload(native, count=20160, stamp=1):
    gmo = SimpleNamespace(
        numElements=count,
        timestampNs=stamp,
        x=np.ones(count),
        y=np.zeros(count),
        z=np.zeros(count),
        scalar=np.arange(count, dtype=np.float32),
    )
    # Only the official parser may interpret/copy this opaque CUDA buffer.
    data = SimpleNamespace(size=128, device='cuda:0')
    native.sensor.get_data.side_effect = lambda annotator: (
        native.events.append('data') or data,
        {},
    )
    native.api.parse_generic_model_output_data.return_value = gmo
    return data, gmo


def test_import_is_native_lazy(monkeypatch):
    original = builtins.__import__

    def no_native(name, *args, **kwargs):
        assert name.split('.')[0] not in {'carb', 'omni', 'isaacsim', 'pxr'}
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', no_native)
    importlib.reload(sys.modules['usim_isaacsim.rtx_lidar'])


def test_verified_rotary_configuration(configuration, native):
    RtxLidar(configuration, native.world)
    kwargs = native.api.Lidar.call_args.kwargs
    attrs = kwargs['attributes']
    prefix = 'omni:sensor:Core:'
    assert kwargs['tick_rate'] == 10
    assert kwargs['accumulate_outputs'] is False
    assert kwargs['aux_output_level'] == 'BASIC'
    assert attrs == {
        prefix + 'scanType': 'ROTARY',
        prefix + 'numberOfEmitters': 28,
        prefix + 'numberOfChannels': 28,
        prefix + 'patternFiringRateHz': 7200,
        prefix + 'scanRateBaseHz': 10,
        prefix + 'instantLidar': True,
        prefix + 'rotationDirection': 'CCW',
        prefix + 'startAzimuthOffsetDeg': -180,
        prefix + 'nearRangeM': 0.1,
        prefix + 'farRangeM': 70,
        prefix + 'maxReturns': 1,
        prefix + 'rangeAccuracyM': 0,
        prefix + 'azimuthErrorStd': 0,
        prefix + 'elevationErrorStd': 0,
        prefix + 'elementsCoordsType': 'CARTESIAN',
        prefix + 'skipDroppingInvalidPoints': True,
    }
    emitter = prefix + 'emitterState:s001:'
    for key, expected in (
        ('azimuthDeg', [0] * 28),
        ('elevationDeg', np.linspace(-7, 52, 28)),
        ('fireTimeNs', [0] * 28),
        ('channelId', range(1, 29)),
    ):
        np.testing.assert_allclose(
            native.attributes[emitter + key].Set.call_args.args[0], expected
        )
    # 720 rotary ticks are half-open: final azimuth is 179.5, not a duplicate 180.
    assert attrs[prefix + 'startAzimuthOffsetDeg'] + 719 * 360 / 720 == 179.5
    native.settings.set_bool.assert_called_once_with(
        '/app/sensors/nv/lidar/outputBufferOnGPU', True
    )
    native.api.LidarSensor.assert_called_once_with(
        native.lidar, annotators=['generic-model-output']
    )


@pytest.mark.parametrize(
    ('changes', 'message'),
    [
        ({'lidar_angle_max': 0}, '360-degree'),
        ({'lidar_hz': 10.5}, 'integral SDK'),
    ],
)
def test_unsupported_configuration_fails_before_native_import(configuration, changes, message):
    with pytest.raises(ValueError, match=message):
        RtxLidar(replace(configuration, **changes), None)


def test_exact_body_rotation_offset_and_local_output(configuration, native):
    adapter = RtxLidar(configuration, native.world)
    data, gmo = payload(native)
    # 120 degrees about (1,1,1) maps (x,y,z) -> (z,x,y); includes roll and pitch.
    result = adapter.capture((10, 20, 30), (0.5, 0.5, 0.5, 0.5))
    kwargs = native.lidar.set_world_poses.call_args.kwargs
    np.testing.assert_array_equal(kwargs['positions'], [[13, 21, 32]])
    np.testing.assert_array_equal(kwargs['orientations'], [[0.5, 0.5, 0.5, 0.5]])
    assert native.events == ['pose', 'render', 'data']
    native.world.render.assert_called_once_with()
    native.sensor.get_data.assert_called_once_with('generic-model-output')
    native.api.parse_generic_model_output_data.assert_called_once_with(data)
    points, intensity = result
    assert points.dtype == intensity.dtype == np.float32
    assert points.shape == (20160, 3)
    np.testing.assert_array_equal(points, np.column_stack((gmo.x, gmo.y, gmo.z)))
    np.testing.assert_array_equal(intensity, gmo.scalar)
    gmo.x[:] = 9
    gmo.scalar[:] = 0
    assert points[0, 0] == 1
    assert intensity[1] == 1


def test_finite_intensity_and_range_filtering(configuration, native):
    adapter = RtxLidar(
        replace(configuration, lidar_samples=8, lidar_vertical_angles=None), native.world
    )
    _, gmo = payload(native, count=8)
    gmo.x[:] = [0, 0.05, 0.1, 70, 71, np.nan, np.inf, 1]
    gmo.scalar[:] = [1, 2, 3, 4, 5, 6, 7, np.nan]
    points, intensity = adapter.capture((0, 0, 0), (1, 0, 0, 0))
    np.testing.assert_allclose(points, [[0.1, 0, 0], [70, 0, 0]])
    np.testing.assert_array_equal(intensity, [3, 4])
    assert (
        native.api.Lidar.call_args.kwargs['attributes']['omni:sensor:Core:numberOfEmitters'] == 1
    )


@pytest.mark.parametrize('empty', [None, SimpleNamespace(size=0)])
def test_absent_payload_is_not_parsed_and_is_bounded(configuration, native, empty):
    adapter = RtxLidar(configuration, native.world)
    native.sensor.get_data.side_effect = lambda annotator: (empty, {})
    for _ in range(29):
        assert adapter.capture((0, 0, 0), (1, 0, 0, 0)) is None
    with pytest.raises(RuntimeError, match='30 consecutive captures'):
        adapter.capture((0, 0, 0), (1, 0, 0, 0))
    assert native.world.render.call_count == 30
    native.api.parse_generic_model_output_data.assert_not_called()


def test_stale_scan_empty_scan_and_recovery(configuration, native):
    adapter = RtxLidar(configuration, native.world)
    _, gmo = payload(native)
    assert adapter.capture((0, 0, 0), (1, 0, 0, 0)) is not None
    for _ in range(29):
        assert adapter.capture((0, 0, 0), (1, 0, 0, 0)) is None
    gmo.timestampNs = 2
    assert adapter.capture((0, 0, 0), (1, 0, 0, 0)) is not None
    gmo.numElements = 0
    assert adapter.capture((0, 0, 0), (1, 0, 0, 0)) is None
    gmo.numElements = 20160
    gmo.timestampNs = 1
    assert adapter.capture((0, 0, 0), (1, 0, 0, 0)) is None


def test_scan_without_meaningful_hits_is_bounded(configuration, native):
    adapter = RtxLidar(configuration, native.world)
    _, gmo = payload(native)
    gmo.x[:] = 0
    for stamp in range(29):
        gmo.timestampNs = stamp
        assert adapter.capture((0, 0, 0), (1, 0, 0, 0)) is None
    gmo.timestampNs = 30
    with pytest.raises(RuntimeError, match='30 consecutive captures'):
        adapter.capture((0, 0, 0), (1, 0, 0, 0))


def test_wrong_emitted_count_is_not_hidden_by_filtering(configuration, native):
    adapter = RtxLidar(configuration, native.world)
    payload(native, count=20159)
    with pytest.raises(RuntimeError, match='20159 rays; expected 20160'):
        adapter.capture((0, 0, 0), (1, 0, 0, 0))


def test_close_releases_only_owned_resources_idempotently(configuration, native):
    adapter = RtxLidar(configuration, native.world)
    native.stage.GetPrimAtPath.return_value.IsValid.return_value = True
    adapter.close()
    adapter.close()
    native.sensor.detach_annotators.assert_called_once_with(['generic-model-output'])
    native.stage.RemovePrim.assert_called_once_with('/World/UsimRtxLidar')
    assert adapter._sensor is adapter._lidar is None
    with pytest.raises(RuntimeError, match='closed'):
        adapter.capture((0, 0, 0), (1, 0, 0, 0))
    native.world.render.assert_not_called()


def test_existing_prim_is_never_wrapped_or_removed(configuration, native):
    native.stage.GetPrimAtPath.return_value.IsValid.return_value = True
    with pytest.raises(ValueError, match='unoccupied prim path'):
        RtxLidar(configuration, native.world)
    native.api.Lidar.assert_not_called()
    native.stage.RemovePrim.assert_not_called()


def test_constructor_failure_removes_owned_prim(configuration, native):
    native.api.LidarSensor.side_effect = RuntimeError('annotator failure')
    native.stage.GetPrimAtPath.return_value.IsValid.side_effect = [False, True]
    with pytest.raises(RuntimeError, match='annotator failure'):
        RtxLidar(configuration, native.world)
    native.stage.RemovePrim.assert_called_once_with('/World/UsimRtxLidar')
