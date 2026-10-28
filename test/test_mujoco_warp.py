"""Warp boundary regressions without allocating a GPU or starting native CUDA."""

from contextlib import nullcontext
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from test.test_mujoco_backend import native as native, scene as scene
from usim.model import ModelError
from usim.specs import ExecutionSpec
from usim_mujoco import WarpBackend


def test_warp_without_cuda_never_compiles_or_falls_back(native, scene, monkeypatch):
    calls, _, sdk = native
    sdk.MjModel.from_xml_path = lambda path: pytest.fail(
        'CPU model was allocated before CUDA check'
    )
    monkeypatch.setitem(sys.modules, 'warp', SimpleNamespace(get_cuda_devices=lambda: []))
    monkeypatch.setitem(sys.modules, 'mujoco_warp', SimpleNamespace())
    with pytest.raises(RuntimeError, match='CPU fallback is disabled'):
        WarpBackend().open(scene, ExecutionSpec(backend='gpu'))
    assert calls.cpu_steps == 0


def test_warp_rejects_explicit_cpu_selection_before_native_import(scene):
    # Given a caller who explicitly selects CPU computation.
    # When opening the CUDA-only provider.
    with pytest.raises(ModelError, match='compute backend'):
        WarpBackend().open(scene, ExecutionSpec(backend='cpu'))
    # Then no CPU physics or GPU allocation was substituted.


def test_warp_uses_official_transfers_and_cuda_step_only(native, scene, monkeypatch):
    calls, model, _ = native

    def put_data(model, host):
        calls.uploads += 1
        return SimpleNamespace(
            ctrl=np.zeros((1, 1), dtype=np.float32),
            qfrc_applied=np.zeros((1, 1), dtype=np.float32),
            time=host.time,
            qpos=host.qpos.copy(),
            qvel=host.qvel.copy(),
        )

    def gpu_step(model, data):
        calls.gpu_steps += 1
        data.time += model.opt.timestep

    def get_data_into(host, model, device):
        calls.downloads += 1
        host.time = device.time

    monkeypatch.setitem(
        sys.modules,
        'warp',
        SimpleNamespace(
            get_cuda_devices=lambda: ['cuda:0'],
            ScopedDevice=lambda device: nullcontext(),
            array=lambda values, **kwargs: np.array(values, dtype=kwargs['dtype']),
            copy=lambda dst, src: np.copyto(dst, src),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        'mujoco_warp',
        SimpleNamespace(
            put_model=lambda actual: model,
            put_data=put_data,
            step=gpu_step,
            get_data_into=get_data_into,
        ),
    )
    with WarpBackend().open(scene, ExecutionSpec(backend='gpu')) as session:
        session.control.set_efforts({'hinge': 0.3})
        result = session.step()
        assert result.observation.extra['time'] == pytest.approx(1 / 30)
        assert calls.gpu_steps == calls.downloads == session.substeps
        assert calls.cpu_steps == 0
        np.testing.assert_allclose(session.warp_data.qfrc_applied, [[0.3]])
        session.reset()
        assert calls.uploads == 2
        assert session.data.time == 0
