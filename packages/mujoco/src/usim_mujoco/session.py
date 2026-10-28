"""Owned native CPU or CUDA physics, generalized state, and scalar joint control."""

from dataclasses import replace
from importlib import import_module
from math import ceil
from typing import Any

from usim.capability import Phase
from usim.facets import JointLayout, StateSchema
from usim.groups import AgentGroup, ExecutionGroup
from usim.model import ModelError
from usim.scene import SceneBindings, SceneEditor, SetJointState
from usim.specs import ControlSpec
from usim.types import StepResult
from .access import Control, State
from .support import unsupported


class Session:
    def __init__(self, backend, spec, execution, name, plan):
        mujoco = import_module('mujoco')
        np = import_module('numpy')

        self.backend, self.spec, self.options = backend, spec, execution
        self.models = backend.models
        self.name = name
        self._phase = Phase.BUILDING
        self.owned = None
        self.model: Any = None
        self.data: Any = None
        self.warp_model: Any = None
        self.warp_data: Any = None
        self.wp: Any = None
        self.mjw: Any = None
        self.device: Any = None
        self.targets = {}
        self.steps = 0
        self.np, self.mj = np, mujoco
        try:
            # Resolve CUDA before compiling or allocating native model resources.
            if backend.gpu:
                mujoco_warp = import_module('mujoco_warp')
                warp = import_module('warp')

                devices = warp.get_cuda_devices()
                if not devices:
                    raise RuntimeError(
                        'mujoco-warp requires a CUDA device; CPU fallback is disabled'
                    )
                self.wp, self.mjw, self.device = warp, mujoco_warp, devices[0]
            self.owned = self.models.load(plan)
            self.model = self.owned.value
            interval = 1.0 / execution.sim_rate
            self.substeps = max(1, ceil(interval / self.model.opt.timestep))
            self.model.opt.timestep = interval / self.substeps
            self.data = mujoco.MjData(self.model)
            self.scalar = {}
            layouts = []
            for index in range(self.model.njnt):
                native_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, index)
                joint_name = native_name or f'joint_{index}'
                kind = int(self.model.jnt_type[index])
                q = int(self.model.jnt_qposadr[index])
                v = int(self.model.jnt_dofadr[index])
                nq, nv = {0: (7, 6), 1: (4, 3), 2: (1, 1), 3: (1, 1)}[kind]
                label = {0: 'free', 1: 'ball', 2: 'slide', 3: 'hinge'}[kind]
                layouts.append(
                    JointLayout(
                        joint_name, label, (q, q + nq), (v, v + nv), 'm' if kind == 2 else 'rad'
                    )
                )
                if kind in {2, 3} and native_name:
                    self.scalar[joint_name] = (q, v)
            self.schema = StateSchema(
                name, tuple(layouts), n_envs=1, batched=False, quaternion_order='wxyz'
            )
            self.control_spec = spec.agents[0].control if spec.agents else ControlSpec()
            c = self.control_spec
            self.control_names = c.joint_names or tuple(self.scalar)
            selected = c.position_joints + c.velocity_joints + c.effort_joints
            if not set(self.control_names + selected) <= self.scalar.keys():
                raise ModelError('Named controls require named hinge or slide joints')
            if c.initial_qpos is not None and len(c.initial_qpos) != len(self.control_names):
                raise ModelError('Initial positions must match the selected scalar joint order')
            self.gains = {}
            for i, joint in enumerate(self.control_names):
                self.gains[joint] = (
                    c.joint_stiffness[i] if c.joint_stiffness is not None else 100.0,
                    c.joint_damping[i] if c.joint_damping is not None else 10.0,
                    c.joint_force_limits[i] if c.joint_force_limits is not None else float('inf'),
                )
            self.state, self.control = State(self), Control(self)
            self.agents = AgentGroup(spec.agents)
            self.execution = ExecutionGroup(episode=self)
            self.scene = SceneEditor(
                (name,),
                SceneBindings(self._assess_edit, self._edit, lambda: self.phase, self._fail),
            )
            if backend.gpu:
                with self.wp.ScopedDevice(self.device):
                    self.warp_model = self.mjw.put_model(self.model)
            self._phase = Phase.READY
            self.reset()
        except BaseException:
            self._fail()
            raise

    @property
    def phase(self):
        return self._phase

    def _require(self):
        if self.phase in {Phase.CLOSED, Phase.FAILED} or self.data is None:
            raise ModelError(f'MuJoCo session is {self.phase.value}')

    def _entity(self, name):
        self._require()
        if name != self.name:
            raise ModelError(f'Unknown scene entity: {name}')

    def assess(self, request):
        report = self.backend.assess(
            replace(request, phase=self.phase, n_envs=self.options.n_envs)
        )
        if report.supported and request.operation.startswith('control.') and not self.scalar:
            return unsupported('This model has no named scalar joints')
        return report

    def _assess_edit(self, change, request):
        report = self.assess(request)
        if not report.supported:
            return report
        if not isinstance(change, SetJointState):
            return unsupported('Only scalar joint state edits are implemented')
        self._entity(change.entity)
        if any(name not in self.scalar for name, _ in change.positions + change.velocities):
            return unsupported('Joint state edits require named hinge or slide joints')
        return report

    def _upload(self):
        if self.backend.gpu:
            with self.wp.ScopedDevice(self.device):
                self.warp_data = self.mjw.put_data(self.model, self.data)

    def _edit(self, change):
        for name, value in change.positions:
            self.data.qpos[self.scalar[name][0]] = value
        for name, value in change.velocities:
            self.data.qvel[self.scalar[name][1]] = value
        self.targets.clear()
        self.data.qfrc_applied[:] = 0
        self.mj.mj_forward(self.model, self.data)
        self._upload()

    def reset(self, seed=None):
        self._require()
        # Native reset is deterministic; no task randomization or reward is invented.
        self.mj.mj_resetData(self.model, self.data)
        initial = self.control_spec.initial_qpos
        if initial is not None:
            for name, value in zip(self.control_names, initial):
                self.data.qpos[self.scalar[name][0]] = value
        self.targets.clear()
        self.steps = 0
        self.mj.mj_forward(self.model, self.data)
        self._upload()
        self._phase = Phase.READY
        return self.state.get_observation(), {}

    def step(self):
        self._require()
        self._phase = Phase.RUNNING
        try:
            for _ in range(self.substeps):
                self.data.qfrc_applied[:] = 0
                if self.targets and self.model.nu:
                    self.mj.mj_forward(self.model, self.data)
                for name, (mode, target) in self.targets.items():
                    q, v = self.scalar[name]
                    kp, kd, limit = self.gains[name]
                    force = target
                    if mode == 'position':
                        force = kp * (target - self.data.qpos[q]) - kd * self.data.qvel[v]
                    elif mode == 'velocity':
                        force = kd * (target - self.data.qvel[v])
                    self.data.qfrc_applied[v] = (
                        self.np.clip(force, -limit, limit) - self.data.qfrc_actuator[v]
                    )
                if self.backend.gpu:
                    with self.wp.ScopedDevice(self.device):
                        for field in ('ctrl', 'qfrc_applied'):
                            destination = getattr(self.warp_data, field)
                            values = getattr(self.data, field).reshape(1, -1)
                            self.wp.copy(
                                destination,
                                self.wp.array(values, dtype=destination.dtype, device=self.device),
                            )
                        self.mjw.step(self.warp_model, self.warp_data)
                        self.mjw.get_data_into(self.data, self.model, self.warp_data)
                else:
                    self.mj.mj_step(self.model, self.data)
            self.steps += 1
            self._phase = Phase.PAUSED
            return StepResult(
                self.state.get_observation(),
                0.0,
                False,
                self.steps >= self.options.max_episode_steps,
                {},
            )
        except BaseException:
            self._fail()
            raise

    def close(self):
        self.targets.clear()
        self.warp_data = self.warp_model = self.data = self.model = None
        if self.owned is not None:
            self.owned.value = None
            self.owned.close()
            self.owned = None
        self._phase = Phase.CLOSED

    def _fail(self):
        self.close()
        self._phase = Phase.FAILED

    def __enter__(self):
        self._require()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
