# Genesis

`usim_genesis` is an optional, individually installable simulator backend.
It uses Genesis World 1.4.3. Install it alongside the core `usim` package in
an environment separate from Isaac Sim or ManiSkill. Genesis also requires
PyTorch 2.9.1 through the `native` extra. Choose its CPU/CUDA build for your
machine; Windows GPU execution needs the CUDA wheel rather than the default CPU wheel.

From the workspace root, for example:

```console
uv venv --python 3.12 runs/genesis-venv
uv pip install --python runs/genesis-venv/Scripts/python.exe -e . -e "packages/genesis[native]"
uv pip install --python runs/genesis-venv/Scripts/python.exe torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

On Linux, use `runs/genesis-venv/bin/python` instead of the Windows executable.
The core package does not import Genesis until this backend is selected.

## External robots

Use `JointControl` with an explicit MJCF `.xml` or URDF `.urdf` file and ordered
single-DOF `joint_names`. A joint may describe a wheel, arm, or gripper; control is
joint-space and does not infer task-specific kinematics. MJCF retains its model
base joint; URDF bases are fixed by default. Set `fixed_base=False` to retain
a mobile URDF's floating base. Free-base DOFs stay in native physics state even
when only wheel and arm joints are mapped for control.
Mapped ball/free joints are rejected because they do not have one DOF per name.
Unmapped joints remain in the scene but do not appear in actions or observations.

```python
import numpy as np

from usim import SimulatorConfig
from usim_genesis import GenesisSimulator

sim = GenesisSimulator(
    SimulatorConfig(
        env_id='JointControl',
        robot_path='/absolute/path/to/robot.urdf',
        robot_uid='my_robot',
        joint_names=('shoulder', 'elbow', 'finger'),
        gripper_joint_names=('finger',),
        initial_qpos=(0.0, 0.2, 0.0),
        backend='cpu',
        render_mode='none',
        obs_mode='state',
        n_envs=2,
        robot_init_qpos_noise=0.0,
    )
)
try:
    observation, info = sim.reset(seed=7)
    result = sim.step(np.array([[0.1, 0.2, 0.0], [-0.1, 0.2, 0.0]]))
    observation, info = sim.reset(seed=7, env_ids=np.array([0]))
finally:
    sim.close()
```

Action order is `joint_names` when provided, otherwise configured arm names
followed by gripper names, not the model's native DOF order. External robots never
inherit CRANE-X7 joint names, mimic
behavior, initial pose, or force limits. Their default initial positions are zero;
provide `initial_qpos` when zero is unsuitable. Neutral PD gains are 100/10;
model-defined effort bounds remain in effect, except an explicit
`gripper_force_limit` overrides the mapped gripper bounds. CRANE actuator profiles
are rejected for external models.

Supported control modes are `pd_joint_pos` (absolute radians/meters),
`pd_joint_delta_pos` (offset from current measured position), `pd_joint_vel`
(radians/meters per second), and `joint_force` (torque/force). Actions must be
finite and shaped `(n_envs, number_of_mapped_joints)`; a vector is accepted only
for one environment. Each `step` advances one `1 / sim_rate` physics interval.
Generic scenes return zero reward, false termination, and truncation at
`max_episode_steps`; they add a ground plane, not task objects.

`JointCommand` combines position and velocity control on disjoint named joints
of the same articulation, independently of the array control mode:

```python
from usim import JointCommand

result = sim.step(
    JointCommand(
        positions={'shoulder': 0.3},
        velocities={'wheel': np.array([1.0, 1.5])},
    )
)
```

Configure `joint_names` to include both arm and wheel joints. Scalar targets
broadcast across environments; per-environment targets must have shape
`(n_envs,)`. A joint cannot appear in both command groups. Unknown joints,
nonfinite values, and invalid batches reject the whole command before changing
controller targets. Unspecified joints keep their previous commands.
`velocity_joint_names` identifies joints that should receive zero velocity
targets at reset rather than a position hold. Array actions still use the
configured `control_mode` across every mapped joint.

## Bundled CRANE-X7

`SimulatorConfig(env_id="PickPlace-CRANE-X7")` retains the bundled robot's rest
pose, gains, cube spawning jitter, dense reward, and success condition. It accepts
either nine mapped values or the compact seven-arm-plus-one-gripper action, whose
last value drives both fingers. `actuator_profile="servo_envelope"` and
`gripper_force_limit` set controller effort bounds. The task intentionally
requires the bundled robot and its default joint mapping.

## Observations and lifecycle

Joint positions and velocities always have shape `(n_envs, mapped_joints)`.
`obs_mode="state"` avoids camera creation. `rgb` supplies uint8 color images;
`rgbd` also supplies float32 depth. `render_mode="none"` disables images, while
`rgb_array` captures off-screen and `human` additionally opens the viewer.
The camera uses 640 by 480 pixels; it is not a robot-mounted sensor. CRANE-X7
retains its workspace viewpoint. External scenes frame the robot's native bounds
at reset, so their images do not inherit the CRANE-X7 close-up coordinates.
Batched scenes render every environment separately rather than copying
one image into the whole batch. RGB shape is `(n_envs, 480, 640, 3)` and depth
shape is `(n_envs, 480, 640)`. Pick/place metrics are in `extra["env_info"]`.

Reset rewinds selected native scene states, zeros velocities, replaces previous
controller targets, and resets task objects without taking a physics step.
It does not advance other environments or reset their episode counters.
Seeded resets reproduce joint noise and cube jitter. `get_observation` refreshes
state without stepping. `close` destroys scene resources and is idempotent;
other Genesis scenes remain usable. Genesis initializes once per process;
switching CPU/CUDA while its runtime is initialized is rejected.

## Verification

Deterministic tests do not require Genesis. With the optional native environment
installed, the opt-in smoke tests exercise external MJCF, external URDF, and
CRANE-X7, plus a free-base wheel-and-arm URDF with mixed named commands, using
two environments, actions, partial reset, observations, and close:

```console
PYTHONPATH="src;packages/genesis/src" USIM_GENESIS_NATIVE=1 USIM_GENESIS_BACKEND=cpu runs/genesis-venv/Scripts/python.exe -m pytest -s test/test_genesis_native.py
```

Use `:` rather than `;` in `PYTHONPATH` on Linux. Set
`USIM_GENESIS_BACKEND=gpu` for CUDA and `USIM_GENESIS_RENDER=rgb_array` for RGB/depth
verification. The first native scene build compiles kernels and can take minutes.

Official API references:
[initial scene](https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/hello_genesis.html),
[joint control](https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/control_your_robot.html),
and [rendering](https://genesis-world.readthedocs.io/en/latest/user_guide/rendering/index.html).
