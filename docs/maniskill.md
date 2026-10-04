# ManiSkill

Use a separate Python 3.12 environment from Isaac. This adapter targets
`mani-skill==3.0.1` and `sapien==3.0.3`; native simulation requires a supported
SAPIEN platform. Linux supports CUDA PhysX parallel environments; Windows CPU
physics and RGBD observation have also been verified. The adapter uses CPU
image buffers when `backend='cpu'`, avoiding unintended CUDA interop.

```sh
uv venv --python 3.12 /tmp/usim-maniskill
uv pip install --python /tmp/usim-maniskill/bin/python . './packages/maniskill[native]'
```

For Windows, use a Python 3.12 environment and its `Scripts/python.exe`.
The tested setup is `runs/maniskill-native-venv`. Windows CUDA PhysX currently
fails in upstream SAPIEN with a missing `cuda.dll`; use the explicit CPU backend
there rather than silently changing a GPU request. WSL additionally needs a
working NVIDIA Vulkan ICD; CUDA visibility alone does not provide rendering.

Verified CRANE reset, action, RGBD observation and close:

```sh
usim run --engine maniskill --env-id PickPlace-CRANE-X7 --compute cpu \
  --obs-mode rgbd --render-mode rgb_array --steps 2 \
  --action 0 0.392699 0 -1.963495 0 -0.785398 1.570796 0.02
```

Standard registered ManiSkill tasks and robots do not load the bundled CRANE
agent or its assets:

```python
from usim import SimulatorConfig, create_simulator

sim = create_simulator(
    'maniskill',
    SimulatorConfig(
        env_id='PickCube-v1',
        robot_uid='panda',
        backend='auto',
        n_envs=2,
        obs_mode='rgb+depth',
        camera_uid='base_camera',
        sim_rate=30,
        max_episode_steps=100,
    ),
)
try:
    obs, info = sim.reset(seed=42)
    print(sim.joint_names, obs.qpos.shape, obs.rgb_image.shape)
    action = sim._env.action_space.sample()
    result = sim.step(action)
    print(result.reward, result.terminated, result.truncated)
finally:
    sim.close()
```

For one environment, joint arrays are `(joints,)`, images are `(H, W, C)`,
and rewards and episode flags are Python scalars. With `n_envs > 1`, arrays,
images, rewards, and flags retain their leading environment dimension. Raw
observations and info retain ManiSkill's original batched tensors. Actions use
the registered agent's native controller layout, not `joint_names`: mimic
grippers can have several joints but one action coordinate. The private `_env`
access in the example is only for inspecting the native action space.

Full `joint_names` come from the loaded articulation in native order, including
mobile joints outside the arm/gripper groups. Optional configured `joint_names`
must match this order. `arm_joint_names` and `gripper_joint_names` can override
the reported groups without changing joint observation or action ordering.
`initial_qpos`, when supplied, follows full native order on every reset.
Named `JointCommand` is explicitly rejected; use the registered controller's
native action vector and scaling. `velocity_joint_names` validates named joints
without changing the registered controller. `fixed_base` must match the loaded
agent's root; it cannot override a registered model's base mobility.
Camera IDs are task-specific: an unknown
ID raises an error listing the cameras present, rather than returning empty
images. State-only observations need no camera. Depth retains ManiSkill's native
units and channel layout.
Use `render_mode="none"` with state-only observations to disable the renderer;
visual observations still require a working Vulkan device.
Some registered tasks also create rendering materials during scene construction,
so a state-only headless run can still require a rendering-capable device.

The installed public command supports reset/observe without guessing actions:

```sh
usim run --engine maniskill --env-id PickCube-v1 --robot-uid panda \
  --compute cpu --obs-mode state --render-mode none --steps 0
```

Use an explicit `--action` vector for stepping. The backend smoke also checks
initial positions, two-step truncation, and idempotent closure:

```sh
python -m usim_maniskill.scripts.native_smoke
python -m usim_maniskill.scripts.native_smoke --backend gpu --num-envs 2
```

`sim_rate` is an integer control frequency in Hz; PhysX runs five substeps per
control step. `max_episode_steps` and `n_envs` are passed to the registered
environment. `backend="cpu"` only supports one environment; use `"auto"` or
`"gpu"` for a batch. Tasks still determine which robots they support.
An explicit `robot_path` is rejected: custom models must first be registered as
ManiSkill agents and selected through `robot_uid`.

The `PickPlace-CRANE-X7` compatibility task, CRANE force profiles and overrides,
hand/scene cameras, and `usim_maniskill.visual_domains` remain available.
CRANE-specific force overrides are rejected for other registered robots.

Upstream batch behavior:
<https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/quickstart.html>
