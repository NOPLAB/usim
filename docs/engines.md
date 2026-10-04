# Engine packages and common API

The workspace has five independently buildable distributions:

| Distribution / import | Responsibility |
| --- | --- |
| `usim` | Interfaces, configuration, CLI, provider discovery, ROS bridges and robot assets |
| `usim-genesis` / `usim_genesis` | Genesis scenes and native robot control |
| `usim-isaacsim` / `usim_isaacsim` | Isaac continuous execution and native episodes |
| `usim-maniskill` / `usim_maniskill` | ManiSkill registered robots, tasks and native controllers |
| `usim-gazebo` / `usim_gazebo` | Supervised Gazebo execution and container staging |

The initial source-preservation checkpoint is recorded in
[migration-manifest.json](migration-manifest.json). Model meshes were preserved
byte-for-byte; text line endings were normalized. Recorded checkpoint hashes
precede subsequent engine enhancements.

## Installation

From the workspace checkout, install one engine's native environment:

```sh
uv sync --python 3.12 --extra genesis
# Or, in a separate environment:
uv sync --python 3.12 --extra maniskill
# Or:
uv sync --python 3.12 --extra isaac
# Gazebo executes in its documented container runtime:
uv sync --python 3.12 --extra gazebo
```

Each backend's base package depends only on `usim`; its `native` extra supplies
native libraries. `usim` itself has no runtime dependencies. A downstream project
can install `usim-genesis[native]`, `usim-maniskill[native]` or
`usim-isaacsim[native]` rather than depending on the whole workspace.
Isaac's Pillow/NumPy requirements conflict with Genesis and ManiSkill; do not
combine their native environments or override native dependency bounds.

## CLI

The CLI belongs to `usim` and does not import engines for help or discovery:

```sh
uv run usim backends
uv run usim run --help
uv run usim simulate --help
```

`run` resets and observes an installed native episode provider. Positive
`--steps` requires an explicit `--action` in that engine's native control space.
There is no generic policy or guessed action conversion in the CLI:

```sh
uv run --extra genesis usim run --engine genesis --compute cpu \
  --env-id PickPlace-CRANE-X7 --render-mode none --obs-mode state
uv run --extra maniskill usim run --engine maniskill --compute gpu \
  --env-id PickPlace-CRANE-X7 --render-mode rgb_array --obs-mode rgb
uv run --extra isaac usim run --engine isaacsim --compute gpu \
  --env-id JointControl --render-mode none --obs-mode state
```

`simulate` launches a continuous runner using a native world and URDF. Existing
wheel/ROS options remain available; see [Gazebo](gazebo.md) and [Isaac](isaac.md).
The difference is execution capability, not robot category.

## Python API

```python
from usim import SimulatorConfig, create_simulator

simulator = create_simulator(
    'maniskill',
    SimulatorConfig(env_id='PickPlace-CRANE-X7', backend='gpu'),
)
try:
    observation, info = simulator.reset(seed=0)
    # Supply an action produced for this task's native controller.
    action = [0.0, 0.39, 0.0, -1.96, 0.0, -0.79, 1.57, 0.0]
    result = simulator.step(action)
finally:
    simulator.close()
```

Use `create_runner("gazebo")` or `create_runner("isaacsim")` with
`SimulationConfig` for supervised continuous execution. `capabilities(engine)`
reports installed `run` and `episode` providers without loading native modules.
Selecting an unsupported capability fails before launching an engine.

`SimulatorConfig` provides robot model/UID, full ordered `joint_names`, optional
component joint groups, `initial_qpos`, `fixed_base`, native control/observation
mode and batch size. It does not classify a robot as mobile or manipulator.
State/action shape and native mode support are documented by each engine.

## One mobile-arm robot

An external MJCF/URDF can contain wheels, arm and gripper in one articulation.
Set `fixed_base=False` to preserve base motion, and provide the controlled joint
order. Where supported, `JointCommand` applies named position and velocity
targets to disjoint joints in that same scene:

```python
from usim import JointCommand

command = JointCommand(
    velocities={'left_wheel': 1.0, 'right_wheel': 1.0},
    positions={'shoulder': 0.3, 'elbow': -0.4},
)
```

Genesis and Isaac map these targets to native joint indices. ManiSkill uses its
registered controller action space; a named mixed command is rejected if that
controller does not define it. Do not split the base and arm across two simulators.
Gazebo remains a continuous ROS runtime and does not promise lockstep episodes.

## Runtime evidence

Adapter fixtures, core tests and wheel builds do not establish native execution.
Run reset/action/observation/close in the installed engine environment, and use
the ROS/container smokes for continuous runners.
See [Genesis](genesis.md), [ManiSkill](maniskill.md) and
[Isaac native episodes](isaacsim-episodes.md) for supported modes and platform limits.
