# Bullet, MuJoCo and MuJoCo Warp

These providers use the existing episode and grouped runtime APIs:

| Provider | Package | Physics |
| --- | --- | --- |
| `bullet` | `usim-bullet[native]` | Bullet CPU through PyBullet |
| `mujoco` | `usim-mujoco[native]` | MuJoCo CPU |
| `mujoco-warp` | `usim-mujoco[warp]` | MuJoCo Warp on a CUDA device |

The base packages depend only on `usim`. Discovery, model planning and CLI
help do not initialize the native SDKs. Native libraries are optional, pinned
dependencies reachable from `pyproject.toml`. Use separate Python 3.12
environments for native execution:

```sh
UV_PROJECT_ENVIRONMENT=runs/bullet uv sync --python 3.12 --no-dev --extra bullet
UV_PROJECT_ENVIRONMENT=runs/mujoco uv sync --python 3.12 --no-dev --extra mujoco
UV_PROJECT_ENVIRONMENT=runs/mujoco-warp uv sync --python 3.12 --no-dev --extra mujoco-warp
```

In PowerShell, set `$env:UV_PROJECT_ENVIRONMENT` before the corresponding
command. This keeps existing pipeline and other engine environments intact.

## CLI

Provide your own URDF or MJCF file and ordered joint names. Positive step
counts require an explicit action vector. For position control:

```sh
usim run --engine bullet --compute cpu \
  --robot /absolute/path/robot.urdf --joint-names hinge \
  --render-mode none --obs-mode state --steps 20 --action 0.3

usim run --engine mujoco --compute cpu \
  --robot /absolute/path/robot.xml --joint-names hinge \
  --render-mode none --obs-mode state --steps 20 --action 0.3
```

Run each command inside its matching environment. `--velocity-joints` assigns
selected joints to velocity control; the remaining joints use position targets.
`--joint-stiffness`, `--joint-damping`, `--joint-force-limits` and
`--initial-qpos` follow the explicit `--joint-names` order.
Use `--free-base` when the source and backend support a floating base.
No robot or application policy is supplied implicitly.

The CLI uses strict import-preservation requirements. Use URDF for Bullet CLI
models; its conservative MJCF importer requires the grouped API with explicit
import requirements. MuJoCo physical targets compensate native actuator forces
on the selected joints; native array actions still use the authored actuators.
Its default physical gains are stiffness 100 and damping 10 when not overridden.

All three providers currently expose one environment per session and state
observations. Bullet also supports its native human GUI; MuJoCo and Warp
currently require `--render-mode none`. Bullet rejects physical stiffness and
damping overrides because its native motor gains use different units; MuJoCo
implements physical joint PD targets. Bullet requires its full native joint
order. MuJoCo requires base mobility and placement to be authored in the source.
The MuJoCo providers accept one complete model rather than composing separate
models into a new world.

These are native episode providers, not continuous ROS runners.
Use `run --engine`, not `simulate --backend`, for these engines.
The all-in-one Docker launcher selects their separate interpreters; see
[containers](../docker/README.md).

## GPU execution

`mujoco-warp` uses the official MuJoCo Warp implementation, not a renderer
attached to CPU MuJoCo. It requires `--compute gpu`, the `warp` extra,
a supported CUDA device and driver. Missing CUDA support fails explicitly;
the provider does not fall back to CPU physics.

Allocate a GPU through Slurm before starting GPU computation, for example:

```sh
srun --partition=main --gres=gpu:1 --time=00:10:00 \
  /absolute/path/to/mujoco-warp/bin/python -m usim.cli run \
  --engine mujoco-warp --compute gpu \
  --robot /absolute/path/robot.xml --joint-names hinge \
  --render-mode none --obs-mode state --steps 20 --action 0.3
```

Both the environment and its base Python interpreter must be accessible on the
allocated node. A shared virtual environment can still contain a symlink to
an interpreter installed only on the submission host. Set
`UV_PYTHON_INSTALL_DIR` to a shared directory before installing Python and
creating the GPU environment.

GPU compilation has first-run overhead. CPU tests, base-package discovery and
successful dependency installation do not establish GPU execution.

The reproducible native check exercises grouped motion and reset, the episode
adapter, and the actual CLI in each matching environment:

```sh
python scripts/smoke_added_backends.py --engine bullet
python scripts/smoke_added_backends.py --engine mujoco
srun --partition=main --gres=gpu:1 --time=00:20:00 \
  python scripts/smoke_added_backends.py --engine mujoco-warp
```

## Python

The grouped API keeps physical state and named control in one scene:

```python
from pathlib import Path
from usim import (
    AgentSpec,
    ControlSpec,
    ControlTargets,
    ExecutionSpec,
    ImportRequirements,
    ModelSource,
    SceneSpec,
    create_backend,
)

scene = SceneSpec(
    agents=(
        AgentSpec(
            'robot',
            ModelSource.from_path(Path('/absolute/path/robot.urdf')),
            ControlSpec(joint_names=('hinge',), position_joints=('hinge',)),
            requirements=ImportRequirements(base='fixed'),
        ),
    )
)
with create_backend('bullet').open(scene, ExecutionSpec()) as session:
    observation, info = session.execution.reset(seed=7)
    session.control.for_entity('robot').apply(ControlTargets(positions={'hinge': 0.3}))
    result = session.execution.step()
    positions = session.state.for_entity('robot').get_qpos()
```

Choose `mujoco` for CPU MuJoCo or `mujoco-warp` with
`ExecutionSpec(backend='gpu')` for GPU physics. Inspect `describe()`,
`assess()` and model import reports before requesting formats, sensors,
batching or scene edits; unsupported behavior is rejected rather than emulated.

## Verification record

Verified in `/mnt/workspace/nop/usim` on 2026-10-27:

| Check | Result |
| --- | --- |
| Bullet CPU grouped control, reset, episode adapter and CLI | Passed; joint moved from `0.1` to `0.2996406` toward `0.3` |
| MuJoCo CPU grouped control, reset, episode adapter and CLI | Passed; joint reached `0.2986723` |
| MuJoCo Warp grouped control, reset, episode adapter and CLI | Passed on a Slurm-allocated RTX PRO 6000 Blackwell GPU; joint reached `0.2986723` |
| Full CPU test suite, including native Bullet and opt-in MuJoCo tests | 1,307 passed, 60 skipped |
| Ruff | Lint and formatting passed |
| Seven distributions | Source distributions and wheels built successfully |
| Installed wheel checks outside the checkout | Core-only import guard and all-provider CLI smoke passed |

The skipped tests require optional USD/CAD/other engine runtimes. The test run
also emitted two existing VTC import deprecation warnings. GPU execution used
the `main` Slurm partition on `mechanigaharakizashi`; no GPU computation was
run on an unallocated local device.

Full backend static checking is not clean: basedpyright with the native CPU
environment reported two MuJoCo binding attribute errors and 880 warnings
about missing stubs, annotations and native type inference. These diagnostics
are not suppressed. Native execution confirms the binding attributes exist.
Container launcher routing is covered by tests; rebuilding the all-SDK Docker
image was not part of this native verification.
