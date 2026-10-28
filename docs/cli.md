# Simulator selection

Use the existing `run --engine` for native episodes and `simulate --backend`
for continuous execution. The CLI discovers providers through the public usim
factory API without loading native SDKs for help.
Install the matching engine distribution in its own native environment; see
[engine installation](engines.md) and [Docker runtimes](../docker/README.md).

```sh
usim backends
usim simulate --help
usim run --help
```

The command selects the execution capability, not the robot category:

| Command | Simulators | Public API |
| --- | --- | --- |
| `simulate` | Gazebo, Isaac Sim | `create_runner(...).run(SimulationConfig(...))` |
| `run --robot` | Installed grouped backends supporting episodes | `create_backend(...).open(SceneSpec(...), ExecutionSpec(...))` |
| `run --robot-uid` | Installed native episode providers | `create_simulator(..., SimulatorConfig(...))` |

Other installed providers are accepted when they register the corresponding
execution capability. Unsupported combinations fail during argument parsing.
Gazebo has no lockstep episode API; Genesis and ManiSkill do not expose the
continuous ROS runner. The CLI does not invent these capabilities.

## Continuous execution

```sh
usim simulate --backend gazebo --world /models/room.sdf \
  --robot-urdf /models/robot.urdf --engine docker --headless --max-seconds 5
usim simulate --backend isaacsim --world /models/room.usd \
  --robot-urdf /models/robot.urdf --no-ros --headless --max-seconds 5
```

Wheel, sensor, ROS topic and stdin supervision options retain their existing
behavior. `--engine docker|podman` is the Gazebo container engine, distinct
from `--backend`. Container options require Gazebo.

## Native episodes

Supply a caller-owned model and its controlled joints, or an explicitly
registered native task and robot. Actions remain in the engine's native space:

```sh
usim run --engine genesis --compute cpu --robot /models/robot.urdf \
  --joint-names hinge --joint-stiffness 100 --joint-damping 10 \
  --render-mode none --obs-mode state --steps 1 --action 0.2
usim run --engine maniskill --compute cpu --robot /models/robot.urdf \
  --joint-names hinge --render-mode none --obs-mode state --steps 1 --action 0.2
usim run --engine isaacsim --compute cpu --robot /models/robot.usd \
  --joint-names hinge --render-mode none --obs-mode state --steps 1 --action 0.2
```

External models use `ModelSource`, named control groups and a context-managed
runtime session. `--action` follows `--joint-names` (or arm then gripper names);
`--velocity-joints` selects velocity targets and other joints receive position
targets. The backend broadcasts scalar targets across environments.
Registered tasks retain their native array action semantics.

The existing `simulate --backend isaac` alias still selects `isaacsim`.
The default for `simulate` remains Isaac Sim; `run` requires an explicit
`--engine`. Use `--steps 0` to reset and observe without submitting an action.
