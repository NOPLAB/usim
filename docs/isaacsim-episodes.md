# Isaac Sim native episodes

`usim_isaacsim.adapter.IsaacSimSimulator` owns a native Isaac Sim 6.1
`SimulationApp`, `World`, and articulation. It is separate from the mobile
`usim_isaacsim.sim` runtime. Use Python 3.12 and a dedicated process and
environment for Isaac; do not combine its dependency set with other engines.
The SDK loads only when constructing the adapter.

```python
import numpy as np
from usim.types import SimulatorConfig
from usim_isaacsim.adapter import IsaacSimSimulator

sim = IsaacSimSimulator(
    SimulatorConfig(
        env_id='JointControl',
        robot_path='/absolute/path/to/robot.urdf',
        joint_names=('hinge',),
        joint_stiffness=(100.0,),
        joint_damping=(10.0,),
        joint_force_limits=(5.0,),
        render_mode='none',
        obs_mode='state',
    )
)
try:
    observation, info = sim.reset(seed=42)
    result = sim.step(observation.qpos + np.zeros(1))
    print(result.observation.qpos, result.reward, result.info)
finally:
    sim.close()
```

## Robots and control

`JointControl` requires caller-owned MJCF `.xml`, URDF, or USD files through
`robot_path`. There is no implicit robot selection. Imported MJCF
and URDF robots honor `fixed_base` (default true); set it false for a mobile
base and arm in one articulation. USD preserves the authored base configuration.
USD inputs must use metres, Z-up, one PhysX articulation, and physical
colliders/inertias. Import outputs go to an owned temporary directory, not
the source asset tree.

```python
config = SimulatorConfig(
    env_id='JointControl',
    robot_uid='my_arm',
    robot_path='/absolute/path/to/arm.urdf',
    arm_joint_names=('shoulder_joint', 'elbow_joint'),
    gripper_joint_names=(),
    initial_qpos=(0.0, 0.0),
    robot_init_qpos_noise=0.0,
    render_mode='none',
    obs_mode='state',
)
```

`joint_names` is the authoritative ordered DOF selection when configured,
independent of the native articulation order. Otherwise joint order is
selected arm joints followed by selected gripper joints. For external robots with no
configured names, all articulation DOFs become arm joints in native order.
Observations are single-environment vectors, not batched arrays.
Actions are absolute PD joint-position targets in radians for revolute
joints and metres for prismatic joints. Targets and reset noise are clipped
to articulation limits. Actions require exactly one target per selected joint;
the caller expands any coupled-joint representation.

PD gains and effort limits retain native model values unless overridden with
`joint_stiffness`, `joint_damping` and `joint_force_limits`. Each tuple follows
the explicitly configured joint order. These are simulator bounds, not calibrated
actuator ratings. Velocity-controlled joints use zero stiffness.
Each `step` advances one native physics timestep of `1/sim_rate`; reset sets
positions, zero velocities and matching targets without a settling step.
Seeded resets reproduce joint noise.

For combined mobile and arm models, specify `velocity_joint_names` within the
selected DOFs. The same articulation then uses zero stiffness velocity drives
for wheel joints and PD position drives for the other joints. Submit disjoint
named mappings through `JointCommand`; joint targets omitted from a command
retain their previous native drive targets.

```python
from usim.types import JointCommand

config = SimulatorConfig(
    env_id='JointControl',
    robot_uid='combined',
    robot_path='/absolute/path/to/combined.urdf',
    fixed_base=False,
    joint_names=('left_wheel', 'shoulder', 'right_wheel'),
    velocity_joint_names=('left_wheel', 'right_wheel'),
    render_mode='none',
    obs_mode='state',
)
# After constructing the simulator with config:
result = sim.step(
    JointCommand(
        positions={'shoulder': 0.3},
        velocities={'left_wheel': 2.0, 'right_wheel': 2.0},
    )
)
```

Wheel velocity targets are in radians per second. Array actions keep their
absolute position semantics for position-only configurations; mixed-mode
configurations require `JointCommand` rather than silently interpreting wheel
slots as positions. Invalid mode/joint mappings fail before any native target
is changed. Reset explicitly clears wheel velocity targets.

## Tasks and cameras

`JointControl` has no task reward: reward is zero, termination is false,
and `info["task_reward"]` is false. Episode limits still truncate it.
Task objects, metrics and rewards belong to the caller. The grouped API can
declare independent scene entities and sensors without a robot-specific recipe.

Use `obs_mode="state"` or `render_mode="none"` to omit camera creation.
`rgb` and `rgbd` use the native RTX sensor at 640 by 480. Set
`camera_uid='scene_camera'` for this adapter; mounted cameras use an explicit
`SensorSpec` through the grouped API. Capturing an observation renders at zero simulation
delta so it does not add a physics timestep. Missing camera frames raise an
error rather than returning a fabricated image.

Unsupported environment IDs, multiple environments, other control modes,
invalid joints and invalid actions fail explicitly. `close()` is idempotent,
stops the world, clears the World singleton, closes Kit, and removes imported
assets. State access after closing raises an error. Start another adapter
in another process rather than restarting Kit in the same interpreter.

## Evidence and runtime smoke

`test/test_isaac_adapter.py` uses deterministic native API fixtures to
exercise import configuration, articulation ordering, gains, actions,
stepping, seeded resets, camera data, cleanup and failures.
Fixtures do not validate the GPU renderer, imported meshes, or PhysX contacts.
The development Python environment has no `isaacsim` package. A direct
bootstrap attempt fails at `from isaacsim import SimulationApp` with
`ModuleNotFoundError: No module named 'isaacsim'`; the alternate existing
environment also has no SDK. Native physics, imported collision meshes and
camera framing therefore remain unverified and must be checked in a separately
installed Isaac 6.1 runtime. No shared environment or lockfile is changed for
this check.

The adapter uses NVIDIA's current
[MJCF importer API](https://docs.isaacsim.omniverse.nvidia.com/latest/importer_exporter/import_mjcf.html)
(`MJCFImporterConfig`, `MJCFImporter.import_mjcf`) rather than deprecated
Kit import commands, following the existing mobile port's 6.1 lifecycle,
URDF importer, articulation and RTX sensor patterns.
