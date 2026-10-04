# Library, ports and ROS Bridge

usim is one Python distribution. Its core has no ROS or simulator imports.
`MobileRobot` describes primitive robot geometry; `render_robot()` produces a
portable URDF. Runtime configuration is separate from authoring, so external
URDFs do not need to be converted into a usim-specific model.

`SimulationConfig`, `Ros2Config`, `Velocity` and `RobotState` are immutable data.
Wheel conversion and `CommandGate` are shared. Motor power starts disabled,
motor requests clear stored commands, and stale commands expire after 0.5 wall
seconds. The gate caps demand at 0.4 m/s and 1 rad/s and rejects nonfinite input.
These limits constrain demand, not instantaneous physical motion.

Each mobile engine implements `Simulator.run(configuration, *, stop=None)`.
The runner owns startup, execution and cleanup. A positive `max_seconds` is
a wall-time deadline after initialization; zero means unlimited.
There is no universal `step()` API: Isaac owns native physics stepping while
Gazebo runs in a separate process. Pretending those are lockstep would hide
timing and lifecycle differences.

`EpisodeSimulator`, `SimulatorConfig`, `Observation`, `StepResult`, `JointCommand`
and the factory describe native episodes (`reset`, `step`, observation and `close`)
in the same `usim` namespace. This API applies where the engine owns native stepping;
Gazebo retains its asynchronous run lifecycle. Execution capabilities do not divide
robots into mobile and manipulator classes. Native engines retain their control
spaces, task rewards and batching; arrays are not silently interpreted as a
universal joint-position action.

`usim` owns interfaces, CLI, factory, ROS bridges and bundled robot descriptions.
`usim_genesis`, `usim_maniskill`, `usim_isaacsim` and `usim_gazebo` are independently
installable engine distributions. Entry points advertise `usim.runners` and
`usim.simulators`; discovery reads metadata without importing engines.
`create_runner()` and `create_simulator()` load only the selected provider.
Core import and CLI help do not load NumPy or native engines.

A combined mobile-arm robot is one articulation in one scene. Its wheel velocity
targets and arm position targets select distinct joints but share physics and
contacts. `JointCommand` represents named mixed targets; engine-specific support
is explicit. `fixed_base=False` allows an imported robot's mobile base to move.
See [engine workflows](engines.md).

The optional `Ros2Client` provides readiness, motor control, velocity publication
and bounded observation. `observe(after=t)` waits for a received state with a
simulation timestamp greater than `t`; it never advances physics.
Readiness requires state, command subscription and motor-service availability.
Publishing a command does not prove it has been applied.

Ports retain native world formats: SDF for Gazebo and collidable metre-scale
Z-up USD for Isaac. URDF geometry and explicit drive parameters are common.
State uses world-source pose, simulation timestamps, XYZW quaternion order and
body-forward/yaw velocity. RGB and depth are asynchronous `rgb8` and metre-valued
`32FC1`; the API does not promise matched camera/odometry frames.

Gazebo Humble runs Python 3.10 in a supervised Docker runtime. The Python 3.12
host exchanges configuration with it and does not load its native ROS modules.
Isaac uses its Python 3.12 runtime and compatible ROS modules. The ROS Bridge is
loaded only when requested; core imports never initialize a native engine.

Application policies, evaluation protocols and robot-specific integrations
belong to consumers, not this library.
