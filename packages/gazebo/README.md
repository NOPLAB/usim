# Grouped Gazebo Classic backend

`usim.backends:gazebo` opens one owned, headless Gazebo Classic/Humble container.
The existing `GazeboSimulator.run()` container API and ROS motor gate remain available.
The grouped implementation does not expose episode stepping.
Install `usim-gazebo[state]` for host-side ndarray observations, or use the
workspace `gazebo` extra. The legacy runner and model planning remain available
without NumPy; this extra accepts both NumPy 1.x and 2.x engine environments.

```python
from usim import (
    AgentSpec,
    ControlSpec,
    EntitySpec,
    ModelSource,
    SceneSpec,
    SensorSpec,
    create_backend,
)

backend = create_backend('gazebo')
scene = SceneSpec(
    entities=(EntitySpec('object', ModelSource.from_path('object.sdf')),),
    agents=(
        AgentSpec(
            'robot',
            ModelSource.from_path('robot.urdf'),
            control=ControlSpec(velocity_joints=('wheel_joint',)),
            sensors=(SensorSpec('camera', 'rgbd'),),
        ),
    ),
)
with backend.open(scene) as session:
    session.control['robot'].set_motor(True)
    session.control['robot'].set_velocities({'wheel_joint': 0.5})
    session.execution.resume()
    observation = session.state['robot'].get_observation()
    session.execution.pause()
```

Opening builds the world and acknowledges every spawn while physics is paused.
`execution.run()` resumes continuous physics and blocks until `stop()`, session close,
or worker failure. `resume()` and `pause()` permit integration with an existing event
loop. Control writes do not advance physics. Pausing disables active motors.
State is unbatched SI data; `StateSchema` gives joint order and sensor layouts.
Sequential service reads and sensor samples are not an atomic snapshot.

`session.scene.plan(change)` and `apply(plan)` use the core revision/phase checks.
Pose, spawn, delete, and `<link>.mass` changes use acknowledged ROS services.
Joint-position state changes require an available `/set_model_configuration` service;
many stock Humble images do not provide it. Joint velocity state writes and
position-target controllers are not claimed. Velocity targets use ODE joint motors
with authored effort limits. Effort targets require both apply/clear force services.
All controllers start disabled, and failed writes close the owned world.

Replacement and articulation edits require `require_same_episode=False`: they really
delete and respawn the model, reset its state, remove agent/controller/sensor bindings,
and invalidate handles. They do not pretend to preserve an episode.

## Models and resources

`backend.models` is a `ModelIO` composition of actual URDF and SDF parsers. It can
load an asset without registering an agent or opening a native engine. The returned
owned value contains validated XML, link/joint metadata, and staged resource paths;
it is not a claim that native physics has already loaded that model.

SDF selection addresses one top-level model, or a world for `SceneSpec.world`.
Resource references may be relative, file, package, or model URIs with explicit
resource roots. Adjacent texture/material trees remain together and live until the
import/session owner closes. Source files are never edited. Expand SDF includes before
importing. Authored base constraints are retained; unsupported base conversions and
multi-DOF joint layouts are rejected. No USD or MJCF converter is silently assumed.
Authored plugins are stripped to prevent an ungated controller; assets containing
plugins require explicit `ImportRequirements(allow_loss=frozenset({'plugins'}))`.

## Sensors and native requirements

RGB, depth, RGBD, and planar lidar `SensorSpec`s attach to a link of the same robot.
Camera rendering uses a bounded Xvfb readiness acknowledgement and software GL.
Images/scans arrive through native ROS subscriptions; missing frames time out rather
than becoming synthetic observations. Resume physics before requesting the first frame.
Image row stride and byte order are preserved; `16UC1` depth is converted from
millimeters to meters. Additional sensors and their timestamps are in
`observation.extra['sensors']` and `['sensor_metadata']`.

The image must contain Gazebo Classic, Humble `gazebo_ros`, the state/properties
world plugins, camera/ray sensor plugins, and Xvfb for cameras. Native plugin absence
or service rejection is an error, never a successful no-op. The default image name is
`usim-gazebo:local`; `Backend(image=..., engine='docker'|'podman')` selects an image.
GPU execution, GUI rendering, multi-world batching, and episode reset/step are not
provided by this grouped adapter. Fixture tests cover its protocol and service
mapping; native container physics/rendering still requires an explicit native run.
