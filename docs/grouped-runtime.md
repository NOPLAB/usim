# Composable models and grouped runtime

The grouped API is additive. Existing `create_simulator`, `create_runner`,
`SimulatorConfig`, `SimulationConfig`, and engine-specific task adapters remain
available. The core still has no runtime dependencies.

## Boundaries

`ModelSource` describes a file, its format, an optional selected subtree and
explicit resource roots. XML detection reads the root tag rather than treating
every `.xml` as MJCF. Use absolute or caller-resolved paths; a backend must not
infer a downstream checkout.

`EntitySpec` adds placement and import requirements. `ControlSpec` specifies
joint order and disjoint position, velocity and effort groups. `AgentSpec`
combines a model, controls and sensors. `RegisteredAgentRef` is a separate
backend-qualified recipe; importing geometry never invents a controller or task.
`SceneSpec` contains independent entities, agents and an optional native world.
`ExecutionSpec` selects episode or continuous execution, not a robot category.

## Discover without initializing physics

```python
from usim import CapabilityRequest, create_backend, list_backends

print(list_backends())
backend = create_backend('genesis')
print(backend.describe())
print(backend.assess(CapabilityRequest('scene.spawn')))
```

Provider discovery uses the `usim.backends` entry point. `describe()` reports
declared operations and candidate formats. It does not prove an arbitrary
model or current runtime can execute an operation. Use `models.plan()` for
asset-specific assessment and `scene.plan()` for runtime edits.

## Small loaders, grouped model access

Each `FormatLoader[T]` supplies one format, `assess(source, requirements)` and
`load(source, context)`. Each `ModelConverter` declares its directed source and
target formats, cost, assessment and conversion. `ModelIO.compose(loaders,
transports)` implements the grouped planner and loader.

For example, a USD loader combined with URDF-to-USD and MJCF-to-USD transports
automatically provides three candidate input formats. Registering a converter
does not imply its reverse exists. No canonical intermediate format or generic
USD-to-URDF converter is assumed.

```python
from usim import ImportRequirements, ModelIO, ModelSource

# Supply actual implementations from an engine or consumer.
models = ModelIO.compose(loaders=(usd_loader,), transports=(urdf_to_usd,))
plan = models.plan(
    ModelSource.from_path('robot.urdf'),
    ImportRequirements(preserve=frozenset({'joint_structure', 'inertia', 'collision'})),
)
with models.load(plan) as imported:
    print(imported.report.route, imported.report.joint_mapping)
    consume(imported.value)
```

The identifiers in this composition snippet stand for caller-supplied providers,
not built-in universal converters. A provider may produce a parsed asset for
scene construction or a native imported asset; an asset lease alone is not a
running physical entity. Isaac import transports require an owned native import
context. Keep import leases alive until their consuming scene is closed.

Conversion planning cannot inspect an intermediate file that does not exist yet.
Such routes are explicitly `conditional`. Loading executes exactly that route,
assesses each actual intermediate and validates the native result. It never
silently retries through another route.

Required preservation must be proven at every stage. `unknown`, `changed` and
`lost` features do not satisfy `preserve`. A loss outside `allow_loss` rejects the
import. Reports retain transformations, missing evidence and source-to-native
joint/link mappings; they are not a claim of numerically equivalent physics.

Plans are bound to their composition and asset fingerprint. Include external
meshes, textures, includes and USD dependencies in `ModelSource.resources` when
they must be part of stale-plan detection. A changed source or resource requires
a fresh plan. Converted assets live in an owned directory, including when the
caller supplies a parent import directory.

## Open, control and observe

```python
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
            ModelSource.from_path('robot.urdf'),
            ControlSpec(
                joint_names=('wheel', 'arm'),
                velocity_joints=('wheel',),
                position_joints=('arm',),
            ),
            requirements=ImportRequirements(base='floating'),
        ),
    )
)
backend = create_backend('genesis')
with backend.open(scene, ExecutionSpec()) as session:
    session.execution.reset(seed=7)
    session.control.apply(
        ControlTargets(
            positions={'arm': 0.2},
            velocities={'wheel': 1.0},
        )
    )
    result = session.execution.step()
    print(session.state.schema, result.observation.qpos)
```

Control target writers do not advance physics. The grouped execution driver owns
advancement. `state.for_entity(name)` and `control.for_entity(name)` select an
entity explicitly; the primary-entity convenience access must reject ambiguity.
Joint layouts state separate qpos and qvel ranges, units, quaternion order and
batching. Do not assume every joint occupies one coordinate or every engine
uses the same batch shape. Sensor layouts describe their native frames and units.
Genesis USD joint identities and import mappings use full prim paths. Named
controls also accept a basename when it identifies exactly one joint; ambiguous
basenames require an explicit path. Joint names supplied in `ControlSpec` remain
the control contract rather than being silently replaced by an importer alias.

Continuous execution exposes `run` and `stop`, not invented episode stepping.
Gazebo service state reads are asynchronous observations, not synchronized
multi-joint snapshots. A command acknowledgement does not prove physical motion.
`session.assess(request)` uses the live phase and selected execution mode;
caller-supplied phase fields cannot authorize access to a closed or running
session. Backend-level assessment instead describes a prospective configuration.

## Backend mapping

| Backend | Models | Execution | Topology changes |
| --- | --- | --- | --- |
| Genesis | Parsed URDF, MJCF and optional USD | Native episodes, CPU/CUDA batches | Rebuild with explicit new-episode permission |
| ManiSkill | Direct URDF/MJCF and registered agents | Native episodes; batching follows native backend constraints | Reconstruct/reconfigure with explicit episode reset |
| Isaac | Native USD plus URDF/MJCF-to-USD transports | Native episodes or continuous native execution | Rebuild and rebind all native views |
| Gazebo | Parsed/resource-owned URDF and SDF | Asynchronous supervised continuous execution | Acknowledged spawn/delete; replacement deletes and respawns |

Importers reject unsupported subtrees, joint types and base conversions rather
than infer a different robot. MJCF global simulation settings, inertial
conversion and engine-specific plugins may require explicit loss permission.
Check the actual report for each model, not just this format table.
Genesis USD imports additionally need `usim-genesis[usd]`; install
`usim-genesis[native,usd]` in its dedicated native environment to enable all
three native file loaders. Keeping the USD parser optional preserves existing
environment combinations and does not install a second USD runtime into Isaac.

Gazebo's live support depends on available ROS services. In particular,
`SetModelConfiguration` is absent from some Humble images; planning a named
joint-state position edit then reports unsupported before attempting it.
Position-target controllers are not manufactured from position state setters.
Sensor frames require native rendering and message production; state-only
fixtures and smokes do not validate rendering.

## Change an existing scene

```python
from usim import EntitySpec, ImportRequirements, ModelSource, SpawnEntity

plan = session.scene.plan(
    SpawnEntity(
        EntitySpec(
            'obstacle',
            ModelSource.from_path('obstacle.urdf'),
            requirements=ImportRequirements(role='rigid_body', base='fixed'),
        )
    ),
    require_same_episode=False,
)
print(plan.support)
session.scene.apply(plan)
```

Changes distinguish pose, joint state, physical parameters, adding/removing an
entity and replacing articulation topology. An `EditArticulation` supplies a
replacement model; it is not an arbitrary engine-independent graph-edit language.
The assessment identifies safe-point requirements, scope, lost state and
reinitialization effects. Rebuilding is never described as preserving an episode.
The default `require_same_episode=True` rejects edits requiring a new episode.

An edit plan belongs to one scene revision and phase. Conditions are assessed
again before native mutation. Structural changes invalidate old handles; pose
and parameter changes do not. Closing or failing a session invalidates access.
Reacquire `state.for_entity` and `control.for_entity` after structural changes.
There is no cross-engine rollback promise: a failed native mutation marks the
session failed and requires cleanup rather than allowing uncertain continuation.

## Verification

`examples/grouped_smoke.py` creates real temporary URDFs, opens the selected
backend, applies targets or acknowledged state changes, observes native state,
adds an independent object and verifies handle invalidation. Run it in the
matching engine environment; it does not install or replace native SDKs.
It deliberately uses CPU configuration and state observations. Isaac still needs
its supported hardware/runtime. GPU computations require the site's GPU
allocation; the example does not request or bypass one.

Fixture tests establish Python contracts, dispatch and resource ownership.
They do not establish native physics, renderer quality or native process cleanup.

Additional opt-in tests use actual engine parsers and physics:

```sh
USIM_GROUPED_NATIVE=genesis python -m pytest -s test/test_grouped_native.py -k real_model
USIM_GROUPED_NATIVE=maniskill python -m pytest -s test/test_grouped_native.py -k real_model
USIM_GAZEBO_GROUPED_NATIVE=1 python -m pytest -s test/test_grouped_native.py -k gazebo_sensor
```

Use the dedicated environment and source paths for the selected engine. The
Genesis test covers URDF, MJCF and authored USD; ManiSkill covers URDF/MJCF with
explicit acceptance of its MJCF inertia loss. The Gazebo test starts native
software rendering and verifies real RGB, depth and lidar measurements of an
authored obstacle. `-s` keeps the native logger's output stream alive at shutdown.

On October 27, 2026, the grouped smoke passed against the existing Genesis and
ManiSkill CPU environments and the local `usim-gazebo:local` Docker image:
Genesis moved the commanded joint from 0 to approximately 0.19057 radians,
and ManiSkill to approximately 0.18625 radians. Both rebuilt a scene with an
additional object and invalidated the old handle. Gazebo loaded native entities,
read joint state, applied a pose service and added an independent object; its
joint-position state service was unavailable. These are state/scene checks,
not proof of equivalence across engines. Separate native Gazebo RGB-D/lidar
tests also passed. Genesis primitive USD control passed after resolving the
SDK's full-path joint names; optional Omniverse baking remains unavailable.

The local Isaac SDK and official 6.1 image were unavailable, and the historical
native verification host rejected SSH authentication. Isaac fixture results
must not be reported as native physics or GUI verification.

The new portable core passes an independent static type check without errors or
warnings. Full backend static checks are not clean: the host environment lacks
ROS/Isaac SDK modules and OpenUSD typing information, and native wrappers still
produce inferred-type, internal-access and type-reference-cycle diagnostics.
Runtime fixtures and native checks above do not replace those missing static
guarantees.
