"""Own one isolated Bullet client, its bodies, and episode state."""

from contextlib import ExitStack
from dataclasses import replace
from importlib import import_module

from usim.capability import Phase
from usim.facets import ControlTargets, JointLayout, StateSchema
from usim.groups import (
    AgentGroup,
    ControlBindings,
    ExecutionGroup,
    JointControlGroup,
    StateBindings,
    StateGroup,
)
from usim.model import ModelError, ModelFormat
from usim.scene import SceneBindings, SceneEditor, SetJointState, SetPose
from usim.types import Observation, StepResult
from .models import unsupported


class Session:
    def __init__(self, backend, spec, options):
        p = import_module('pybullet')

        self.backend, self.models, self.spec, self.options = backend, backend.models, spec, options
        self._p = p
        self._client = None
        self._resources = ExitStack()
        self._phase = Phase.BUILDING
        self._bodies, self._joints, self._controls, self._efforts = {}, {}, {}, {}
        self._steps = 0
        self.agents = AgentGroup(spec.agents)
        self.scene = SceneEditor(
            tuple(item.name for item in spec.entities + spec.agents),
            SceneBindings(self._assess_change, self._apply, lambda: self.phase, self._fail),
        )
        self.execution = ExecutionGroup(episode=self)
        try:
            # Plan and validate every asset before allocating a physics client.
            entries = [(item, item.source, item.requirements) for item in spec.entities] + [
                (item, item.model, item.requirements) for item in spec.agents
            ]
            assets = [
                (
                    item,
                    self._resources.enter_context(
                        self.models.load(self.models.plan(source, requirements))
                    ).value,
                )
                for item, source, requirements in entries
            ]
            client = p.connect(p.GUI if options.render_mode == 'human' else p.DIRECT)
            if client < 0:
                raise ModelError('Bullet could not create a physics client')
            self._client = client
            self._resources.callback(p.disconnect, physicsClientId=client)
            p.setGravity(0, 0, -9.81, physicsClientId=client)
            p.setTimeStep(1 / options.sim_rate, physicsClientId=client)
            p.setRealTimeSimulation(0, physicsClientId=client)
            for item, asset in assets:
                if asset.source.format is ModelFormat.URDF:
                    body = p.loadURDF(
                        str(asset.source.path),
                        item.pose.position,
                        item.pose.orientation_xyzw,
                        useFixedBase=asset.requirements.base == 'fixed',
                        flags=p.URDF_USE_INERTIA_FROM_FILE,
                        physicsClientId=client,
                    )
                else:
                    bodies = p.loadMJCF(str(asset.source.path), physicsClientId=client)
                    if len(bodies) != 1:
                        raise ModelError('Bullet MJCF import must produce exactly one body')
                    body = bodies[0]
                    p.resetBasePositionAndOrientation(
                        body,
                        item.pose.position,
                        item.pose.orientation_xyzw,
                        physicsClientId=client,
                    )
                self._bodies[item.name] = body
                joints = {}
                for index in range(p.getNumJoints(body, physicsClientId=client)):
                    info = p.getJointInfo(body, index, physicsClientId=client)
                    if info[2] == p.JOINT_FIXED:
                        continue
                    if info[2] not in {p.JOINT_REVOLUTE, p.JOINT_PRISMATIC}:
                        raise ModelError('Bullet provider requires scalar hinge or slide joints')
                    name = info[1].decode()
                    if name in joints:
                        raise ModelError('Native active joint names must be unique')
                    joints[name] = (index, info[2], info[10])
                    p.setJointMotorControl2(
                        body, index, p.VELOCITY_CONTROL, force=0, physicsClientId=client
                    )
                if set(joints) != set(asset.joints):
                    raise ModelError(
                        f'Bullet native joint mapping differs from asset: {tuple(joints)}'
                    )
                self._joints[item.name] = joints
            for agent in spec.agents:
                control = agent.control
                names = tuple(self._joints[agent.name])
                if control.joint_names and control.joint_names != names:
                    raise ModelError(f'joint_names must match Bullet native order: {names}')
                groups = control.position_joints + control.velocity_joints + control.effort_joints
                if not set(groups) <= set(names):
                    raise ModelError('Unknown Bullet control joint')
                if control.initial_qpos is not None and len(control.initial_qpos) != len(names):
                    raise ModelError('Initial positions must match Bullet joint count')
                self._controls[agent.name] = replace(
                    control,
                    joint_names=names,
                    position_joints=control.position_joints if groups else names,
                )
                for index, value in enumerate(control.initial_qpos or ()):
                    joint = self._joints[agent.name][names[index]][0]
                    p.resetJointState(
                        self._bodies[agent.name], joint, value, physicsClientId=client
                    )
            self._snapshot = p.saveState(physicsClientId=client)
            self._resources.callback(p.removeState, self._snapshot, physicsClientId=client)
            self._phase = Phase.READY
        except BaseException:
            self._fail()
            raise

    @property
    def phase(self):
        return self._phase

    def _require(self):
        if self._client is None or self.phase in {Phase.CLOSED, Phase.FAILED}:
            raise ModelError(f'Bullet session is {self.phase.value}')
        return self._client

    @property
    def state(self):
        return States(self)

    def state_for(self, name):
        return self.state.for_entity(name)

    @property
    def control(self):
        return Controls(self)

    def assess(self, request):
        report = self.backend.assess(replace(request, phase=self.phase, n_envs=1))
        if not report.supported:
            return report
        if request.operation.startswith('control.'):
            field = {
                'control.position': 'position_joints',
                'control.velocity': 'velocity_joints',
                'control.effort': 'effort_joints',
            }[request.operation]
            if not any(getattr(control, field) for control in self._controls.values()):
                return unsupported('No joints are bound to that control mode')
        if request.operation == 'state.read' and not self._bodies:
            return unsupported('The scene contains no state-bearing entities')
        return report

    def _assess_change(self, change, request):
        report = self.assess(request)
        if not report.supported:
            return report
        if isinstance(change, SetJointState):
            names = self._joints[change.entity]
            if any(name not in names for name, _ in change.positions + change.velocities):
                return unsupported('Joint state edit refers to an unknown active joint')
        return report

    def _apply(self, change):
        client = self._require()
        p = self._p
        if isinstance(change, SetPose):
            p.resetBasePositionAndOrientation(
                self._bodies[change.entity],
                change.pose.position,
                change.pose.orientation_xyzw,
                physicsClientId=client,
            )
        elif isinstance(change, SetJointState):
            body = self._bodies[change.entity]
            positions, velocities = dict(change.positions), dict(change.velocities)
            for name in positions.keys() | velocities.keys():
                index = self._joints[change.entity][name][0]
                current = p.getJointState(body, index, physicsClientId=client)
                p.resetJointState(
                    body,
                    index,
                    positions.get(name, current[0]),
                    velocities.get(name, current[1]),
                    physicsClientId=client,
                )
        else:
            raise ModelError('Unsupported Bullet scene edit')

    def _observation(self):
        if not self._bodies:
            return Observation()
        name = next(iter(self._controls), next(iter(self._bodies)))
        return self.state.for_entity(name).get_observation()

    def reset(self, seed=None):
        client = self._require()
        try:
            self._p.restoreState(self._snapshot, physicsClientId=client)
            self._efforts.clear()
            # restoreState does not reliably restore motor control commands.
            for name, joints in self._joints.items():
                for index, _, _ in joints.values():
                    self._p.setJointMotorControl2(
                        self._bodies[name],
                        index,
                        self._p.VELOCITY_CONTROL,
                        force=0,
                        physicsClientId=client,
                    )
            self._steps = 0
            self._phase = Phase.READY
            return self._observation(), {}
        except BaseException:
            self._fail()
            raise

    def step(self):
        client = self._require()
        if self.phase is Phase.RUNNING:
            raise ModelError('A Bullet step is already running')
        self._phase = Phase.RUNNING
        try:
            # Bullet torque commands last one step; persist submitted effort targets explicitly.
            for (name, joint), value in self._efforts.items():
                self._p.setJointMotorControl2(
                    self._bodies[name],
                    self._joints[name][joint][0],
                    self._p.TORQUE_CONTROL,
                    force=value,
                    physicsClientId=client,
                )
            self._p.stepSimulation(physicsClientId=client)
            self._steps += 1
            result = StepResult(
                self._observation(),
                0.0,
                False,
                self._steps >= self.options.max_episode_steps,
                {'sim_time': self._steps / self.options.sim_rate},
            )
            self._phase = Phase.PAUSED
            return result
        except BaseException:
            self._fail()
            raise

    def close(self):
        self._client = None
        self._phase = Phase.CLOSED
        self._efforts.clear()
        self._resources.close()

    def _fail(self):
        try:
            self.close()
        finally:
            self._phase = Phase.FAILED

    def __enter__(self):
        self._require()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


class States:
    def __init__(self, session):
        session._require()
        self._session = session

    def for_entity(self, name):
        session = self._session
        handle = session.scene.handle(name)
        joints = session._joints[name]

        def values(component):
            import numpy as np

            client = session._require()
            session.scene.resolve(handle)
            return np.asarray(
                [
                    session._p.getJointState(session._bodies[name], index, physicsClientId=client)[
                        component
                    ]
                    for index, _, _ in joints.values()
                ],
                dtype=float,
            )

        def observation():
            client = session._require()
            session.scene.resolve(handle)
            position, orientation = session._p.getBasePositionAndOrientation(
                session._bodies[name], physicsClientId=client
            )
            return Observation(
                qpos=values(0),
                qvel=values(1),
                extra={'pose': position + orientation},
            )

        schema = StateSchema(
            name,
            tuple(
                JointLayout(
                    joint,
                    'prismatic' if kind == session._p.JOINT_PRISMATIC else 'revolute',
                    (i, i + 1),
                    (i, i + 1),
                    'm' if kind == session._p.JOINT_PRISMATIC else 'rad',
                )
                for i, (joint, (_, kind, _)) in enumerate(joints.items())
            ),
            batched=False,
        )
        return StateGroup(schema, StateBindings(observation, lambda: values(0), lambda: values(1)))

    def _primary(self):
        session = self._session
        names = tuple(session._controls) or tuple(session._bodies)
        if len(names) != 1:
            raise ModelError('Select state.for_entity(name) for an empty or multi-entity scene')
        return self.for_entity(names[0])

    @property
    def schema(self):
        return self._primary().schema

    def get_observation(self):
        return self._primary().get_observation()

    def get_qpos(self):
        return self._primary().get_qpos()

    def get_qvel(self):
        return self._primary().get_qvel()


class Controls:
    def __init__(self, session):
        session._require()
        self._session = session

    def for_entity(self, name):
        session = self._session
        handle = session.scene.handle(name)
        if name not in session._controls:
            unsupported('No controller is bound to this Bullet entity').require()
        control = session._controls[name]

        def validate(targets):
            session._require()
            session.scene.resolve(handle)
            if session.phase is Phase.RUNNING:
                raise ModelError('Control writes require an episode safe point')
            for group in (targets.positions, targets.velocities, targets.efforts):
                if any(not isinstance(v, (int, float)) and len(v) != 1 for v in group.values()):
                    raise ModelError('Bullet target batch must contain one environment')
            if control.joint_force_limits is not None:
                for joint, value in targets.efforts.items():
                    value = value if isinstance(value, (int, float)) else value[0]
                    limit = control.joint_force_limits[control.joint_names.index(joint)]
                    if abs(value) > limit:
                        raise ModelError('Effort exceeds the configured joint force limit')

        def write(mode, targets):
            client = session._require()
            p = session._p
            try:
                for joint, value in targets.items():
                    value = value if isinstance(value, (int, float)) else value[0]
                    index, _, maximum = session._joints[name][joint]
                    force = (
                        control.joint_force_limits[control.joint_names.index(joint)]
                        if control.joint_force_limits is not None
                        else maximum
                    )
                    if mode == p.TORQUE_CONTROL:
                        session._efforts[name, joint] = value
                        kwargs = {'force': value}
                    else:
                        kwargs = {
                            'targetPosition'
                            if mode == p.POSITION_CONTROL
                            else 'targetVelocity': value,
                        }
                        # MJCF joint metadata can omit a motor effort limit. In that
                        # case use Bullet's native motor default, not a zero-force motor.
                        if force > 0:
                            kwargs['force'] = force
                    p.setJointMotorControl2(
                        session._bodies[name], index, mode, physicsClientId=client, **kwargs
                    )
            except BaseException:
                session._fail()
                raise

        return JointControlGroup(
            control,
            ControlBindings(
                validate,
                lambda values: write(session._p.POSITION_CONTROL, values),
                lambda values: write(session._p.VELOCITY_CONTROL, values),
                lambda values: write(session._p.TORQUE_CONTROL, values),
            ),
        )

    def _primary(self):
        names = tuple(self._session._controls)
        if len(names) != 1:
            raise ModelError('Select control.for_entity(name) for an empty or multi-agent scene')
        return self.for_entity(names[0])

    def apply(self, targets):
        self._primary().apply(targets)

    def set_positions(self, targets):
        self.apply(ControlTargets(positions=targets))

    def set_velocities(self, targets):
        self.apply(ControlTargets(velocities=targets))

    def set_efforts(self, targets):
        self.apply(ControlTargets(efforts=targets))
