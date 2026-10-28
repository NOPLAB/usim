"""Episode interface backed by the same owned grouped native session."""

from usim.facets import ControlTargets
from usim.interface import EpisodeSimulator
from usim.model import ModelError, ModelSource
from usim.specs import AgentSpec, ControlSpec, ExecutionSpec, SceneSpec
from usim.types import JointCommand, Observation


class MuJoCoSimulator(EpisodeSimulator):
    gpu = False

    def __init__(self, config):
        from .backend import Backend, WarpBackend

        super().__init__(config)
        self._session = None
        if config.robot_path is None:
            raise ModelError('MuJoCo requires an explicit robot_path to MJCF or URDF')
        if config.robot_uid:
            raise ModelError('MuJoCo loads caller-owned models, not registered robot UIDs')
        if config.robot_init_qpos_noise:
            raise ModelError('MuJoCo requires robot_init_qpos_noise=0; reset is deterministic')
        if config.control_mode not in {'pd_joint_pos', 'pd_joint_vel', 'effort', 'native'}:
            raise ModelError(f'Unknown MuJoCo control mode: {config.control_mode}')
        names = config.joint_names or config.arm_joint_names + config.gripper_joint_names
        positions = tuple(n for n in names if n not in config.velocity_joint_names)
        groups = {
            'pd_joint_pos': (
                positions if config.velocity_joint_names else (),
                config.velocity_joint_names,
                (),
            ),
            'pd_joint_vel': ((), names, ()),
            'effort': ((), (), names),
            'native': ((), (), ()),
        }[config.control_mode]
        control = ControlSpec(
            joint_names=names,
            position_joints=groups[0],
            velocity_joints=groups[1],
            effort_joints=groups[2],
            initial_qpos=config.initial_qpos,
            joint_stiffness=config.joint_stiffness,
            joint_damping=config.joint_damping,
            joint_force_limits=config.joint_force_limits,
        )
        provider = WarpBackend() if self.gpu else Backend()
        session = provider.open(
            SceneSpec(
                agents=(AgentSpec('robot', ModelSource.from_path(config.robot_path), control),)
            ),
            ExecutionSpec(
                backend=config.backend,
                n_envs=config.n_envs,
                sim_rate=config.sim_rate,
                max_episode_steps=config.max_episode_steps,
                render_mode=config.render_mode,
                obs_mode=config.obs_mode,
            ),
        )
        try:
            free_base = any(layout.kind == 'free' for layout in session.schema.joints)
            if config.fixed_base == free_base:
                raise ModelError('fixed_base must match the authored native model')
            components = config.arm_joint_names + config.gripper_joint_names
            if not set(components) <= session.scalar.keys():
                raise ModelError('Component names must be named scalar joints')
        except BaseException:
            session.close()
            raise
        self._session = session
        self._is_running = True

    def _require(self):
        if self._session is None:
            raise ModelError('MuJoCo simulator is closed')
        self._session._require()
        return self._session

    @property
    def joint_names(self):
        return list(self._require().control_names)

    def reset(self, seed=None):
        _, info = self._require().reset(seed)
        return self.get_observation(), info

    def step(self, action):
        s = self._require()
        if isinstance(action, JointCommand):
            writes = {}
            for mode, targets in (
                ('positions', action.positions),
                ('velocities', action.velocities),
            ):
                writes[mode] = {}
                for name, value in targets.items():
                    array = s.np.asarray(value, dtype=float)
                    if array.shape not in {(), (1,)}:
                        raise ModelError('A single environment requires one target per joint')
                    writes[mode][name] = float(array.reshape(-1)[0])
            if writes['positions'] or writes['velocities']:
                s.control.apply(ControlTargets(**writes))
        else:
            values = s.np.asarray(action, dtype=float)
            native = self.config.control_mode == 'native'
            expected = s.model.nu if native else len(s.control_names)
            if values.shape != (expected,) or not s.np.isfinite(values).all():
                raise ModelError(f'MuJoCo action must be a finite vector of shape ({expected},)')
            if native:
                s.data.ctrl[:] = values
            elif expected:
                targets = dict(zip(s.control_names, values))
                partitioned = any(
                    (
                        s.control_spec.position_joints,
                        s.control_spec.velocity_joints,
                        s.control_spec.effort_joints,
                    )
                )
                if partitioned:
                    command = ControlTargets(
                        positions={
                            n: v for n, v in targets.items() if n in s.control_spec.position_joints
                        },
                        velocities={
                            n: v for n, v in targets.items() if n in s.control_spec.velocity_joints
                        },
                        efforts={
                            n: v for n, v in targets.items() if n in s.control_spec.effort_joints
                        },
                    )
                else:
                    mode = {
                        'pd_joint_pos': 'positions',
                        'pd_joint_vel': 'velocities',
                        'effort': 'efforts',
                    }[self.config.control_mode]
                    command = ControlTargets(**{mode: targets})
                s.control.apply(command)
        result = s.step()
        result.observation = self.get_observation()
        return result

    def get_observation(self):
        s = self._require()
        return Observation(
            qpos=self.get_qpos(), qvel=self.get_qvel(), extra={'time': float(s.data.time)}
        )

    def get_qpos(self):
        s = self._require()
        return s.data.qpos[[s.scalar[name][0] for name in s.control_names]].copy()

    def get_qvel(self):
        s = self._require()
        return s.data.qvel[[s.scalar[name][1] for name in s.control_names]].copy()

    def close(self):
        session, self._session = self._session, None
        self._is_running = False
        if session is not None:
            session.close()


class MuJoCoWarpSimulator(MuJoCoSimulator):
    gpu = True
