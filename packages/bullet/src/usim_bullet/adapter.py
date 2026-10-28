"""EpisodeSimulator compatibility over the same native grouped Bullet session."""

from usim.interface import EpisodeSimulator
from usim.model import ImportRequirements, ModelError, ModelSource
from usim.specs import AgentSpec, ControlSpec, ExecutionSpec, SceneSpec
from usim.types import JointCommand
from .backend import Backend
from .models import unsupported


class BulletSimulator(EpisodeSimulator):
    def __init__(self, config):
        super().__init__(config)
        self._session = None
        if not config.robot_path:
            raise ModelError('Bullet requires an explicit robot_path')
        if config.robot_uid:
            unsupported('Bullet does not provide registered robot agents').require()
        if config.robot_init_qpos_noise != 0:
            unsupported(
                'Bullet requires explicit initial positions; set robot_init_qpos_noise=0'
            ).require()
        if config.control_mode != 'pd_joint_pos':
            unsupported(
                'Bullet arrays are physical position targets, not native actions'
            ).require()
        groups = config.arm_joint_names + config.gripper_joint_names
        if set(groups) & set(config.velocity_joint_names):
            raise ModelError('Position and velocity joint groups must be disjoint')
        names = config.joint_names or groups
        scene = SceneSpec(
            agents=(
                AgentSpec(
                    'robot',
                    ModelSource.from_path(config.robot_path),
                    control=ControlSpec(
                        joint_names=config.joint_names,
                        position_joints=tuple(
                            n for n in names if n not in config.velocity_joint_names
                        ),
                        velocity_joints=config.velocity_joint_names,
                        initial_qpos=config.initial_qpos,
                        joint_stiffness=config.joint_stiffness,
                        joint_damping=config.joint_damping,
                        joint_force_limits=config.joint_force_limits,
                    ),
                    requirements=ImportRequirements(
                        base='fixed' if config.fixed_base else 'floating'
                    ),
                ),
            )
        )
        self._session = Backend().open(
            scene,
            ExecutionSpec(
                backend=config.backend,
                n_envs=config.n_envs,
                sim_rate=config.sim_rate,
                max_episode_steps=config.max_episode_steps,
                render_mode=config.render_mode,
                obs_mode=config.obs_mode,
            ),
        )
        self._is_running = True

    def _require(self):
        if self._session is None:
            raise ModelError('Bullet simulator is closed')
        self._session._require()
        return self._session

    @property
    def joint_names(self):
        return [joint.name for joint in self._require().state.schema.joints]

    def reset(self, seed=None):
        observation, info = self._require().execution.reset(seed)
        return observation, dict(info)

    def step(self, action):
        import numpy as np

        session = self._require()
        if isinstance(action, JointCommand):
            if action.positions or action.velocities:
                from usim.facets import ControlTargets

                positions = {
                    name: value
                    if isinstance(value, (float, int))
                    else tuple(value.reshape(-1).tolist())
                    for name, value in action.positions.items()
                }
                velocities = {
                    name: value
                    if isinstance(value, (float, int))
                    else tuple(value.reshape(-1).tolist())
                    for name, value in action.velocities.items()
                }
                session.control.apply(ControlTargets(positions, velocities))
        else:
            values = np.asarray(action, dtype=float)
            control = session._controls['robot']
            names = tuple(
                name
                for name in control.joint_names
                if name in control.position_joints + control.velocity_joints
            )
            if values.shape != (len(names),) or not np.isfinite(values).all():
                raise ModelError(f'Bullet action must be a finite vector of shape {(len(names),)}')
            if names:
                from usim.facets import ControlTargets

                targets = dict(zip(names, values.tolist()))
                session.control.apply(
                    ControlTargets(
                        positions={
                            n: v for n, v in targets.items() if n in control.position_joints
                        },
                        velocities={
                            n: v for n, v in targets.items() if n in control.velocity_joints
                        },
                    )
                )
        return session.execution.step()

    def get_observation(self):
        return self._require().state.get_observation()

    def get_qpos(self):
        return self._require().state.get_qpos()

    def get_qvel(self):
        return self._require().state.get_qvel()

    def close(self):
        session, self._session = self._session, None
        self._is_running = False
        if session is not None:
            session.close()
