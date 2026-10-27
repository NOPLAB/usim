"""Per-session unregistered BaseAgent classes with explicit physical controls."""

from usim.model import ModelError
from usim.specs import AgentSpec
from usim_maniskill.models import Asset


def _agent_factory(spec: AgentSpec, asset: Asset):
    from mani_skill.agents.base_agent import BaseAgent
    from mani_skill.agents.controllers import (
        PassiveControllerConfig,
        PDJointPosControllerConfig,
        PDJointVelControllerConfig,
    )

    class DirectAgent(BaseAgent):
        uid = spec.name
        fix_root_link = asset.fixed_base

        def _load_articulation(self, initial_pose=None):
            from usim_maniskill.native import build_asset

            self.robot = build_asset(self.scene, asset, spec.name, spec.pose)
            self.robot_link_names = [link.name for link in self.robot.get_links()]

        @property
        def _controller_configs(self):
            names = tuple(joint.name for joint in self.robot.get_active_joints())
            control = spec.control
            selected = control.position_joints + control.velocity_joints + control.effort_joints
            if not set(selected) <= set(names):
                raise ModelError('Explicit controls refer to unknown native joints')
            if control.joint_names and control.joint_names != names:
                raise ModelError(f'joint_names must match native state order: {names}')
            positions = control.position_joints if selected else names
            position_kwargs = {}
            velocity_kwargs = {}
            for values, key in (
                (control.joint_stiffness, 'stiffness'),
                (control.joint_damping, 'damping'),
                (control.joint_force_limits, 'force_limit'),
            ):
                if values is not None:
                    by_name = dict(zip(control.joint_names, values, strict=True))
                    position_kwargs[key] = [by_name[name] for name in positions]
                    if key != 'stiffness':
                        velocity_kwargs[key] = [by_name[name] for name in control.velocity_joints]
            configs = {'balance_passive_force': False}
            if positions:
                configs['positions'] = PDJointPosControllerConfig(
                    list(positions),
                    lower=None,
                    upper=None,
                    normalize_action=False,
                    **{'stiffness': 100.0, 'damping': 10.0, **position_kwargs},
                )
            if control.velocity_joints:
                configs['velocities'] = PDJointVelControllerConfig(
                    list(control.velocity_joints),
                    lower=-float('inf'),
                    upper=float('inf'),
                    normalize_action=False,
                    **{'damping': 10.0, **velocity_kwargs},
                )
            passive = tuple(
                name for name in names if name not in positions + control.velocity_joints
            )
            if passive:
                configs['passive'] = PassiveControllerConfig(list(passive), damping=0.0)
            return {'pd_joint_pos': configs}

    return DirectAgent
