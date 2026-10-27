# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Exercise the installed grouped backend against real, temporary robot models.

Run in an engine's existing environment, for example:
    uv run --extra genesis python examples/grouped_smoke.py --engine genesis
    uv run --extra maniskill python examples/grouped_smoke.py --engine maniskill
    uv run --extra isaac python examples/grouped_smoke.py --engine isaacsim
    uv run --extra gazebo python examples/grouped_smoke.py --engine gazebo
"""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final

from usim import (
    AgentSpec,
    CapabilityRequest,
    ControlSpec,
    ControlTargets,
    EntitySpec,
    ExecutionSpec,
    ImportRequirements,
    ModelError,
    ModelSource,
    Pose,
    SceneSpec,
    SetJointState,
    SetPose,
    SpawnEntity,
    create_backend,
)


INERTIA: Final = (
    '<inertial><mass value="1"/>'
    '<inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>'
)
GEOMETRY: Final = (
    '<visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual>'
    '<collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision>'
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--engine', choices=('genesis', 'maniskill', 'isaacsim', 'gazebo'), required=True
    )
    args = parser.parse_args()
    provider = create_backend(args.engine)
    with TemporaryDirectory(prefix='usim-grouped-smoke-') as temporary:
        directory = Path(temporary)
        robot = directory / 'robot.urdf'
        robot.write_text(
            '<robot name="robot"><link name="base">'
            + INERTIA
            + GEOMETRY
            + '</link><link name="arm">'
            + INERTIA
            + GEOMETRY
            + '</link>'
            '<joint name="hinge" type="revolute"><parent link="base"/><child link="arm"/>'
            '<origin xyz="0 0 0.2"/><axis xyz="0 0 1"/>'
            '<limit lower="-1" upper="1" effort="10" velocity="2"/></joint></robot>',
            encoding='utf-8',
        )
        box = directory / 'box.urdf'
        box.write_text(
            '<robot name="box"><link name="box">' + INERTIA + GEOMETRY + '</link></robot>',
            encoding='utf-8',
        )
        scene = SceneSpec(
            agents=(
                AgentSpec(
                    'robot',
                    ModelSource.from_path(robot),
                    ControlSpec(
                        joint_names=('hinge',), position_joints=('hinge',), initial_qpos=(0.0,)
                    ),
                    requirements=ImportRequirements(base='fixed'),
                ),
            )
        )
        continuous = args.engine == 'gazebo'
        if continuous:
            scene = SceneSpec(
                entities=(
                    EntitySpec(
                        'robot',
                        ModelSource.from_path(robot),
                        requirements=ImportRequirements(),
                    ),
                )
            )
        options = ExecutionSpec(mode='continuous' if continuous else 'episode')
        with provider.open(scene, options) as session:
            joint_state_available = True
            if continuous:
                joint_state_available = session.assess(
                    CapabilityRequest('state.read', phase=session.phase)
                ).supported
            initial = None
            if not continuous:
                initial = session.state.get_qpos().copy()
                session.execution.reset(seed=7)
                session.control.for_entity('robot').apply(ControlTargets(positions={'hinge': 0.2}))
                for _ in range(8):
                    session.execution.step()
                moved = session.state.get_qpos().copy()
                if abs(float(moved.reshape(-1)[0]) - float(initial.reshape(-1)[0])) <= 1e-6:
                    raise ModelError('Native stepping did not move the commanded joint')
            else:
                if joint_state_available:
                    initial = session.state.get_qpos().copy()
                edit = session.scene.plan(SetJointState('robot', positions=(('hinge', 0.2),)))
                moved = None
                if edit.support.supported:
                    session.scene.apply(edit)
                    moved = session.state.get_qpos().copy()
                    if abs(float(moved.reshape(-1)[0]) - 0.2) > 1e-4:
                        raise ModelError('Gazebo did not apply the acknowledged joint state')
            session.scene.apply(
                session.scene.plan(SetPose('robot', Pose(position=(0.0, 0.0, 0.5))))
            )
            if continuous:
                observed = session.state.for_entity('robot').get_observation()
                match observed.extra.get('pose'):
                    case Pose(position=(x, y, z)):
                        if max(abs(x), abs(y), abs(z - 0.5)) > 1e-5:
                            raise ModelError('Native pose differs from the acknowledged edit')
                    case _:
                        raise ModelError('Gazebo did not return its native pose')
            old_handle = session.scene.handle('robot')
            edit = session.scene.plan(
                SpawnEntity(
                    EntitySpec(
                        'box',
                        ModelSource.from_path(box),
                        Pose(position=(1.0, 0.0, 0.2)),
                        ImportRequirements(
                            role='rigid_body', base='authored' if continuous else 'fixed'
                        ),
                    )
                ),
                require_same_episode=False,
            )
            session.scene.apply(edit)
            stale_rejected = False
            try:
                session.scene.resolve(old_handle)
            except ModelError:
                stale_rejected = True
            if not stale_rejected:
                raise ModelError('A topology edit did not invalidate the old native handle')
            output = {
                'engine': args.engine,
                'initial_qpos': initial.tolist() if initial is not None else None,
                'commanded_qpos': moved.tolist() if moved is not None else None,
                'scene_revision': session.scene.revision,
                'box': session.scene.resolve(session.scene.handle('box')),
                'edit_effects': list(edit.support.effects),
                'stale_handle_rejected': stale_rejected,
            }
        print(json.dumps(output))


if __name__ == '__main__':
    main()
