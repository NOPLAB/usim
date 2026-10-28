# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Check native grouped, episode and CLI control in an installed engine environment.

Run with that environment's Python, not an isolated script dependency environment:
    python scripts/smoke_added_backends.py --engine bullet
    python scripts/smoke_added_backends.py --engine mujoco
    srun --gres=gpu:1 python scripts/smoke_added_backends.py --engine mujoco-warp
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final

from usim import (
    AgentSpec,
    ControlSpec,
    ControlTargets,
    ExecutionSpec,
    ImportRequirements,
    JointCommand,
    ModelError,
    ModelSource,
    SceneSpec,
    SimulatorConfig,
    create_backend,
    create_simulator,
)

URDF: Final = """<robot name="native_smoke">
  <link name="base">
    <inertial><mass value="1"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision>
  </link>
  <link name="arm">
    <inertial><mass value="1"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision>
  </link>
  <joint name="hinge" type="revolute">
    <parent link="base"/><child link="arm"/><origin xyz="0 0 0.2"/>
    <axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="10" velocity="2"/>
  </joint>
</robot>"""


def main() -> None:
    """Exercise real joint motion, deterministic reset and process-owned cleanup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=('bullet', 'mujoco', 'mujoco-warp'), required=True)
    args = parser.parse_args()
    compute = 'gpu' if args.engine == 'mujoco-warp' else 'cpu'
    with TemporaryDirectory(prefix='usim-added-native-') as temporary:
        model = Path(temporary) / 'robot.urdf'
        model.write_text(URDF, encoding='utf-8')
        scene = SceneSpec(
            agents=(
                AgentSpec(
                    'robot',
                    ModelSource.from_path(model),
                    ControlSpec(
                        joint_names=('hinge',),
                        position_joints=('hinge',),
                        initial_qpos=(0.1,),
                    ),
                    requirements=ImportRequirements(base='fixed'),
                ),
            )
        )
        with create_backend(args.engine).open(
            scene, ExecutionSpec(backend=compute, sim_rate=120)
        ) as session:
            session.execution.reset(seed=7)
            initial = float(session.state.for_entity('robot').get_qpos().reshape(-1)[0])
            session.control.for_entity('robot').apply(ControlTargets(positions={'hinge': 0.3}))
            for _ in range(60):
                session.execution.step()
            moved = float(session.state.for_entity('robot').get_qpos().reshape(-1)[0])
            if not math.isfinite(moved) or moved <= initial + 0.05:
                raise ModelError(
                    f'Grouped joint did not approach its target: {initial} -> {moved}'
                )
            session.execution.reset(seed=7)
            reset = float(session.state.for_entity('robot').get_qpos().reshape(-1)[0])
            if not math.isclose(reset, initial, abs_tol=1e-6):
                raise ModelError(f'Reset did not restore joint state: {initial} -> {reset}')

        simulator = create_simulator(
            args.engine,
            SimulatorConfig(
                env_id='JointControl',
                robot_path=str(model),
                joint_names=('hinge',),
                initial_qpos=(0.1,),
                backend=compute,
                sim_rate=120,
                render_mode='none',
                obs_mode='state',
                robot_init_qpos_noise=0,
            ),
        )
        try:
            simulator.reset(seed=7)
            for _ in range(60):
                simulator.step(JointCommand(positions={'hinge': 0.3}))
            episode = float(simulator.get_qpos().reshape(-1)[0])
            if not math.isfinite(episode) or episode <= 0.15:
                raise ModelError(f'Episode joint did not approach its target: {episode}')
        finally:
            simulator.close()

        command = subprocess.run(
            [
                sys.executable,
                '-m',
                'usim.cli',
                'run',
                '--engine',
                args.engine,
                '--compute',
                compute,
                '--robot',
                str(model),
                '--joint-names',
                'hinge',
                '--steps',
                '20',
                '--action',
                '0.3',
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if command.returncode:
            raise ModelError(f'Native CLI failed:\n{command.stdout}\n{command.stderr}')
        print(
            json.dumps(
                {
                    'engine': args.engine,
                    'initial': initial,
                    'grouped': moved,
                    'reset': reset,
                    'episode': episode,
                    'cli_exit': command.returncode,
                    'cli_output': command.stdout.strip(),
                }
            )
        )


if __name__ == '__main__':
    main()
