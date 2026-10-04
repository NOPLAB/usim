"""Run a real reset/step/close smoke against the installed native backend."""

import argparse
import json

import numpy as np

from usim import SimulatorConfig, create_simulator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-id', default='PickCube-v1')
    parser.add_argument('--robot-uid', default='panda')
    parser.add_argument('--backend', default='cpu')
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--obs-mode', default='state')
    parser.add_argument('--camera-uid', default='base_camera')
    parser.add_argument('--render-mode', default='none')
    args = parser.parse_args()
    config = SimulatorConfig(
        env_id=args.env_id,
        robot_uid=args.robot_uid,
        backend=args.backend,
        n_envs=args.num_envs,
        obs_mode=args.obs_mode,
        camera_uid=args.camera_uid,
        render_mode=args.render_mode,
        max_episode_steps=2,
        robot_init_qpos_noise=0,
    )
    sim = create_simulator('maniskill', config)
    try:
        obs, _ = sim.reset(seed=42)
        expected_shape = (len(sim.joint_names),)
        if args.num_envs > 1:
            expected_shape = (args.num_envs, *expected_shape)
        assert obs.qpos.shape == expected_shape
        assert obs.qvel.shape == expected_shape
        initial = obs.qpos if args.num_envs == 1 else obs.qpos[0]
        config.initial_qpos = tuple(float(value) for value in initial)
        obs, _ = sim.reset(seed=42)
        np.testing.assert_allclose(obs.qpos, np.broadcast_to(initial, expected_shape), atol=1e-6)
        if 'rgb' in args.obs_mode:
            assert obs.rgb_image.dtype == np.uint8
            assert obs.rgb_image.ndim == (3 if args.num_envs == 1 else 4)
            assert np.ptp(obs.rgb_image) > 0
        if 'depth' in args.obs_mode:
            assert obs.depth_image is not None
        action = np.zeros(sim._env.action_space.shape, dtype=np.float32)
        first = sim.step(action)
        assert not np.any(first.truncated)
        result = sim.step(action)
        assert np.all(result.truncated)
        assert np.isfinite(result.reward).all()
        if args.num_envs > 1:
            assert result.reward.shape == (args.num_envs,)
            assert result.terminated.shape == (args.num_envs,)
            assert result.truncated.shape == (args.num_envs,)
        print(
            json.dumps(
                {
                    'env_id': args.env_id,
                    'robot_uid': args.robot_uid,
                    'num_envs': args.num_envs,
                    'joints': sim.joint_names,
                    'qpos_shape': obs.qpos.shape,
                    'rgb_shape': None if obs.rgb_image is None else obs.rgb_image.shape,
                    'depth_shape': None if obs.depth_image is None else obs.depth_image.shape,
                    'reward': np.asarray(result.reward).tolist(),
                    'truncated': np.asarray(result.truncated).tolist(),
                }
            )
        )
    finally:
        sim.close()
    sim.close()
    assert not sim.is_running
    print('NATIVE_SMOKE_PASS')


if __name__ == '__main__':
    main()
