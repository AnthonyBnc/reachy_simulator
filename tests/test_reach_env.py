import time

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

import reachy_sim  # noqa: F401  (registers ReachyReach-v0)


def test_check_env():
    check_env(gym.make("ReachyReach-v0").unwrapped, skip_render_check=True)


def test_same_seed_same_episode():
    def run():
        env = gym.make("ReachyReach-v0")
        obs, _ = env.reset(seed=42)
        env.action_space.seed(42)
        out = [obs["observation"]]
        for _ in range(50):
            obs, r, *_ = env.step(env.action_space.sample())
            out.append(np.append(obs["observation"], r))
        return out
    a, b = run(), run()
    assert all(np.allclose(x, y) for x, y in zip(a, b))


def test_random_actions_stay_inside_joint_limits():
    env = gym.make("ReachyReach-v0").unwrapped
    env.reset(seed=0)
    lo, hi = env.backend.joint_limits.T
    for _ in range(300):
        env.step(env.action_space.sample())
        assert np.all(env.q_cmd >= lo - 1e-9) and np.all(env.q_cmd <= hi + 1e-9)


def test_goals_are_inside_box():
    env = gym.make("ReachyReach-v0").unwrapped
    for s in range(20):
        env.reset(seed=s)
        assert env.shield.distance_outside_box(env.goal) == 0.0


def test_speed():
    env = gym.make("ReachyReach-v0")
    env.reset(seed=0)
    n, t0 = 500, time.perf_counter()
    for _ in range(n):
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            env.reset()
    sps = n / (time.perf_counter() - t0)
    print(f"{sps:.0f} env steps/s")
    assert sps > 300