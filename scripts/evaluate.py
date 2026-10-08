"""Evaluate a trained policy headless"""

import argparse
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

import reachy_sim

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--goal", type=float, nargs=3, metavar=("X", "Y", "Z"))
parser.add_argument("--episodes", type=int, default=100)
args = parser.parse_args()

env = gym.make("ReachyReach-v0", fixed_goal=args.goal)
model = SAC.load(args.model)
reached, held, final_cm, time_to_reach, interventions = [], [], [], [], []
for ep in range(args.episodes):
    obs, _ = env.reset(seed=10_000 + ep)  # goals and start poses never seen in training
    inside, n, done = [], 0, False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        inside.append(info["is_success"])
        n += info["shield_interventions"]
        done = terminated or truncated
    reached.append(any(inside))
    held.append(all(inside[-20:]))                 # still on the goal for the last 1 s
    final_cm.append(100 * info["distance"])
    if any(inside):
        time_to_reach.append(0.05 * (inside.index(True) + 1))
    interventions.append(n)

print(f"episodes:                {args.episodes}")
print(f"reached the goal:        {100 * np.mean(reached):.0f}%")
print(f"reached AND held (1 s):  {100 * np.mean(held):.0f}%   <- success")
print(f"final distance:          mean {np.mean(final_cm):.1f} cm, worst {np.max(final_cm):.1f} cm")
if time_to_reach:
    print(f"time to reach:           {np.mean(time_to_reach):.1f} s")
print(f"shield interventions:    {np.mean(interventions):.1f} per episode")