"""Move the green ball; the trained policy makes the right hand follow it.

    mjpython scripts/follow_ball.py --model runs/any/model.zip      (macOS)
    python scripts\\follow_ball.py --model runs\\any\\model.zip        (Windows)

Move the ball:
    Mouse:    double-click the green ball, then hold Ctrl and drag with the RIGHT mouse button
    Keyboard: arrow keys = forward / back / left / right, PageUp / PageDown = up / down

Needs a model trained on random goals with --no-wrist.
"""

import argparse
import time

import gymnasium as gym
import mujoco.viewer
import numpy as np
from stable_baselines3 import SAC

import reachy_sim  # noqa: F401

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
args = parser.parse_args()

env = gym.make("ReachyReach-v0", control_wrist=False).unwrapped
model = SAC.load(args.model)
backend = env.backend
obs, _ = env.reset()

STEP = 0.02  # 2 cm per key press
KEYS = {  # GLFW key code -> change of the goal in the torso frame (x forward, y left, z up)
    265: [STEP, 0, 0], 264: [-STEP, 0, 0],    # Up / Down arrow: forward / back
    263: [0, STEP, 0], 262: [0, -STEP, 0],    # Left / Right arrow: left / right
    266: [0, 0, STEP], 267: [0, 0, -STEP],    # PageUp / PageDown: up / down
}
nudge = np.zeros(3)


def on_key(key):
    if key in KEYS:
        nudge[:] += KEYS[key]


def ball_in_torso_frame():
    """Where the green ball is now (it may have been dragged with the mouse)."""
    R = backend.data.xmat[backend.torso].reshape(3, 3)
    return R.T @ (backend.data.mocap_pos[backend.goal_mocap] - backend.data.xpos[backend.torso])


with mujoco.viewer.launch_passive(backend.model, backend.data, key_callback=on_key) as viewer:
    last_print = 0.0
    while viewer.is_running():
        start = time.perf_counter()

        # 1. Read where the ball is, add any key presses, keep it inside the safe box
        goal = env.set_goal(ball_in_torso_frame() + nudge)
        nudge[:] = 0

        # 2. The observation must contain the NEW goal
        hand = obs["achieved_goal"].astype(float)
        obs["desired_goal"] = goal.astype(np.float32)
        obs["observation"][17:20] = (goal - hand).astype(np.float32)  # the "goal - hand" part

        # 3. Policy -> action -> one step. The time limit is ignored, so there is no reset.
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _, info = env.step(action)

        if start - last_print > 1.0:  # once per second
            print(f"goal {goal.round(2)}  distance {100 * info['distance']:.1f} cm")
            last_print = start

        viewer.sync()
        time.sleep(max(0.0, backend.dt - (time.perf_counter() - start)))  # real-time speed