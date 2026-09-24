"""Example 2: a smooth joint-space goto(), like reachy.r_arm.goto(joints, duration).

    mjpython scripts/ex2_goto.py

Also prints the hand position in the TORSO frame, the same frame as the 4x4 poses
recorded with the real robot (reachy2-robot-controller/demo_Jul2026/robot/poses.py).
"""

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

SCENE = str(Path(__file__).resolve().parents[1] / "assets" / "reach_scene.xml")
model = mujoco.MjModel.from_xml_path(SCENE)
data = mujoco.MjData(model)

# Same joint order as reachy2-sdk
R_ARM = ["r_shoulder_pitch", "r_shoulder_roll", "r_elbow_yaw", "r_elbow_pitch",
         "r_wrist_roll", "r_wrist_pitch", "r_wrist_yaw"]
ACT = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in R_ARM]
QADR = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in R_ARM]
TORSO = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
HAND = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_arm_tip")


def hand_in_torso():
    """Hand position in the torso frame."""
    R = data.xmat[TORSO].reshape(3, 3)
    return R.T @ (data.xpos[HAND] - data.xpos[TORSO])


def goto(target_deg, duration=2.0, viewer=None):
    """Smooth move to `target_deg` (7 joints, DEGREES) in `duration` seconds."""
    start = data.qpos[QADR].copy()
    goal = np.radians(target_deg)
    n_steps = int(duration / model.opt.timestep)
    for i in range(n_steps):
        s = (i + 1) / n_steps
        s = 3 * s**2 - 2 * s**3                      # smooth start and stop
        data.ctrl[ACT] = start + s * (goal - start)  # moving target
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
            time.sleep(model.opt.timestep)
    reached = np.degrees(data.qpos[QADR])
    print(f"target {np.round(target_deg, 1)}\n  reached {np.round(reached, 1)}"
          f"\n  hand (torso frame) = {np.round(hand_in_torso(), 3)} m")


HOME = [0, 0, 0, 0, 0, 0, 0]
RAISE = [-60, -15, 0, -90, 0, 0, 0]
WAVE_A = [-60, -15, 0, -90, 0, 0, 40]
WAVE_B = [-60, -15, 0, -90, 0, 0, -40]

if __name__ == "__main__":
    with mujoco.viewer.launch_passive(model, data) as viewer:
        goto(RAISE, viewer=viewer)
        for _ in range(3):
            goto(WAVE_A, duration=0.8, viewer=viewer)
            goto(WAVE_B, duration=0.8, viewer=viewer)
        goto(HOME, viewer=viewer)

        while viewer.is_running():  # keep the window open
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)
