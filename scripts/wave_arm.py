"""Drive the Reachy 2 right arm, gripper and head with simple sine waves.

    mjpython scripts/wave_arm.py
"""

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

SCENE = str(Path(__file__).resolve().parents[1] / "assets" / "reach_scene.xml")

model = mujoco.MjModel.from_xml_path(SCENE)
data = mujoco.MjData(model)


def actuator(name):
    """Look up a motor by name (never hard-code indexes: the left arm comes first)."""
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


shoulder_pitch = actuator("r_shoulder_pitch")
elbow_pitch = actuator("r_elbow_pitch")
wrist_yaw = actuator("r_wrist_yaw")
gripper = actuator("r_hand_finger")
neck_yaw = actuator("neck_yaw")

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    while viewer.is_running():
        t = time.time() - start

        # Motor targets in RADIANS, kept inside each joint's range
        data.ctrl[shoulder_pitch] = -0.8 + 0.3 * np.sin(t)  # lift arm forward, swing a bit
        data.ctrl[elbow_pitch] = -1.2                        # bend the elbow
        data.ctrl[wrist_yaw] = 0.6 * np.sin(2 * t)           # wave the hand
        data.ctrl[gripper] = 1.0 + 0.8 * np.sin(t)           # open / close the gripper
        data.ctrl[neck_yaw] = 0.3 * np.sin(0.5 * t)          # look left / right

        mujoco.mj_step(model, data)     # advance physics by one timestep (2 ms)
        viewer.sync()                   # redraw the window
        time.sleep(model.opt.timestep)  # roughly real-time
