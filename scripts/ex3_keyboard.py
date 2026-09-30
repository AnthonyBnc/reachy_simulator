"""Example 3: control the robot live with the keyboard.

    mjpython scripts/ex3_keyboard.py

Click the viewer window first, then:
    Up / Down      right shoulder pitch
    E / D          right elbow pitch
    Left / Right   neck yaw
Each key press changes the target by 5 degrees, clipped to the motor's range.
This is what an RL policy does every step: a small, clipped change of the targets.
"""

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

SCENE = str(Path(__file__).resolve().parents[1] / "assets" / "reach_scene.xml")
model = mujoco.MjModel.from_xml_path(SCENE)
data = mujoco.MjData(model)


def act(name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


STEP = np.radians(5)
FPS = 60
STEPS_PER_FRAME = round(1 / (FPS * model.opt.timestep))  # physics steps between redraws (~8)
# GLFW key code -> (actuator, direction)
KEYS = {
    265: (act("r_shoulder_pitch"), -1),  # Up: raise arm forward
    264: (act("r_shoulder_pitch"), +1),  # Down: lower arm
    ord("E"): (act("r_elbow_pitch"), -1),  # bend elbow
    ord("D"): (act("r_elbow_pitch"), +1),  # straighten elbow
    263: (act("neck_yaw"), +1),  # Left: look left
    262: (act("neck_yaw"), -1),  # Right: look right
}


def on_key(key):
    if key not in KEYS:
        return
    i, direction = KEYS[key]
    lo, hi = model.actuator_ctrlrange[i]
    data.ctrl[i] = np.clip(data.ctrl[i] + direction * STEP, lo, hi)  # a mini safety shield
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"{name:<18} target = {np.degrees(data.ctrl[i]):7.1f} deg")


if __name__ == "__main__":
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        wall_start = time.perf_counter()
        while viewer.is_running():
            for _ in range(STEPS_PER_FRAME):
                mujoco.mj_step(model, data)
            viewer.sync()  # redraw ~60 times per second
            ahead = data.time - (time.perf_counter() - wall_start)  # keep real-time speed
            if ahead > 0:
                time.sleep(ahead)
