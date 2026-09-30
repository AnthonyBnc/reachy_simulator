"""Example 1: control ONE joint and watch it move (no window).

    python scripts/ex1_one_joint.py

Shows the basic loop: write a target into data.ctrl -> mj_step -> read data.qpos.
The joint does not jump to the target: the position motor pulls it there over time,
and gravity can leave a small steady-state error.
"""

from pathlib import Path

import mujoco
import numpy as np

SCENE = str(Path(__file__).resolve().parents[1] / "assets" / "reach_scene.xml")

model = mujoco.MjModel.from_xml_path(SCENE)
data = mujoco.MjData(model)

# Find things by NAME (motor indexes differ from joint indexes)
act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "r_elbow_pitch")
jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "r_elbow_pitch")
qadr = model.jnt_qposadr[jnt]  # where this joint's angle lives in data.qpos

target_deg = -90.0
data.ctrl[act] = np.radians(target_deg)

for step in range(1001):  # 1001 steps x 2 ms = 2 s
    mujoco.mj_step(model, data)
    if step % 100 == 0:
        angle = np.degrees(data.qpos[qadr])
        print(f"t={step * model.opt.timestep:.1f}s  elbow={angle:7.2f} deg  (target {target_deg})")
