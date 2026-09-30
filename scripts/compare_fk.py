"""Step 1.2: compare the MuJoCo model with the real Reachy 2. The arm does NOT move.

For a list of joint configurations, the real robot computes the hand pose with
reachy.r_arm.forward_kinematics(joints) (no motion), and MuJoCo computes the pose of
`r_arm_tip` in the `torso` frame for the same joints. The script prints the difference.

    python scripts/compare_fk.py --host 192.168.0.120   # needs the robot (or Pollen's Docker sim)
    python scripts/compare_fk.py --sim-only             # MuJoCo side only, no robot needed

If every pose shows the SAME rotation offset, the SDK's end-effector frame is simply defined
differently from `r_arm_tip` (a fixed offset we can add). If the error changes from pose to pose,
a joint sign, zero offset or link length differs: write it down in docs/SIM_TO_REAL.md.
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np

SCENE = str(Path(__file__).resolve().parents[1] / "assets" / "reach_scene.xml")

R_ARM = ["r_shoulder_pitch", "r_shoulder_roll", "r_elbow_yaw", "r_elbow_pitch",
         "r_wrist_roll", "r_wrist_pitch", "r_wrist_yaw"]

# Test configurations in DEGREES (SDK order), all inside the model's joint ranges
TEST_POSES = {
    "zero":           [0, 0, 0, 0, 0, 0, 0],
    "elbow_90":       [0, 0, 0, -90, 0, 0, 0],
    "shoulder_fwd":   [-60, -15, 0, -90, 0, 0, 0],
    "shoulder_side":  [0, -45, 0, -45, 0, 0, 0],
    "elbow_yaw":      [-30, -20, 45, -70, 0, 0, 0],
    "wrist":          [-40, -15, 0, -80, 15, -15, 45],
}


class MujocoFK:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(SCENE)
        self.data = mujoco.MjData(self.model)
        self.qadr = [self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                     for n in R_ARM]
        self.torso = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.hand = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "r_arm_tip")

    def pose(self, joints_deg):
        """4x4 pose of r_arm_tip in the torso frame (kinematics only, no physics)."""
        self.data.qpos[self.qadr] = np.radians(joints_deg)
        mujoco.mj_forward(self.model, self.data)
        R_t = self.data.xmat[self.torso].reshape(3, 3)
        R_h = self.data.xmat[self.hand].reshape(3, 3)
        T = np.eye(4)
        T[:3, :3] = R_t.T @ R_h
        T[:3, 3] = R_t.T @ (self.data.xpos[self.hand] - self.data.xpos[self.torso])
        return T


def rotation_angle_deg(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.0.120")
    parser.add_argument("--sim-only", action="store_true")
    args = parser.parse_args()

    sim = MujocoFK()
    poses = dict(TEST_POSES)

    reachy = None
    if not args.sim_only:
        from reachy2_sdk import ReachySDK

        reachy = ReachySDK(host=args.host)
        if not reachy.is_connected():
            raise SystemExit(f"Could not connect to Reachy at {args.host}")
        poses["current (real)"] = reachy.r_arm.get_current_positions()  # degrees

    print(f"{'pose':<16} {'mujoco xyz (m)':<26} {'real xyz (m)':<26} {'pos err':>9} {'rot diff':>9}")
    for name, joints in poses.items():
        T_sim = sim.pose(joints)
        if reachy is None:
            print(f"{name:<16} {np.array2string(T_sim[:3, 3], precision=3):<26}")
            continue
        T_real = np.asarray(reachy.r_arm.forward_kinematics(joints_positions=list(joints), degrees=True))
        pos_err_mm = np.linalg.norm(T_sim[:3, 3] - T_real[:3, 3]) * 1000
        rot_diff = rotation_angle_deg(T_sim[:3, :3].T @ T_real[:3, :3])
        print(f"{name:<16} {np.array2string(T_sim[:3, 3], precision=3):<26} "
              f"{np.array2string(T_real[:3, 3], precision=3):<26} {pos_err_mm:7.1f}mm {rot_diff:7.1f}deg")

    if reachy is not None:
        print("\nPass if pos err < 10 mm for every pose. Record the results in docs/SIM_TO_REAL.md.")


if __name__ == "__main__":
    main()
