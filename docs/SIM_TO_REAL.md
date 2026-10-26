# Sim-to-Real Notes

Records of how the MuJoCo model compares with **our** Reachy 2. Fill in during Step 1.2 of [SIMULATION_PHASE.md](SIMULATION_PHASE.md) and extend during system identification (P4).

---

## 1. Model vs. real kinematics (Step 1.2)

**Robot:** `10.116.19.109` · **SDK:** reachy2-sdk 1.0.7 · **Model:** `assets/reachy2` (Pollen commit `f6d8284`) · **Date:** `2026-10-15`

### 1.1 Forward kinematics comparison (no arm motion)

```bash
python scripts/compare_fk.py --host <robot-ip>
```

The robot computes `r_arm.forward_kinematics(joints)` and MuJoCo computes `r_arm_tip` in the `torso` frame for the same joints.

| Pose (deg, SDK order) | MuJoCo xyz (m) | Real xyz (m) | Pos err (mm) | Rot diff (deg) |
|---|---|---|---|---|
| zero `[0,0,0,0,0,0,0]` | 0.030, −0.368, −0.63 | 0.030, −0.368, −0.638 | 0.0 | 0.0 |
| elbow_90 `[0,0,0,−90,0,0,0]` | 0.387, −0.205, −0.27 | 0.387, −0.205, −0.270 | 0.0 | 0.0 |
| shoulder_fwd `[−60,−15,0,−90,0,0,0]` | 0.421, −0.146, 0.20 | 0.421, −0.146, 0.206 | 0.0 | 0.0 |
| shoulder_side `[0,−45,0,−45,0,0,0]` | 0.347, −0.621, −0.27 | 0.347, −0.621, −0.274 | 0.0 | 0.0 |
| elbow_yaw `[−30,−20,45,−70,0,0,0]` | 0.444, −0.101, −0.29 | 0.444, −0.101, −0.298 | 0.0 | 0.0 |
| wrist `[−40,−15,0,−80,15,−15,45]` | 0.490, −0.172, 0.01 | 0.490, −0.172, 0.013 | 0.0 | 0.0 |
| current (real) | −0.032, −0.071, −0.646 | −0.032, −0.071, −0.646 | 0.0 | 0.0 |

**Pass if** the position error is < 10 mm for every pose.

**Result (2026-10-15): PASS**, 0.0 mm / 0.0° on every pose. The SDK's `forward_kinematics` is computed from the robot's own model, so this confirms the MuJoCo model and the SDK use the same kinematics and the same hand frame. It does not check motor calibration; the dry run of `scripts/run_real.py` covers reading the real joint angles.

How to read the result:
- **The same rotation difference on every pose:** the SDK's end-effector frame is defined differently from `r_arm_tip`. Add a fixed offset in the backend.
- **An error that grows with one joint:** that joint's sign, zero offset or link length differs. Note which joint in §1.3.

### 1.2 Joint directions (viewer vs. real robot)

Move one joint at a time by a small positive amount (+10°): in the viewer with `mjpython scripts/ex3_keyboard.py` or the Control sliders, and on the robot with a small relative `goto`.

| # | Joint | + direction in sim | + direction on robot | Same? |
|---|---|---|---|---|
| 0 | shoulder pitch | | | |
| 1 | shoulder roll | | | |
| 2 | elbow yaw | | | |
| 3 | elbow pitch | | | |
| 4 | wrist roll | | | |
| 5 | wrist pitch | | | |
| 6 | wrist yaw | | | |
| – | gripper (`r_hand_finger` rad ↔ SDK opening 0–100) | | | |

### 1.3 Differences found and how they are handled

| Difference | Handled where |
|---|---|
| _none yet_ | |

---

## 2. Joint limits

| Joint | Model range (deg) | Real robot limit (deg) | Shield soft limit (deg) |
|---|---|---|---|
| r_shoulder_pitch | −180 … 180 | | |
| r_shoulder_roll | −180 … 0 | | |
| r_elbow_yaw | −180 … 180 | | |
| r_elbow_pitch | −128.9 … 5.7 | | |
| r_wrist_roll | −20.1 … 20.1 | | |
| r_wrist_pitch | −20.1 … 20.1 | | |
| r_wrist_yaw | −90 … 90 | | |

---

## 3. Actuator tracking (P4, system identification)

_To be filled: step responses sim vs. real, fitted `kp` / `kv` / damping / latency._

Baseline in sim (`scripts/ex1_one_joint.py`, `r_elbow_pitch` 0 → −90°, `kp=500 kv=10`): overshoot to −90.25° at 0.6 s, settles at −89.55° (0.45° gravity error).
