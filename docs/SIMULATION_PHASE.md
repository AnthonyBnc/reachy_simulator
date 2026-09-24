# Simulation Phase — Plan and Next Steps

> Status: **in progress** (last updated 2026-09-24)
> Scope: everything that happens **before** the real robot is involved: milestones **P0 → P2**, plus the sim-side part of P4 (domain randomization).
> Big picture and design rationale: [PROJECT_BASELINE.md](PROJECT_BASELINE.md).

**Final goal of the simulation phase:** a policy trained in MuJoCo reaches random 3D goals with the Reachy 2 right arm (≥ 95% success over 100 seeded episodes). It must stay inside the safety limits, move smoothly, and hold up under randomized physics. It should be ready to hand over to the real-robot phase (P3).

---

## 0. Where we are now

| Item | Status |
|---|---|
| Project baseline doc, README | ✅ Done |
| Python 3.10 `.venv` + pinned `requirements.txt` (mujoco 3.14, gymnasium 1.3, SB3 2.9, reachy2-sdk 1.0.7) | ✅ Done |
| Pollen's `reachy2_mujoco` cloned to `third_party/` (reference only, not installed) | ✅ Done |
| Robot model vendored into `assets/reachy2/` + `assets/test_scene.xml` | ✅ Done |
| Learning scripts: `wave_arm.py`, `ex1_one_joint.py`, `ex2_goto.py`, `ex3_keyboard.py` | ✅ Written and tested headless; run them yourself to see them |
| Fixed-base scene `assets/reach_scene.xml` | ✅ Done |
| Model-vs-real check (`scripts/compare_fk.py`, `docs/SIM_TO_REAL.md`) | 🟡 Script ready, needs the robot |
| Python package `reachy_sim/`, backend, env, training | ⬜ Not started |

### 0.1 Model facts (verified 2026-09-24)

| Fact | Consequence |
|---|---|
| Arm joint order in the model = SDK order (shoulder pitch, shoulder roll, elbow yaw, elbow pitch, wrist roll, wrist pitch, wrist yaw) | Same 7-vector in sim and real |
| **Motor order ≠ joint order**: in `data.ctrl`, the left arm comes first (6–12), then the right arm (13–19) | Always look up motors and joints **by name** (`mj_name2id`), never by hard-coded index |
| Robot sits on a `freejoint` named `mobile_base` in `test_scene.xml` | Use `reach_scene.xml` (base welded). The torso still rises ~11 mm in the first 0.5 s as the tripod (torso-height slide) settles; after that it moves < 0.1 mm while the arms swing. Let the sim settle after reset. |
| Wrist roll / pitch limited to **±20°** in the model (parent project allows ±30° relative moves) | Safety shield clips to **absolute** model limits too, not only relative deltas |
| All arm motors have the same placeholder gains `kp=500, kv=10` | Must be tuned against the real robot later (P4, system identification) |
| End-effector body: `r_arm_tip`; reference frame body: `torso` | Express hand position in the torso frame, the same frame as the recorded SDK poses |
| MuJoCo 3.14 fails to find meshes when a scene is loaded with a **relative** path | Always load scenes with an absolute path: `Path(__file__).resolve()...` in code, `$PWD/...` in the terminal |

### 0.2 Which command to use

| What | Command |
|---|---|
| Just look at a scene | `python -m mujoco.viewer --mjcf=$PWD/assets/reach_scene.xml` |
| Your script uses `mujoco.viewer.launch_passive` | `mjpython scripts/<script>.py` |
| Headless (training, tests, printing) | `python scripts/<script>.py` |

---

## Step 1 — Finish P0: understand and prepare the model

### 1.0 Learn the control loop ✅ (scripts ready)
Run the learning scripts in order and make sure you understand each one:

1. `ex1_one_joint.py`: set `data.ctrl`, step, read `data.qpos`. Watch the joint move toward the target but not jump there instantly.
2. `ex2_goto.py`: a smooth joint-space `goto()` like the SDK's, and reading the hand position in the torso frame.
3. `ex3_keyboard.py`: small clipped changes to the targets, which is what an RL policy does.

**Done when:** you can explain `model` vs `data`, `ctrl` vs `qpos`, and `mj_step` vs `mj_forward`.

### 1.1 Fixed-base scene ✅
Create `assets/reach_scene.xml`, copied from `test_scene.xml`:
- Remove `<freejoint name="mobile_base" />`, so the robot is welded to the world, matching `mobile_base: null`.
- Remove the bottle and keep or shrink the table.
- Add a **goal marker**: a small sphere on a `mocap="true"` body with `contype="0" conaffinity="0"`, so it's visible but never collides.

Keep scenes in `assets/`, next to `reachy2/`, so the include paths stay simple (`reachy2/...`, no `../`).

**Done when:** `nq` is 7 lower than `test_scene.xml`, and the robot doesn't move when the arm swings fast.

**Result:** `nq` 75 → 61 (−7 base, −7 bottle). The torso moves < 0.1 mm during an aggressive two-arm swing (vs. a free-sliding base before). The goal marker is the mocap body `goal`: set `data.mocap_pos[0]`.

### 1.2 Check the model against the real robot 🟡 (script ready, needs the robot; can run in parallel with Step 2)
- Move each right-arm joint in the viewer and on the real robot, and record the + direction and zero pose in a table.
- `scripts/compare_fk.py --host <ip>`: for 6 test joint configurations plus the robot's current pose, the robot computes `r_arm.forward_kinematics(joints)` (confirmed in SDK 1.0.7; **the arm does not move**) and MuJoCo computes `r_arm_tip` in the torso frame. Prints the position and rotation difference per pose. `--sim-only` runs the MuJoCo side without a robot.
- Record results in [SIM_TO_REAL.md](SIM_TO_REAL.md). First hint: the model's hand position at `elbow_yaw` (0.444, −0.101, −0.298) is close to the recorded `GRASP_POSE` (0.428, −0.162, −0.283), so the frames look consistent.

**Done when:** the hand position differs by < 1 cm for every pose in `compare_fk.py` (6 test poses + the robot's current pose). Any sign or offset differences are written down in `docs/SIM_TO_REAL.md`.

### 1.3 Commit ✅
```bash
git add assets scripts docs README.md
git commit -m "P0: vendored Reachy 2 model, fixed-base scene, control examples"
```

---

## Step 2 — P1: build the environment

Create the Python package so the code is reusable, not just loose scripts.

### 2.1 Package skeleton ⬜
```txt
pyproject.toml                  # name = reachy_sim, python = 3.10
reachy_sim/
  __init__.py
  robot.py                      # joint names, SDK order, model paths
  backends/mujoco_backend.py
  safety/shield.py
  envs/base_env.py
  envs/reach_env.py
configs/
  safety/default.yaml
  env/reach.yaml
tests/
```
Install it in editable mode with `pip install -e .`, so scripts can `import reachy_sim`.

### 2.2 `MujocoBackend` ⬜
Wraps everything you learned in Step 1.0 behind the `RobotBackend` interface from the baseline doc (§5.3):

| Method | Does |
|---|---|
| `reset(q_init)` | `mj_resetData`, set arm joints to home (+ small noise), `mj_forward` |
| `get_joint_positions()` | the 7 right-arm `qpos` values, looked up by name, in radians |
| `get_ee_pose()` | 4×4 pose of `r_arm_tip` in the `torso` frame |
| `send_joint_targets(q)` | write the 7 values to the right-arm `ctrl` indices (by name) |
| `step()` | `mj_step` × `frame_skip` |

Control rate: **20 Hz** (`timestep 0.002 s × frame_skip 25`).

**Done when:** a unit test sends a target, steps, and checks that the joints get within 2° of it.

### 2.3 Safety shield ⬜
Pure numpy, no MuJoCo inside, so the same code runs on the real robot later:
1. Limit each step's change: `|Δq| ≤ max_delta_per_step`. Start with **5° per step at 20 Hz**. (The parent project's limits in §2.4 are for one slow `goto`, so they're too large per step.)
2. Clip to the model's **absolute** joint ranges minus a margin (e.g. 5°).
3. Keep the end-effector inside a workspace box (torso frame), taken from the recorded poses plus a margin.
4. Count every intervention (`info["shield_interventions"]`).

Values go in `configs/safety/default.yaml`.

**Done when:** `tests/test_safety_shield.py` covers each rule.

### 2.4 `ReachyBaseEnv` + `ReachEnv` (T1) ⬜

| Part | Choice |
|---|---|
| Action | `Box(-1, 1, (7,))` → `Δq = a × max_delta_per_step` → shield → `send_joint_targets` |
| Observation (dict, for HER) | `observation`: q (7), q̇ (7), hand pos (3), prev action (7); `achieved_goal`: hand pos (3); `desired_goal`: goal pos (3) |
| Goal sampling | uniform inside the workspace box; move the mocap marker to the goal |
| Reward (dense) | `-‖hand − goal‖ − 0.01‖a‖² − 0.01‖a − a_prev‖² − 0.1·interventions` |
| Reward (sparse, for HER) | `0` if `‖hand − goal‖ < 3 cm`, else `-1` |
| Episode | 100 steps (5 s at 20 Hz). `terminated` on success is optional; `truncated` at 100 steps. |
| Render | `render_mode="human"` (viewer) and `"rgb_array"` (videos) |

Register it as `ReachyReach-v0` with `gymnasium.register`.

### 2.5 Scripted reference controller ⬜
A simple Jacobian controller (`mj_jac` on `r_arm_tip`, damped least squares), driven through the **same** env and action space. It proves the task can be solved with these limits and gives a baseline success rate for RL to match.

**Done when:** it reaches ≥ 95% of the sampled goals. If not, the workspace box or limits are wrong, so fix those before training RL.

### 2.6 Tests and speed ⬜
- `gymnasium.utils.env_checker.check_env` passes.
- The same seed gives the same episode (determinism).
- Headless speed is ≥ 1,000 env steps/s on one CPU core (`scripts/benchmark_env.py`).

**P1 done when:** everything in 2.1–2.6 is green.

---

## Step 3 — P2: train baselines

### 3.1 Training script ⬜
`scripts/train.py --env reach --algo sac --seed 0`
- Reads `configs/algo/<algo>.yaml` and `configs/env/<env>.yaml`.
- Saves the config, checkpoints and TensorBoard logs to `runs/<env>_<algo>_<seed>/`.
- Uses `SubprocVecEnv` with 4–8 envs for PPO, and 1–4 for SAC.

### 3.2 Evaluation script ⬜
`scripts/evaluate.py --run runs/reach_sac_0 --episodes 100`
- Uses a **fixed list of goals** (same seeds every time), so results can be compared.
- Reports success rate, final error (cm), time to success, jerk, shield interventions and max joint speed.
- `--video` saves a few episodes as MP4 via `rgb_array`.

### 3.3 Baseline runs ⬜
Train each on `ReachyReach-v0` with 3 seeds and write the results to `docs/EXPERIMENTS.md`:

| Algo | Reward | Expect |
|---|---|---|
| SAC | dense | first to try; should work within ~100–300k steps |
| PPO | dense | needs more steps; good sanity check |
| TD3 | dense | comparison to SAC |
| SAC + HER | sparse | checks the goal-conditioned setup, needed later for grasping |

**Done when:** at least one algorithm has ≥ 95% success, compared against the scripted controller from 2.5.

### 3.4 Next tasks (after Reach works) ⬜
- **T2 Reach with orientation:** 6D goals sampled around the recorded poses (`PRE_GRASP_POSE`, `HANDOVER_POSE`, …); SAC+HER / TQC.
- **T3 Head tracking:** a moving target, with the neck joints as the action. The mapping to `look_at` is handled later in `RealBackend`.
- **T4 Grasp ball:** add the gripper action, then use residual RL on top of the scripted grasp workflow from the parent project.

---

## Step 4 — Prepare for the real robot while still in sim ⬜

Do this before leaving simulation, so the policy doesn't overfit to one exact physics setup:

1. **Domain randomization** (`reachy_sim/randomization/`): each episode, randomize `kp`/`kv` (±20%), joint damping/friction, link masses (±10%), control latency (0–2 steps) and observation noise (≈ 0.5°).
2. Retrain the best algorithm with DR on. Evaluate on **held-out** randomization settings.
3. **Stress tests:** larger latency, weaker motors, and noisy joint readings. Record how quickly success drops.

**Simulation phase complete when:**
- [ ] Reach success ≥ 95% (100 fixed goals) **with DR on**
- [ ] Shield interventions < 1% of steps
- [ ] Motion is smooth (no chattering; jerk within the scripted controller's range)
- [ ] Results + configs + checkpoints recorded in `docs/EXPERIMENTS.md`
- [ ] Model-vs-real check from 1.2 done, and any differences handled

Then move on to **P3**: `RealBackend`, a dry run on the robot, and the protocol in PROJECT_BASELINE.md §9.

---

## Immediate next actions (in order)

1. Run `ex1` → `ex2` → `ex3` yourself and make sure each one makes sense (Step 1.0).
2. When you're next with the robot: `python scripts/compare_fk.py --host <ip>` and fill in [SIM_TO_REAL.md](SIM_TO_REAL.md) (Step 1.2).
3. Start the package skeleton and `MujocoBackend` (Steps 2.1–2.2).
