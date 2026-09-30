# Reachy Simulator — Project Baseline

> Status: **Draft v0.2** (2026-09-24). P0 in progress. Step-by-step plan for the current phase: [SIMULATION_PHASE.md](SIMULATION_PHASE.md)
> Owner: BaoAn
> Parent project: [`reachy2-robot-controller`](../../reachy2-robot-controller) 

This document defines the scope, architecture, conventions and roadmap for **Reachy Simulator**: a MuJoCo + Gymnasium environment for Pollen Robotics' Reachy 2, used to train reinforcement learning (RL) policies in simulation and transfer them safely to the real robot.

---

## 1. Goals

### 1.1 Primary goal

Build a reproducible **sim-to-real RL pipeline** for Reachy 2:

```txt
MuJoCo model of Reachy 2
  → Gymnasium environment (tasks, rewards, safety limits)
  → Baseline RL algorithms (PPO / SAC / TD3 / HER)
  → Adaptation techniques (domain randomization, residual RL, fine-tuning)
  → Deployment on the real robot through reachy2-sdk
```

### 1.2 Success criteria

| # | Criterion | Measured by |
|---|---|---|
| G1 | Sim env passes Gymnasium's `check_env` and runs ≥ 1,000 steps/s headless (single env, CPU) | `tests/`, benchmark script |
| G2 | A baseline policy solves the **Reach** task in sim with ≥ 95% success | Eval over 100 seeded episodes |
| G3 | The same policy runs on the real robot through the same interface with ≥ 80% success | Real-robot eval protocol (§9.4) |
| G4 | No safety violation on the real robot (joint limit, speed limit, workspace box) | Safety shield logs |
| G5 | Every experiment is reproducible from a config file + seed | Hydra/YAML config + run logs |

### 1.3 Non-goals (for now)

- Mobile base control (disabled on our robot: `mobile_base: null`, VESC and lidar missing).
- Lidar-based perception.
- Voice / LLM interaction (already covered by the parent project).
- Full contact-rich dexterous manipulation. Gripper tasks are limited to open/close grasping.

---

## 2. What we inherit from `reachy2-robot-controller`

The parent project already gives us verified facts about **our** robot. The simulator must follow them so policies transfer cleanly.

### 2.1 Software versions (confirmed on the real robot)

```txt
reachy2-sdk==1.0.7
reachy2-sdk-api==1.0.11
Python 3.10.x             (matches the Python version on the robot itself; avoid 3.13+ because grpcio may fail to build)
```

### 2.2 Available hardware

| Component | Real robot | Used in sim v1? |
|---|---|---|
| Head / neck (`look_at`) | Yes | Yes (head-tracking task) |
| Right arm, 7 DoF | Yes | **Yes (main target)** |
| Left arm, 7 DoF | Yes | Later (bimanual) |
| Grippers (`set_opening` 0–100) | Yes | Yes |
| Antennas | Yes | No (cosmetic) |
| Teleop camera (RGB stereo) | Yes | Later (vision obs) |
| Depth camera | Detected | Later |
| Mobile base | **No** | No |
| Lidar | **No** | No |

### 2.3 Joint convention (per arm)

```txt
0 = shoulder pitch
1 = shoulder roll
2 = elbow yaw
3 = elbow pitch
4 = wrist roll
5 = wrist pitch
6 = wrist yaw
```

The SDK exposes joint positions in **degrees**. MuJoCo uses **radians**. All conversions happen in one place: the robot backend layer (§5.3). Policies never see degrees.

### 2.4 Safety limits already in use

Head (`look_at` target):

```txt
x: 0.4 to 0.7     y: -0.25 to 0.25     z: -0.15 to 0.20     duration: 0.5 to 3.0 s
```

Arm, maximum change relative to the current pose (degrees):

```txt
shoulder pitch 20 | shoulder roll 25 | elbow yaw 20 | elbow pitch 25
wrist roll 30     | wrist pitch 20   | wrist yaw 30
arm move duration: >= 1.0 s
```

Gripper: `0` = fully closed (never near fingers), `25` = ball grasp, `100` = open.

These values become the **default safety shield** config (§5.4) and the **action bounds** of the environment.

### 2.5 Reusable assets

| Asset | Source | Use in this project |
|---|---|---|
| Recorded 4×4 end-effector poses (`BASE_POSE`, `PRE_GRASP_POSE`, `GRASP_POSE`, `LIFT_POSE`, `HANDOVER_POSE`, …) | `demo_Jul2026/robot/poses.py` | Goal sampling, reset states, scripted baseline, residual RL base policy |
| Pose recorder | `demo_Jul2026/record_dual_arm_poses.py` | Collect real-robot poses/trajectories for system identification |
| Grasp workflow (pre-grasp → grasp → lift → handover → release) | `demo_Jul2026/workflows/grasp_workflow.py` | Defines the stages of the Grasp task (§6) and a scripted baseline to beat |
| Pose validation (`validate_pose_matrix`) | `demo_Jul2026/robot/motion.py` | Reuse for goal validation |

---

## 3. Technology stack

| Layer | Choice | Notes |
|---|---|---|
| Physics | **MuJoCo 3.14** (`mujoco` Python bindings) | Viewer on macOS: `python -m mujoco.viewer` for the standalone viewer; `mjpython` only for scripts using `launch_passive`. Headless training uses plain `python`. Load scenes with an **absolute** path (relative paths break mesh loading). |
| Robot model | Reachy 2 MJCF from Pollen's [`reachy2_mujoco`](https://github.com/pollen-robotics/reachy2_mujoco) | Robot files only (22 MB) vendored into `assets/reachy2/` from commit `f6d8284` (see `assets/reachy2/SOURCE.md`). Pollen's Python package (fake SDK server) is **not** installed: it pins `mujoco==3.2.6` and targets Linux. |
| Env API | **Gymnasium ≥ 1.0** | `gymnasium.Env` + `GoalEnv`-style dict obs for goal tasks |
| RL baselines | **Stable-Baselines3 ≥ 2.x** (+ `sb3-contrib`) | PPO, SAC, TD3, HER, (TQC, RecurrentPPO from contrib) |
| Vectorization | `gymnasium.vector` / SB3 `SubprocVecEnv` | Optional later: MJX (JAX) for GPU-parallel training |
| Config | YAML (Hydra optional) | One config per experiment |
| Logging | TensorBoard (default), Weights & Biases (optional) | |
| Real robot | `reachy2-sdk==1.0.7` | Same versions as parent project |
| Testing | `pytest` | |

Pollen also ships an official Docker-based simulation (MuJoCo/Gazebo) that exposes the real SDK API against a virtual robot ([docs](https://docs.pollen-robotics.com/developing-with-reachy-2/simulation/simulation-installation/)). It is too slow for RL training, but it is useful as a **third backend** for testing deployment code end-to-end before touching the real robot (§5.3).

---

## 4. System architecture

```txt
                       ┌──────────────────────────────┐
                       │        configs/*.yaml        │
                       └──────────────┬───────────────┘
                                      │
┌──────────────┐   actions   ┌────────▼─────────┐   commands   ┌────────────────────┐
│   Policy     │────────────►│  Gymnasium Env   │─────────────►│   Safety Shield    │
│ (SB3 / own)  │◄────────────│  task + reward   │              │ clip, rate-limit,  │
└──────────────┘ observations└────────▲─────────┘              │ workspace, e-stop  │
                                      │                        └─────────┬──────────┘
                                      │ state                            │
                             ┌────────┴─────────────────────────────────▼──────────┐
                             │           RobotBackend (common interface)            │
                             ├──────────────────┬──────────────────┬────────────────┤
                             │  MujocoBackend   │ PollenSimBackend │  RealBackend   │
                             │  (training)      │ (Docker, SDK)    │  (reachy2-sdk) │
                             └──────────────────┴──────────────────┴────────────────┘
```

Key design rule: **the environment and the policy never know which backend is running.** Switching from sim to real is a config change (`backend: mujoco | pollen_sim | real`).

---

## 5. Core components

### 5.1 MuJoCo model (`assets/`)

- Import the Reachy 2 MJCF, keep upper body only (torso fixed to world; mobile base removed or frozen).
- Verified facts about the model (2026-09-24): arm joint order = SDK order; actuator order ≠ joint order (left arm first), so always use name lookups; base is a `freejoint` (`mobile_base`) that must be removed; wrist roll/pitch limited to ±20°; end-effector body `r_arm_tip`, reference body `torso`. Details in [SIMULATION_PHASE.md §0.1](SIMULATION_PHASE.md).
- Actuators: position actuators per joint, with `kp`, `damping`, `forcerange` tuned to match the real robot (§8, stage A0).
- Add task scenes on top of the robot: `reach_scene.xml`, `grasp_scene.xml` (table + ball), `head_track_scene.xml` (moving target). Scenes live directly in `assets/` next to `reachy2/`, so include paths stay `reachy2/...` (no `../`).
- Timestep: `0.002 s` physics; control at **10–20 Hz** (`frame_skip` 25–50). The real robot is driven at the same control rate.

### 5.2 Gymnasium environments (`reachy_sim/envs/`)

All envs share a base class `ReachyBaseEnv(gymnasium.Env)` that owns: backend, safety shield, control rate, reset logic, domain randomization hooks, rendering.

**Action space (default)** — normalized, `Box(-1, 1, shape=(7,))` (+1 for gripper tasks)

```txt
a ∈ [-1, 1]^7  →  Δq = a * max_delta_per_step  (radians)
q_target = clip(q_current + Δq, soft_joint_limits)
gripper (optional): a_g ∈ [-1, 1] → opening ∈ [0, 100] (min 25 when near humans)
```

Delta-joint actions are chosen because they (a) map directly to the parent project's "relative motion only" safety rule and (b) transfer better than torques, since the real SDK is position-controlled.

**Observation space (default, state-based)**

```txt
q            joint positions (7)            rad
q_dot        joint velocities (7)           rad/s (finite difference on real robot)
ee_pos       end-effector position (3)      m, torso frame
ee_quat      end-effector orientation (4)
gripper      opening (1)                    normalized
goal         task goal (3 or 7)             m / quat
prev_action  last action (7[+1])
```

Goal tasks use a dict observation: `{"observation", "achieved_goal", "desired_goal"}` so that HER works out of the box.

Vision observations (teleop camera RGB) come in a later phase (§10).

### 5.3 Robot backends (`reachy_sim/backends/`)

```python
class RobotBackend(Protocol):
    def reset(self, q_init: np.ndarray | None = None) -> None: ...
    def get_joint_positions(self) -> np.ndarray: ...          # rad, shape (7,)
    def get_ee_pose(self) -> np.ndarray: ...                  # 4x4, torso frame
    def send_joint_targets(self, q: np.ndarray) -> None: ...  # rad
    def set_gripper(self, opening: float) -> None: ...        # 0..100
    def look_at(self, x: float, y: float, z: float) -> None: ...
    def step(self, dt: float) -> None: ...                    # sim: advance physics; real: sleep to keep rate
    def close(self) -> None: ...
```

| Backend | Implementation notes |
|---|---|
| `MujocoBackend` | Direct `mujoco.MjData` access. Deterministic with seed. Supports domain randomization. |
| `PollenSimBackend` | Uses `reachy2-sdk` against Pollen's Docker simulation. Same code path as real → smoke test for deployment. |
| `RealBackend` | Uses `reachy2-sdk==1.0.7`. Converts rad ↔ deg. Streams joint goals at the control rate (use per-joint `goal_position` + `send_goal_positions()`; verify exact API on SDK 1.0.7). Uses `goto(..., duration>=1.0)` only for reset/home moves. Reads current pose before every episode, saves a home pose, returns home at the end (parent project rule). |

### 5.4 Safety shield (`reachy_sim/safety/`)

Runs in **every** backend (so sim training learns under the same constraints), but is mandatory and strict on `RealBackend`:

1. Clip joint targets to soft joint limits (hard limits minus a margin).
2. Rate-limit: `|Δq| ≤ max_delta_per_step` per joint (derived from §2.4).
3. Cartesian workspace box for the end-effector (no self-collision zone, no table penetration, no zone in front of the operator).
4. Head targets clamped to the `look_at` limits in §2.4.
5. Gripper floor (e.g. never below 25) unless the task explicitly needs a full close.
6. Watchdog: if the policy stops sending actions for > N ms, or tracking error exceeds a threshold → hold position and end the episode.
7. Every intervention is logged (`shield_interventions` metric). A high intervention rate means the policy is not ready for real deployment.

---

## 6. Task roadmap

| ID | Task | Description | Obs | Algorithms | Done when |
|---|---|---|---|---|---|
| T0 | **Joint tracking** | Follow random joint targets. Used to validate actuator parameters and sim-vs-real gap. | state | scripted / PPO | sim-real tracking error below threshold |
| T1 | **Reach** | Move right-arm end-effector to a random 3D goal inside a safe workspace box. | state + goal | PPO, SAC, SAC+HER | ≥ 95% sim, ≥ 80% real (err < 3 cm) |
| T2 | **Reach with orientation** | Reach a full 6D pose. Goals sampled around recorded poses (`PRE_GRASP_POSE`, `HANDOVER_POSE`, …). | state + goal | SAC+HER, TQC | ≥ 90% sim (3 cm, 15°) |
| T3 | **Head tracking** | Keep a moving target centered using `look_at`. Mirrors the face-tracking demo. | target pixel/3D pos | PPO | smooth tracking, no limit hits |
| T4 | **Grasp ball** | Pre-grasp → grasp → lift, following the parent project's grasp workflow. Sparse + staged reward. | state (+ ball pos) | SAC+HER, residual RL | ≥ 80% sim, ≥ 60% real |
| T5 | **Handover** | Grasp → move to `HANDOVER_POSE` → release. | state | residual RL on scripted workflow | stretch goal |
| T6 | **Vision-based reach/grasp** | Replace ball position with camera image. | RGB (+depth) | PPO/SAC + CNN encoder | stretch goal |

### 6.1 Reward design principles

- Reach: dense `-‖ee - goal‖` plus success bonus; penalty on action magnitude and action change (smoothness transfers better).
- Goal tasks: also provide a **sparse** version (`-1` until success) for HER.
- Always add a penalty for safety-shield interventions so the policy learns to stay inside safe limits instead of relying on the clip.
- Keep reward terms individually logged (`info["reward_terms"]`).

---

## 7. Baseline algorithms

| Algorithm | Why | First task |
|---|---|---|
| **PPO** | Stable, easy to tune, works with many parallel envs | T1, T3 |
| **SAC** | Sample-efficient, good for continuous control and real-robot fine-tuning | T1, T2, T4 |
| **TD3** | Deterministic comparison point for SAC | T1 |
| **SAC + HER** | Sparse goal-conditioned tasks | T2, T4 |
| **TQC** (sb3-contrib) | Often stronger than SAC on manipulation | T2, T4 |
| **Scripted baseline** | Existing grasp workflow from the parent project; the RL policy must beat it | T4, T5 |

Each algorithm gets a default hyperparameter file under `configs/algo/`. All comparisons are reported over **≥ 3 seeds**.

---

## 8. Adaptation and sim-to-real methods

Ordered from simplest to most advanced. Each stage is only added when the previous one is not enough.

| Stage | Method | What it does |
|---|---|---|
| A0 | **System identification** | Record real joint trajectories (T0) and fit MuJoCo `kp`, damping, friction, armature, latency to minimize tracking error. |
| A1 | **Domain randomization (DR)** | Randomize per episode: actuator gains, damping, friction, link masses (±10–20%), control latency (0–2 steps), observation noise, goal/object positions. |
| A2 | **Action/observation filtering** | Low-pass filter on actions, action-rate penalty, frame stacking or history for latency. |
| A3 | **Residual RL** | Policy outputs a correction on top of the scripted pose-based controller (`a = a_scripted + a_residual`). Safest route for T4/T5. |
| A4 | **Adaptive policy (RMA-style)** | Train with privileged sim parameters, then learn an adaptation module that estimates them from recent history on the real robot. |
| A5 | **Real-world fine-tuning** | Short SAC fine-tuning on the real robot with strict shield and small action limits, starting from the sim policy + replay buffer. |

---

## 9. Sim-to-real workflow

### 9.1 Stages

```txt
1. Train in MujocoBackend (with DR)
2. Evaluate in MujocoBackend with held-out randomization
3. Run the same policy in PollenSimBackend (Docker) → checks SDK path, units, frames, rate
4. Real robot, "dry run": policy runs, actions are logged but NOT sent
5. Real robot, reduced limits (50% of §2.4 deltas, slower rate), operator at E-stop
6. Real robot, full evaluation protocol (§9.4)
```

### 9.2 Frames and units checklist

- [ ] Torso frame used everywhere (the recorded 4×4 poses are in the SDK's torso frame; confirm MJCF frame matches).
- [ ] Radians in the env, degrees only inside `RealBackend`.
- [ ] Gripper opening 0–100 mapped identically in sim and real.
- [ ] Joint order identical (§2.3).
- [ ] Control rate identical; real backend measures and logs actual loop time.

### 9.3 Real-robot safety protocol

Extends the parent project's startup routine:

1. E-stop within reach; arms have clear space; nobody inside the workspace box.
2. `mobile_base: null` stays in the robot YAML.
3. Connection test (`check_reachy_access.py` from parent project).
4. Read and save the home pose before every run.
5. Start with head-only / T0 tracking before any learned arm policy.
6. Shield in strict mode; watchdog enabled.
7. Always return to home pose at the end or on any exception (`try/finally`).
8. Only one arm moves at a time until bimanual is explicitly tested.

### 9.4 Evaluation protocol

- Fixed list of goals (same seeds) for sim and real.
- Metrics per episode: success, final error (cm / deg), time to success, path length, jerk, shield interventions, max joint velocity.
- Real-robot results: ≥ 20 episodes per policy, video + logs saved under `runs/<exp>/real_eval/`.
- Report the **sim-to-real gap** = sim success − real success, per task and per adaptation stage.

---

## 10. Proposed repository structure

```txt
reachy_simulator/
├── README.md
├── pyproject.toml               # package: reachy_sim
├── requirements.txt
├── docs/
│   ├── PROJECT_BASELINE.md      # this file
│   ├── SIMULATION_PHASE.md      # step-by-step plan for P0–P2 (+ sim-side DR)
│   ├── SIM_TO_REAL.md           # detailed notes, sysid results
│   └── EXPERIMENTS.md           # experiment log / results table
├── assets/
│   ├── reachy2/                 # vendored MJCF + meshes (pinned commit, SOURCE.md, LICENSE)
│   ├── test_scene.xml           # Pollen's test scene (robot + table + bottle)
│   └── reach_scene.xml, ...     # task scenes (fixed base), next to reachy2/
├── configs/
│   ├── env/                     # reach.yaml, grasp.yaml, ...
│   ├── algo/                    # ppo.yaml, sac.yaml, sac_her.yaml, ...
│   ├── safety/                  # default.yaml (values from §2.4), strict.yaml
│   └── robot/                   # real.yaml (host IP, rate), pollen_sim.yaml
├── reachy_sim/
│   ├── envs/                    # ReachyBaseEnv, ReachEnv, GraspEnv, HeadTrackEnv
│   ├── backends/                # mujoco_backend.py, pollen_sim_backend.py, real_backend.py
│   ├── safety/                  # shield.py, watchdog.py
│   ├── randomization/           # domain randomization
│   ├── policies/                # scripted baselines, residual wrappers
│   └── utils/                   # frames, units, pose helpers (ported from parent project)
├── scripts/
│   ├── ex1_one_joint.py, ex2_goto.py, ex3_keyboard.py, wave_arm.py   # learning scripts
│   ├── benchmark_env.py         # env steps/s
│   ├── train.py                 # python scripts/train.py env=reach algo=sac
│   ├── evaluate.py              # --backend mujoco|pollen_sim|real
│   ├── sysid_record.py          # record real trajectories
│   └── sysid_fit.py
├── tests/
│   ├── test_env_api.py          # gymnasium check_env
│   ├── test_safety_shield.py
│   └── test_units_frames.py
└── runs/                        # gitignored: checkpoints, logs, videos
```

---

## 11. Milestones

| Phase | Deliverable | Exit criterion | Status |
|---|---|---|---|
| **P0 — Setup** | Repo skeleton, pinned deps, Reachy 2 MJCF loads and renders | Robot model vendored and loads ✅; joint order verified against §2.3 ✅; fixed-base scene ✅; control examples ✅; FK check vs. real robot (needs robot) | 🟡 Almost done |
| **P1 — Env core** | `ReachyBaseEnv`, `MujocoBackend`, safety shield, T0 + T1 envs | `check_env` passes; tests green; ≥ 1,000 steps/s | ⬜ |
| **P2 — Baselines** | PPO / SAC / TD3 / SAC+HER on T1, T2 | Results table in `EXPERIMENTS.md` (3 seeds each) | ⬜ |
| **P3 — Real interface** | `RealBackend`, `PollenSimBackend`, dry-run mode | Policy runs end-to-end in Pollen sim; dry run on real logs sane actions | ⬜ |
| **P4 — SysID + DR** | Fitted actuator params, DR config | Sim-real tracking error on T0 reduced; T1 real success ≥ 80% | ⬜ |
| **P5 — Grasp** | T4 env, scripted baseline, residual RL | T4 real success ≥ 60% | ⬜ |
| **P6 — Adaptation** | RMA-style module and/or real fine-tuning | Measurable reduction in sim-to-real gap vs P4 | ⬜ |
| **P7 — Vision (stretch)** | Camera-based observations | T6 working in sim | ⬜ |

---

## 12. Risks and open questions

| Risk / question | Mitigation |
|---|---|
| Official MJCF may not match our robot exactly (versions, gripper, head) | Verify joint limits and kinematics against the real robot using recorded poses (§2.5) in P0 ([SIMULATION_PHASE.md step 1.2](SIMULATION_PHASE.md)) |
| Pollen's fake SDK server (`reachy2_mujoco` package) needs `mujoco==3.2.6` and Linux | Not used. We only reuse the model files; for SDK-level testing use Pollen's Docker simulation (`PollenSimBackend`) |
| SDK 1.0.7 real-time streaming rate and latency unknown | Measure in P3; choose control rate from measurements, not assumptions |
| Real hard joint limits not yet confirmed (parent project uses relative limits) | Read limits from the robot / Pollen config and store in `configs/safety/` |
| Gripper contact physics in MuJoCo differ from reality | Prefer residual RL + scripted grasp for T4; randomize friction |
| Training on macOS (no CUDA) is slow | CPU is fine for state-based SB3; use a GPU machine or MJX only if needed |
| Robot IP changes between sessions (seen in parent project: `10.116.19.109`, `192.168.0.120`) | IP only in `configs/robot/real.yaml`, never hard-coded |
| Safety on the real robot | Shield + watchdog + dry-run stage + operator protocol (§9.3) are mandatory, not optional |

---

## 13. References

- Parent project: `reachy2-robot-controller/README.md`, `Connection_Guide.md`, `demo_Jul2026/`
- Pollen Robotics — Reachy 2 MuJoCo: https://github.com/pollen-robotics/reachy2_mujoco
- Pollen Robotics — Reachy 2 simulation docs: https://docs.pollen-robotics.com/developing-with-reachy-2/simulation/simulation-installation/
- Reachy 2 documentation: https://docs.pollen-robotics.com/
- MuJoCo: https://mujoco.readthedocs.io/
- Gymnasium: https://gymnasium.farama.org/
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/
- Andrychowicz et al., *Hindsight Experience Replay* (2017)
- Tobin et al., *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World* (2017)
- Kumar et al., *RMA: Rapid Motor Adaptation for Legged Robots* (2021)
- Johannink et al., *Residual Reinforcement Learning for Robot Control* (2019)
