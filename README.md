# Reachy Simulator

A MuJoCo + Gymnasium setup for **Pollen Robotics' Reachy 2**. You train reinforcement learning policies in simulation, then run them on the real robot through the same interface.

> **Status:** P0 in progress. The environment is set up and the Reachy 2 model loads in MuJoCo. Training and evaluation commands are **planned**.
>
> - Next steps: [docs/SIMULATION_PHASE.md](docs/SIMULATION_PHASE.md)
> - Full design and roadmap: [docs/PROJECT_BASELINE.md](docs/PROJECT_BASELINE.md)

This project builds on [`reachy2-robot-controller`](../reachy2-robot-controller) and reuses its SDK versions, safety limits, recorded poses and grasp workflow.

---

## How it works

```txt
MuJoCo (Reachy 2 model)
  → Gymnasium env (task, reward, safety shield)
  → RL baselines (PPO / SAC / TD3 / SAC+HER / TQC)
  → Sim-to-real (system ID, domain randomization, residual RL, fine-tuning)
  → Real Reachy 2 via reachy2-sdk
```

The environment talks to a single `RobotBackend` interface. Changing the backend is a config change:

| Backend | Purpose |
|---|---|
| `mujoco` | Fast training and evaluation |
| `pollen_sim` | Pollen's Docker simulation through the real SDK, to test deployment code |
| `real` | The physical robot (safety shield in strict mode) |

---

## Requirements

- Python **3.10** (tested with 3.10.18), to match the Python version on the Reachy 2 itself. Avoid 3.13+ because `grpcio` may fail to build.
- MuJoCo ≥ 3.x
- Gymnasium ≥ 1.0
- Stable-Baselines3 ≥ 2.x and `sb3-contrib`
- `reachy2-sdk==1.0.7` and `reachy2-sdk-api==1.0.11`. These are the versions confirmed on our robot.

---

## Installation

```bash
python3.10 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
python -m pip install --upgrade pip setuptools wheel
```

```bash
python -m pip install -r requirements.txt
```

---

## Usage

Activate the venv first:

```bash
source .venv/bin/activate
```

View the robot model. Use an **absolute** path, because MuJoCo 3.14 can't find the meshes with a relative one:

```bash
python -m mujoco.viewer --mjcf=$PWD/assets/reach_scene.xml
```

`reach_scene.xml` has the base fixed (like `mobile_base: null`) and a green goal marker. `test_scene.xml` is Pollen's original, with a free base, a table and a bottle.

Learning scripts, in order: `python scripts/ex1_one_joint.py`, `mjpython scripts/ex2_goto.py`, `mjpython scripts/ex3_keyboard.py`.

Drive the robot from a script. Scripts that use `mujoco.viewer.launch_passive` need `mjpython` on macOS:

```bash
mjpython scripts/wave_arm.py
```

| What | Command |
|---|---|
| Just look at a scene | `python -m mujoco.viewer --mjcf=$PWD/assets/<scene>.xml` |
| Script with `launch_passive` | `mjpython scripts/<script>.py` |
| Headless (training, tests) | `python scripts/<script>.py` |

### Planned

Train a baseline:

```bash
python scripts/train.py env=reach algo=sac seed=0
```

Evaluate a policy in simulation:

```bash
python scripts/evaluate.py --backend mujoco --checkpoint runs/reach_sac_0/best.zip
```

Evaluate in Pollen's Docker simulation:

```bash
python scripts/evaluate.py --backend pollen_sim --checkpoint runs/reach_sac_0/best.zip
```

Do a dry run on the real robot. Actions are logged but not sent:

```bash
python scripts/evaluate.py --backend real --dry-run --checkpoint runs/reach_sac_0/best.zip
```

Monitor training:

```bash
tensorboard --logdir runs
```

---

## Tasks

| ID | Task | Status |
|---|---|---|
| T0 | Joint tracking (sim vs. real calibration) | Planned |
| T1 | Reach: end-effector to 3D goal | Planned |
| T2 | Reach with orientation: 6D pose | Planned |
| T3 | Head tracking with `look_at` | Planned |
| T4 | Grasp ball | Planned |
| T5 | Handover | Stretch |
| T6 | Vision-based reach/grasp | Stretch |

---

## Project layout

```txt
reachy_simulator/
├── assets/        # Reachy 2 model (vendored from Pollen, see assets/reachy2/SOURCE.md) + scenes
├── third_party/   # full clone of pollen-robotics/reachy2_mujoco, reference only (gitignored)
├── configs/       # env / algo / safety / robot YAML
├── reachy_sim/    # envs, backends, safety, randomization, policies, utils
├── scripts/       # view_model, train, evaluate, sysid
├── tests/
├── docs/          # PROJECT_BASELINE.md, SIMULATION_PHASE.md, (later) SIM_TO_REAL.md, EXPERIMENTS.md
└── runs/          # checkpoints and logs (gitignored)
```

---

## Real-robot safety

Before running any policy on the real Reachy 2:

1. Keep the E-stop within reach and make sure nobody is in the arm workspace.
2. Keep `mobile_base: null` in the robot config.
3. Run the connection test from the parent project (`check_reachy_access.py`).
4. Always do a **dry run** first, then use reduced limits, then run the full evaluation.
5. Never disable the safety shield or watchdog on the `real` backend.

For all limits and the full protocol, see [docs/PROJECT_BASELINE.md §2.4, §5.4 and §9.3](docs/PROJECT_BASELINE.md).

---

## Roadmap

| Phase | Goal |
|---|---|
| P0 🟡 | Repo setup, Reachy 2 model loads and renders |
| P1 | Base env, MuJoCo backend, safety shield, T0/T1 |
| P2 | Baseline results (PPO, SAC, TD3, SAC+HER) |
| P3 | Real and Pollen-sim backends, dry-run mode |
| P4 | System ID + domain randomization, T1 on the real robot |
| P5 | Grasp with residual RL |
| P6 | Adaptation (RMA-style / real fine-tuning) |
| P7 | Vision (stretch) |

---

## References

- [Project baseline](docs/PROJECT_BASELINE.md)
- [Simulation phase plan](docs/SIMULATION_PHASE.md)
- [pollen-robotics/reachy2_mujoco](https://github.com/pollen-robotics/reachy2_mujoco)
- [Reachy 2 documentation](https://docs.pollen-robotics.com/)
- [MuJoCo](https://mujoco.readthedocs.io/) · [Gymnasium](https://gymnasium.farama.org/) · [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
