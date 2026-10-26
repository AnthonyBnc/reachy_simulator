# Task 1 — Reach: training guide

How to train, evaluate and compare RL policies for **Task 1 (Reach)**: Reachy 2 moves its **right hand** to a 3D point and **holds it there**.

To run a trained policy on the real robot, see [RUN_REAL.md](RUN_REAL.md).

---

## 1. Quick start

```bash
cd reachy_simulator
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1

# Train SAC on random goals, 4 joints (~15 min on a laptop CPU)
python scripts/train.py --name reach --no-wrist --steps 100000

# Test it on 100 goals it never saw
python scripts/evaluate.py --model runs/reach_sac_<date_time>/model.zip --no-wrist

# Plot reward and success vs. training steps
python scripts/plot_rewards.py runs/reach_sac_<date_time>
```

`<date_time>` is printed at the end of training (`Saved runs/reach_sac_20261026_152210/model.zip`).

---

## 2. The task

| Element | Definition |
|---|---|
| **Goal** | A 3D point in front of the robot (the green ball), in the torso frame: x forward, y left, z up. Either one **fixed** point, or a new **random** reachable point every episode inside the workspace box (x 0.15…0.65, y −0.60…0.05, z −0.40…0.25 m). |
| **Success** | The hand tip (`r_arm_tip`) is within **3 cm** of the goal and **stays** there for the last second of the episode. |
| **Action** | One number in [−1, 1] per joint = change of that joint's target by up to ±5° per step. 4 joints with `--no-wrist` (shoulder pitch/roll, elbow yaw/pitch; the wrist stays at 0°), otherwise 7. |
| **Observation** | Joint angles, joint velocities, hand position, direction to the goal (goal − hand), previous action, and the goal. |
| **Reward** | −distance to the goal, +1 for every step within 3 cm, small penalties for large or jerky actions, −0.1 each time the safety shield has to correct a command. |
| **Episode** | 100 steps at 20 Hz = 5 s. Starts with the arm hanging at home (±3°). No early ending: the policy must reach **and hold**. |
| **Safety** | Every command goes through the safety shield: at most 5° per joint per step, a safe working range per joint (`configs/safety/default.yaml`), and the hand may not move further out of the workspace box. |

The environment is `ReachyReach-v0` (`reachy_sim/envs/reach_env.py`):

```python
import gymnasium as gym
import reachy_sim  # registers ReachyReach-v0

env = gym.make("ReachyReach-v0", control_wrist=False)                                # random goals, 4 joints
env = gym.make("ReachyReach-v0", fixed_goal=[0.40, -0.20, 0.00], control_wrist=False)  # one fixed goal
```

---

## 3. Training

```bash
python scripts/train.py --name <name> [options]
```

| Option | Default | Meaning |
|---|---|---|
| `--name` | (required) | Start of the run folder name. Use the same name for all runs of one experiment. |
| `--algo` | `sac` | `sac`, `ppo`, `td3` or `ddpg` |
| `--goal X Y Z` | random | Train on one fixed goal (meters, torso frame) instead of random goals |
| `--no-wrist` | off | Control only the 4 shoulder/elbow joints. **Recommended**: with 7 joints the policy pins the wrist against its limits and random-goal success drops from 97% to 72%. |
| `--steps` | 100,000 | Training steps (1 step = 50 ms of simulation) |
| `--seed` | 0 | Random seed. Use 0, 1, 2 to repeat an experiment. |
| `--video-every` | 10% of `--steps` | How often to film a test episode **and** save a checkpoint |
| `--no-video` | off | Skip the training video (slightly faster) |

### Recommended settings

| Goal | Algorithm | Steps | Expected success |
|---|---|---|---|
| Fixed | any | 100,000 (SAC learns it in ~30k) | 100% |
| Random | SAC | 100,000 | ~97% |
| Random | PPO | 500,000 | ~96% |
| Random | TD3 | 100,000 | ~72% |
| Random | DDPG | 100,000 | ~4% |

PPO needs more steps because it is **on-policy**: it learns from each batch of 2,048 steps and then discards it, while SAC, TD3 and DDPG reuse all past experience from a replay buffer. PPO computes less per step, so 500k PPO steps take about as long as 100k SAC steps.

### Examples

```bash
# Random goals, 4 joints, SAC
python scripts/train.py --name reach --no-wrist --steps 100000

# One fixed goal
python scripts/train.py --name fixed --goal 0.40 -0.20 0.00 --no-wrist --steps 30000

# Another algorithm
python scripts/train.py --name reach --no-wrist --algo ppo --steps 500000
```

### What a run saves

Every run gets its own folder; nothing is ever overwritten:

```txt
runs/<name>_<algo>_<date>_<time>/
├── args.json             how it was trained (algorithm, goal, steps, --no-wrist, seed)
├── monitor.monitor.csv   one line per episode: reward (r), length (l), time (t), is_success
├── checkpoints/          the policy after every 10% of training: model_<steps>_steps.zip
├── model.zip             the final policy
└── training.mp4          one test episode at the start and after every 10% of training
```

Stable-Baselines3 also creates `runs/<name>_<algo>_<date>_<time>_1/` with the TensorBoard logs.
`runs/` is ignored by Git, so results stay on the machine that trained them.

### Watching a run

```bash
tensorboard --logdir runs                         # live curves in the browser
wc -l runs/<name>_*_2*/monitor.monitor.csv       # finished episodes = lines - 2
```

---

## 4. Comparing algorithms (in parallel)

```bash
export OMP_NUM_THREADS=2        # each run uses 2 CPU threads, so 4 runs don't fight over the cores

for a in sac td3 ddpg; do
  python scripts/train.py --name cmp --no-wrist --algo $a --steps 100000 > runs/cmp_$a.log 2>&1 &
done
python scripts/train.py --name cmp --no-wrist --algo ppo --steps 500000 > runs/cmp_ppo.log 2>&1 &

jobs                            # 4 running jobs; keep this terminal open until it is empty
```

Add `--goal 0.40 -0.20 0.00` (and a different `--name`, e.g. `fixed_goal`) for a fixed-goal comparison.

---

## 5. Plotting

```bash
python scripts/plot_rewards.py runs/cmp_sac_2* runs/cmp_ppo_2* runs/cmp_td3_2* runs/cmp_ddpg_2* --out runs/compare.png
```

- **Top:** reward per episode (faint) and its moving average (solid).
- **Bottom:** success rate (moving average).
- The legend shows the algorithm name (and the seed, if not 0), read from each run's `args.json`.
- The order of the folders on the command line is the order in the legend.
- `--window N` changes the moving average (default 50 episodes). TensorBoard folders (`*_1`) are skipped automatically.

---

## 6. Evaluating

```bash
python scripts/evaluate.py --model runs/<run>/model.zip --no-wrist                        # random goals
python scripts/evaluate.py --model runs/<run>/model.zip --goal 0.40 -0.20 0.00 --no-wrist  # fixed goal
python scripts/evaluate.py --model runs/<run>/checkpoints/model_50000_steps.zip --no-wrist # a checkpoint
```

It runs 100 episodes with goals and start poses never used in training, and prints:

| Line | Meaning |
|---|---|
| `reached the goal` | % of episodes where the hand got within 3 cm at some point |
| `reached AND held (1 s)` | **The success rate**: within 3 cm for the whole last second |
| `final distance` | Distance at the end of the episode, mean and worst |
| `time to reach` | Seconds until the hand first got within 3 cm |
| `shield interventions` | How often the safety shield had to correct a command, per episode |

The algorithm is read from `args.json`, so no `--algo` is needed. **Use the same `--goal` and `--no-wrist` as in training**, or the test doesn't match what the policy learned.

To evaluate every run of an experiment:

```bash
for d in runs/cmp_*_2*/; do [ -f "$d/model.zip" ] && echo "== $d" && python scripts/evaluate.py --model "$d/model.zip" --no-wrist; done
```

---

## 7. Watching a trained policy

```bash
# Move the green ball (keys or mouse); the hand follows it. Needs a random-goal, --no-wrist model.
mjpython scripts/follow_ball.py --model runs/<run>/model.zip      # macOS
python scripts/follow_ball.py --model runs/<run>/model.zip        # Windows / Linux
```

Keys: arrows = forward / back / left / right, Page Up / Page Down = up / down (click the viewer first).
Mouse: double-click the ball, then Ctrl + right-drag.

Every run also has its `training.mp4`.

---

## 8. Results so far (seed 0, default hyperparameters, 100 unseen test goals)

| | SAC | PPO | TD3 | DDPG |
|---|---|---|---|---|
| Random goals | **97%** (100k) | **96%** (500k) | 72% (100k) | 4% (100k) |
| Fixed goal (500k) | 100%, 0.3 cm | 100%, 0.3 cm | 100%, 0.8 cm | 100%, 1.5 cm |

A fixed goal is solved by every algorithm; random goals separate them. Fallback order if an algorithm doesn't learn a task: **SAC → PPO → TD3 → DDPG**.

These are single runs; repeat with `--seed 1` and `--seed 2` before relying on the ranking.

---

## 9. Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `command not found: python` / `pip` / `pytest` | The venv isn't active | `source .venv/bin/activate` |
| `No module named 'reachy_sim'` | Package not installed | `pip install -e .` |
| Error about an action or observation shape when loading a model | `--no-wrist` doesn't match training | Add or remove `--no-wrist` |
| A fixed-goal model scores ~0% | Evaluated on random goals | Add the same `--goal X Y Z` |
| `plot_rewards.py` shows a run as `skipped` | The folder has no `monitor.monitor.csv` (a TensorBoard `_1` folder, or a run from before logging existed) | Expected |
| Training stopped | The terminal was closed, or the computer slept | Keep the terminal open and the computer plugged in and awake |
| `mujoco.viewer` crashes on macOS | Standalone viewer started with `mjpython` | `python -m mujoco.viewer`; use `mjpython` only for scripts with `launch_passive` |

### Known limitations

- `--no-wrist` is not read from `args.json`; it must be given to `evaluate.py` and matches `follow_ball.py`'s 4-joint setup.
- `scripts/record_video.py` still uses the 7-joint, fixed-goal setup from phase 1; use each run's `training.mp4` instead.
