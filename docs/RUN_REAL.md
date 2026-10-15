# Running the trained Reach policy on the real Reachy 2 (`scripts/run_real.py`)

This script takes a policy trained in simulation (`scripts/train.py`) and runs it on the
**real Reachy 2 right arm**: the hand moves to a 3D goal point and holds it there.
By default it is a **dry run**: it connects to the robot and reads the arm, but sends nothing.

---

## 1. What the script does

The script runs the same control loop as training (`ReachEnv.step()` in
`reachy_sim/envs/reach_env.py`). The only differences are where the joint angles come from
and where the commands go:

| | Training (simulation) | `run_real.py --live` | `run_real.py` (dry run) |
|---|---|---|---|
| Joint angles come from | MuJoCo | the real robot (`r_arm.get_current_positions()`) | MuJoCo, starting from the real arm's pose |
| Commands go to | MuJoCo | the real robot (`send_goal_positions`) | MuJoCo only, nothing is sent |
| Safety shield | yes | yes, at reduced speed | yes, at reduced speed |

MuJoCo is still used on the real robot, but **only to compute the hand position from the
joint angles**. `scripts/compare_fk.py` showed this matches the robot's own kinematics
(0.0 mm error on every test pose, see `docs/SIM_TO_REAL.md`).

### Control loop (20 Hz, 50 ms per step)

1. **Read the state.** Live: read the 7 real joint angles (degrees, converted to radians),
   and compute joint speed as `(q - q_prev) / 0.05`. Dry run: read the simulated arm.
2. **Build the observation.** The real angles are written into the MuJoCo model, then the
   env's own `_get_obs()` is called. The policy therefore sees exactly the same layout as in
   training: joint angles (7), joint speeds (7), hand position (3), goal − hand (3),
   previous action (4 with `--no-wrist`, 7 otherwise).
3. **Ask the policy.** `model.predict(obs, deterministic=True)` returns one value in [−1, 1]
   per controlled joint.
4. **Turn it into a target.** `target = q_cmd + action × max_delta`, where `max_delta` is
   5° per step (100°/s) × `--speed`.
5. **Safety shield** (`reachy_sim/safety/shield.py`, the same code as in training):
   - limits each joint's change per step
   - keeps joints inside the soft limits in `configs/safety/default.yaml`
   - never moves the hand further outside the workspace box
6. **Send.** Live: set `goal_position` on all 7 joints, then `r_arm.send_goal_positions()`.
   Dry run: step the MuJoCo simulation by 50 ms.
7. **Check success.** Success means the hand stays within **3 cm** of the goal for **1 s**
   (20 steps in a row).

`q_cmd` is the **last command sent**, not the measured angle. Building each new target from
the measured angle makes the arm sag under gravity. Training uses the same convention.

---

## 2. Safety features

| Feature | What it does |
|---|---|
| Dry run by default | Nothing moves unless `--live` is given. |
| Reduced speed | `--speed 0.5` (default) = 50°/s max per joint; training used 100°/s. |
| Confirmation | In live mode you must type `yes` (E-stop in hand, nobody near the arm). |
| Start pose check | Training always started within 3° of the zero pose (arm straight down). If any joint is more than 10° away, live mode asks to move the arm to zero in 4 s first. The dry run starts from zero too, so it matches. |
| Safety shield | Same limits as in training: joint limits, step limit, workspace box. |
| Tracking watchdog | Live: if the real arm is more than 15° behind the command, the script stops. |
| Goal clipping | The goal is moved inside the safety box if it is outside; a message is printed. |
| Safe stop | On success, the time limit, an error or Ctrl+C, the arm is turned off with `turn_off_smoothly()` and lowers slowly. |

**The E-stop is still the main safety device.** Keep it in hand during every live run.

---

## 3. Setting up another computer

### 3.1 Requirements
- Python **3.10** (the same version as on the Reachy 2)
- On the same network as the robot (default IP `10.116.19.109`)

### 3.2 Get the code and install
```bash
git clone https://github.com/AnthonyBnc/reachy_simulator.git
cd reachy_simulator
python -m venv .venv
```
Activate the venv. On Windows PowerShell:
```bash
.venv\Scripts\Activate.ps1
```
On macOS / Linux:
```bash
source .venv/bin/activate
```
Then install the project and its dependencies:
```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

### 3.3 Get the trained model
`runs/` is in `.gitignore`, so **the model is not in the repository by default**. Either:

- **Copy it by hand** (USB, cloud drive) to `runs/any/model.zip` on the new computer, or
- **Force-add it to git** on the training computer (the file is about 3.4 MB):
  ```bash
  git add -f runs/any/model.zip
  ```

### 3.4 Check the setup
```bash
python scripts/check_machine.py
python -m pytest -q
python scripts/compare_fk.py --host 10.116.19.109
```
Expected: `RESULT: USABLE`, all tests pass, and `pos err` under 10 mm on every pose.

---

## 4. Usage

### Arguments

| Argument | Default | Meaning |
|---|---|---|
| `--model` | required | Path to the trained model, e.g. `runs/any/model.zip` |
| `--goal X Y Z` | required | Goal in the **torso frame**, in metres: x forward, y left, z up |
| `--no-wrist` | off | **Required** for models trained with `--no-wrist` (our `runs/any` model) |
| `--host` | `10.116.19.109` | Robot IP |
| `--live` | off | Really move the robot. Without it: dry run. |
| `--speed` | `0.5` | Fraction of the trained max speed (0 to 1) |
| `--seconds` | `10` | Time limit |

**Allowed goal region** (safety box, `configs/safety/default.yaml`):
x 0.15 to 0.65 m, y −0.60 to 0.05 m, z −0.40 to 0.25 m.
A good first goal, close to the arm and low: `0.35 -0.30 -0.25`.

### Step 1: dry run (nothing moves)
```bash
python scripts/run_real.py --model runs/any/model.zip --no-wrist --goal 0.35 -0.30 -0.25
```
**Good result:** `dist` gets smaller on each line, then `SUCCESS`, with few shield
interventions. **Bad result:** `Time limit reached`, `dist` not shrinking, or `workspace` /
`joint_limit` on most lines. Do not go live in that case.

### Step 2: live, slow
E-stop in hand, nobody near the arm:
```bash
python scripts/run_real.py --model runs/any/model.zip --no-wrist --goal 0.35 -0.30 -0.25 --live --speed 0.3
```
Type `yes` at each prompt. The arm may first move to the zero pose (4 s), then move to the goal.

### Step 3: increase gradually
`--speed 0.5`, then other goals inside the safety box. Only one change at a time.

---

## 5. Reading the output

```
Arm now (deg): [ 0.  0.  0.  0.  0.  0.  0.]  hand [ 0.03  -0.368 -0.638]
Goal (m):      [ 0.35 -0.3  -0.25]  distance 50.7 cm
Mode: LIVE, speed 30%
t= 1.0s  dist  18.0 cm  cmd (deg) [ -8.7 -12.4  -0.6 -52.1   0.    0.    0. ]  lag  2.5
...
SUCCESS: held the goal for 1 s, final distance 1.8 cm.
Shield interventions: 1 in 57 steps.
```

| Field | Meaning |
|---|---|
| `dist` | Hand-to-goal distance (cm). Should go down. |
| `cmd (deg)` | Joint targets sent, in SDK order: shoulder pitch, shoulder roll, elbow yaw, elbow pitch, wrist roll, wrist pitch, wrist yaw. |
| `lag` | Largest gap between command and real angle (degrees). Above 15° stops the run. |
| `step_limit` / `joint_limit` / `workspace` | The safety shield changed the command on that step, and why. |

---

## 6. Troubleshooting

| Message | Cause | What to do |
|---|---|---|
| `Could not connect to Reachy` | Wrong IP, or not on the robot's network | Check `--host`; run `check_reachy_access.py` from `reachy2-robot-controller` |
| `Goal moved inside the safety box` | Goal outside the allowed region | Pick a goal inside the box (section 4) |
| `STOP: arm is N deg behind the command` | The motors don't follow fast enough | Lower `--speed` |
| `Time limit reached` | Goal not reached in time | Check the dry run first; try a closer goal or a longer `--seconds` |
| Many `workspace` / `joint_limit` lines | The policy pushes against the safety limits | Choose a different goal; don't increase speed |
| Shape error when loading the model | `--no-wrist` doesn't match the training | Use `--no-wrist` for models trained with it, and leave it out otherwise |

---

## 7. Known limitations

- **Only the right arm**, shoulder and elbow (4 joints). With `--no-wrist` the wrist stays at
  its starting angle. The gripper is not used.
- **No domain randomization.** The policy has only seen the simulator's exact physics, so expect
  it to be less precise on the real arm than in simulation (97% success, 1.0 cm mean error).
- **Shield interventions were 7.8% of steps in simulation** (project target: under 1%).
- **One fixed goal per run.** There's no moving target on the real robot yet (`follow_ball.py`
  is simulation only).
- **Tested so far:** the live path against a fake robot that copies commands exactly
  (success in 2.8 s, 1 shield intervention). Not yet tested: the dry-run path, and anything
  on the real robot.