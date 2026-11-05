"""Run a trained Reach policy on the real Reachy 2 right arm. Dry run unless --live.

    python scripts/run_real.py --model runs/any/model.zip --no-wrist --goal 0.35 -0.30 -0.25
    python scripts/run_real.py --model runs/any/model.zip --no-wrist --goal 0.35 -0.30 -0.25 --live --speed 0.3
    python scripts/run_real.py --model runs/any/model.zip --no-wrist --goal 0.35 -0.30 -0.25 --fake --live

Every run adds one row to results/runs.csv and saves its steps to results/traces/.
Compare dry runs (sim) and live runs (real) with scripts/compare_runs.py. See docs/RUN_REAL.md.
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from reachy_sim.algos import load_model

import reachy_sim  # noqa: F401  (registers ReachyReach-v0)

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--goal", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"),
                    help="goal in the torso frame (m): x forward, y left, z up")
parser.add_argument("--no-wrist", action="store_true", help="required for models trained with --no-wrist")
parser.add_argument("--host", default="10.116.19.196")
parser.add_argument("--live", action="store_true", help="really move the robot (default: dry run)")
parser.add_argument("--speed", type=float, default=0.5, help="fraction of the trained max speed (0 to 1)")
parser.add_argument("--fake", action="store_true", help="test only: fake robot that copies commands")
parser.add_argument("--results", default="results", help="folder for runs.csv and traces/")
args = parser.parse_args()

LAG_LIMIT_DEG = 15.0   # watchdog
SUCCESS_STEPS = 20     # steps in a row within 3 cm

# ---------- 1. Same env and policy as training ----------
env = gym.make("ReachyReach-v0", control_wrist=not args.no_wrist).unwrapped
model = load_model(args.model)
env.reset(seed=0)
backend = env.backend
dt = backend.dt  # seconds per step (1 / control_hz in ReachEnv)

speed = float(np.clip(args.speed, 0.0, 1.0))
env.max_delta *= speed          # how far one action can move a joint
env.shield.max_delta *= speed   # the shield's step limit must shrink too

goal = env.set_goal(args.goal)  # set_goal already clips into the safety box
if not np.allclose(goal, args.goal):
    print(f"Goal moved inside the safety box: {np.round(goal, 3)}")


# ---------- 2. Robot ----------
class FakeJoint:
    def __init__(self):
        self.goal_position = 0.0


class FakeArm:
    """Copies every command exactly, so the live path can be tested without the robot."""

    def __init__(self):
        self.joints = {f"j{i}": FakeJoint() for i in range(7)}
        self._q = [0.0] * 7

    def get_current_positions(self):
        return list(self._q)

    def send_goal_positions(self, check_positions=True):
        self._q = [j.goal_position for j in self.joints.values()]

    def turn_on(self):
        pass

    def turn_off_smoothly(self):
        print("(fake) turn_off_smoothly")

    def goto(self, q, duration=4, wait=True):
        self._q = list(q)


if args.fake:
    arm = FakeArm()
else:
    from reachy2_sdk import ReachySDK

    reachy = ReachySDK(host=args.host)
    if not reachy.is_connected():
        raise SystemExit(f"Could not connect to Reachy at {args.host}")
    arm = reachy.r_arm


def read_q():
    """Real joint angles in radians, SDK order."""
    return np.radians(arm.get_current_positions())  # SDK gives degrees


def write_sim_state(q, qdot):
    """Put the real arm's state into MuJoCo, only so _get_obs() and predict_ee() see it."""
    backend.data.qpos[backend.qadr] = q
    backend.data.qvel[backend.dadr] = qdot
    mujoco.mj_forward(backend.model, backend.data)  # kinematics only, no physics step


def send(q_cmd):
    """Send 7 targets (radians) to the real arm."""
    for joint, deg in zip(arm.joints.values(), np.degrees(q_cmd)):
        joint.goal_position = float(deg)
    arm.send_goal_positions()


# ---------- 3. Before moving ----------
q_real = read_q()
print("Arm now (deg):", np.round(np.degrees(q_real), 1))
print("Goal (m):", np.round(goal, 3))
print("Mode:", "LIVE" if args.live else "DRY RUN", f"speed {speed:.0%}")

if args.live:
    if input("E-stop in hand, nobody near the arm? Type yes: ").strip() != "yes":
        raise SystemExit("Cancelled.")
    arm.turn_on()
    if np.max(np.abs(np.degrees(q_real))) > 10:
        if input("Arm is far from zero. Move it to zero in 4 s? Type yes: ").strip() != "yes":
            arm.turn_off_smoothly()
            raise SystemExit("Cancelled.")
        arm.goto([0.0] * 7, duration=4, wait=True)

# Start exactly like training: zero pose, zero previous action
q_start = read_q() if args.live else np.zeros(7)
backend.reset(q_start)
env.q_cmd = q_start.copy()  # last command, NOT the measured angle (or the arm sags)
env.prev_action = np.zeros(env.n_act)
q_prev = q_start.copy()

# ---------- 4. Control loop (no time limit: stops on success, watchdog or Ctrl+C) ----------
held, steps, interventions = 0, 0, 0
result, distance, max_lag, t_reach, trace = "error", float("nan"), 0.0, None, []
try:
    while True:
        start = time.perf_counter()

        if args.live:
            # Read the real arm, speed by finite difference
            q = read_q()
            write_sim_state(q, (q - q_prev) / dt)
            q_prev = q

            # Watchdog: real arm too far behind the last command
            lag = float(np.max(np.abs(np.degrees(env.q_cmd - q))))
            if lag > LAG_LIMIT_DEG:
                print(f"STOP: arm is {lag:.1f} deg behind the command")
                result, max_lag = "watchdog", max(max_lag, lag)  # log the lag that tripped it
                break

            # Observation -> policy
            obs = env._get_obs()
            action, _ = model.predict(obs, deterministic=True)

            # Same maths as ReachEnv.step(): action -> target -> shield
            a = np.clip(action, -1.0, 1.0)
            delta = np.zeros(7)
            delta[:env.n_act] = a * env.max_delta
            env.q_cmd, shield = env.shield.filter(env.q_cmd, env.q_cmd + delta,
                                                  predict_ee=backend.predict_ee)

            send(env.q_cmd)
            env.prev_action = a
            distance = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))
            reasons = shield["reasons"]
        else:
            # Dry run: the env does everything in simulation, nothing is sent
            obs = env._get_obs()
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, _, info = env.step(action)
            distance, reasons = info["distance"], info["shield_reasons"]
            lag = 0.0

        # Success = within 3 cm for 20 steps in a row
        steps += 1
        interventions += bool(reasons)
        held = held + 1 if distance < env.success_threshold else 0
        max_lag = max(max_lag, lag)
        if t_reach is None and distance < env.success_threshold:
            t_reach = steps * dt
        trace.append([round(steps * dt, 3), round(100 * distance, 2), round(lag, 2), " ".join(reasons),
                      *np.round(np.degrees(env.q_cmd), 2)])
        if steps % max(1, round(1 / dt)) == 0 or reasons:  # about once a second
            print(f"t={steps * dt:4.1f}s  dist {100 * distance:5.1f} cm  "
                  f"cmd (deg) {np.round(np.degrees(env.q_cmd), 1)}  lag {lag:4.1f}  {' '.join(reasons)}")
        if held >= SUCCESS_STEPS:
            print(f"SUCCESS: held the goal for {SUCCESS_STEPS * dt:.0f} s, final distance {100 * distance:.1f} cm.")
            result = "success"
            break

        time.sleep(max(0.0, dt - (time.perf_counter() - start)))  # keep dt per step
except KeyboardInterrupt:
    print("Stopped by Ctrl+C.")
    result = "ctrl+c"
finally:
    print(f"Shield interventions: {interventions} in {steps} steps.")
    if args.live:
        arm.turn_off_smoothly()  # lowers the arm slowly, always runs

    # ---------- 5. Log the run ----------
    model_path = Path(args.model)
    policy = f"{model_path.parent.parent.name}/{model_path.parent.name}"  # e.g. reach_fixed_goal/sac
    mode = ("fake" if args.fake else "real") if args.live else "sim"
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = Path(args.results)
    trace_file = out / "traces" / f"{stamp}_{mode}_{policy.replace('/', '_')}.csv"
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    with open(trace_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "dist_cm", "lag_deg", "shield"] + [f"cmd{i}_deg" for i in range(7)])
        w.writerows(trace)
    runs_file = out / "runs.csv"
    new = not runs_file.exists()
    with open(runs_file, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "policy", "mode", "goal", "speed", "hz", "result", "steps",
                        "t_reach_s", "final_cm", "interventions", "max_lag_deg", "trace"])
        w.writerow([stamp, policy, mode, " ".join(f"{g:.2f}" for g in goal), speed, round(1 / dt), result, steps,
                    "" if t_reach is None else round(t_reach, 2), round(100 * distance, 2), interventions,
                    round(max_lag, 2), trace_file.name])
    print(f"Logged to {runs_file}")