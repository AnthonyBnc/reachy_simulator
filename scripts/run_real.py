"""Run a trained Reach policy on the real Reachy 2 right arm. Dry run unless --live.

    python scripts/run_real.py --model runs/any/model.zip --no-wrist --goal 0.35 -0.30 -0.25
    python scripts/run_real.py --model runs/any/model.zip --no-wrist --goal 0.35 -0.30 -0.25 --live --speed 0.3
    python scripts/run_real.py --model runs/any/model.zip --no-wrist --goal 0.35 -0.30 -0.25 --fake --live

See docs/RUN_REAL.md.
"""

import argparse
import time

import gymnasium as gym
import mujoco
import numpy as np
from stable_baselines3 import SAC

import reachy_sim  # noqa: F401  (registers ReachyReach-v0)

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--goal", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"),
                    help="goal in the torso frame (m): x forward, y left, z up")
parser.add_argument("--no-wrist", action="store_true", help="required for models trained with --no-wrist")
parser.add_argument("--host", default="10.116.19.109")
parser.add_argument("--live", action="store_true", help="really move the robot (default: dry run)")
parser.add_argument("--speed", type=float, default=0.5, help="fraction of the trained max speed (0 to 1)")
parser.add_argument("--seconds", type=float, default=10.0)
parser.add_argument("--fake", action="store_true", help="test only: fake robot that copies commands")
args = parser.parse_args()

LAG_LIMIT_DEG = 15.0   # watchdog
SUCCESS_STEPS = 20     # 20 steps x 50 ms = held for 1 s

# ---------- 1. Same env and policy as training ----------
env = gym.make("ReachyReach-v0", control_wrist=not args.no_wrist).unwrapped
model = SAC.load(args.model)
env.reset(seed=0)
backend = env.backend
dt = backend.dt  # 0.05 s

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

# ---------- 4. Control loop (20 Hz) ----------
held, steps, interventions = 0, 0, 0
t0 = time.perf_counter()
try:
    while time.perf_counter() - t0 < args.seconds:
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
        if steps % 20 == 0 or reasons:
            print(f"t={steps * dt:4.1f}s  dist {100 * distance:5.1f} cm  "
                  f"cmd (deg) {np.round(np.degrees(env.q_cmd), 1)}  lag {lag:4.1f}  {' '.join(reasons)}")
        if held >= SUCCESS_STEPS:
            print(f"SUCCESS: held the goal for 1 s, final distance {100 * distance:.1f} cm.")
            break

        time.sleep(max(0.0, dt - (time.perf_counter() - start)))  # keep 50 ms per step
    else:
        print("Time limit reached.")
except KeyboardInterrupt:
    print("Stopped by Ctrl+C.")
finally:
    print(f"Shield interventions: {interventions} in {steps} steps.")
    if args.live:
        arm.turn_off_smoothly()  # lowers the arm slowly, always runs