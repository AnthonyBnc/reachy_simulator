"""
Train an RL policy on ReachyReach-v0 (headless) and record a video of the whole training.

    python scripts/train.py --name fixed --goal 0.40 -0.20 0.00 --steps 30000    # one fixed point
    python scripts/train.py --name any --no-wrist --steps 200000                  # any point, 4 joints
    python scripts/train.py --name any --no-wrist --algo td3 --steps 200000       # sac (default), ppo, td3, ddpg

Every run gets its own folder runs/<name>_<date>_<time>/ (never overwritten) with:
    args.json            how this policy was trained (goal, steps, --no-wrist, ...)
    monitor.monitor.csv  one line per episode: reward, length, time, success
    checkpoints/         the policy after every 10% of training
    model.zip            the final policy
    training.mp4         one test episode at the start and after every 10% of training
Plot reward vs steps with:  python scripts/plot_rewards.py runs/<run folder>
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

import reachy_sim  # noqa: F401  (registers ReachyReach-v0)
from reachy_sim.algos import ALGOS

ROOT = Path(__file__).resolve().parents[1]


class TrainingVideo(BaseCallback):
    """Every `every` steps, run one test episode with the current policy and add it to one MP4."""

    def __init__(self, path, every, total_steps, fixed_goal=None, control_wrist=True, seed=123):
        super().__init__()
        self.path, self.every, self.total_steps, self.seed = str(path), every, total_steps, seed
        # A separate env only for filming, so the training env stays headless and fast
        self.video_env = gym.make("ReachyReach-v0", render_mode="rgb_array", fixed_goal=fixed_goal,
                                  control_wrist=control_wrist)
        self.writer = None

    def _record_episode(self):
        obs, _ = self.video_env.reset(seed=self.seed)  # same seed -> same goal in every clip
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = self.video_env.step(action)
            done = terminated or truncated

            frame = self.video_env.render()
            if self.writer is None:
                height, width = frame.shape[:2]
                fps = self.video_env.metadata["render_fps"]
                self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            status = "ON GOAL" if info["is_success"] else ""
            cv2.putText(frame, f"training step {self.num_timesteps:,} / {self.total_steps:,}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"distance {100 * info['distance']:.1f} cm  {status}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 255, 120) if status else (255, 255, 255), 2)
            self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"[video] step {self.num_timesteps:,}: final distance {100 * info['distance']:.1f} cm")

    def _on_training_start(self):
        self._record_episode()  # before any learning: the untrained policy

    def _on_step(self):
        if self.num_timesteps % self.every == 0:
            self._record_episode()
        return True  # keep training

    def _on_training_end(self):
        if self.writer is not None:
            self.writer.release()
        self.video_env.close()
        print(f"Saved {self.path}")


parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True, help="run name, saved under runs/<name>/")
parser.add_argument("--goal", type=float, nargs=3, metavar=("X", "Y", "Z"),
                    help="fixed goal in the torso frame (m). Omit for random goals.")
parser.add_argument("--algo", choices=ALGOS, default="sac", help="RL algorithm (default: sac)")
parser.add_argument("--steps", type=int, default=100_000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--video-every", type=int, help="film a test episode and save a checkpoint every N steps (default: 10%% of --steps)")
parser.add_argument("--no-video", action="store_true")
parser.add_argument("--no-wrist", action="store_true",
                    help="control only the 4 shoulder/elbow joints; the wrist stays at 0")
args = parser.parse_args()

run_name = f"{args.name}_{args.algo}_{datetime.now():%Y%m%d_%H%M%S}"  # a new folder every run, never overwrite
run_dir = ROOT / "runs" / run_name
run_dir.mkdir(parents=True)
(run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))  # how this policy was trained

env = Monitor(gym.make("ReachyReach-v0", fixed_goal=args.goal, control_wrist=not args.no_wrist),
              filename=str(run_dir / "monitor"), info_keywords=("is_success",))
extra ={}
if args.algo in("td3", "ddpg"):
    n = env.action_space.shape[0]
    extra["action_noise"] = NormalActionNoise(mean=np.zeros(n), sigma=0.1 * np.ones(n)) 
model = ALGOS[args.algo]("MultiInputPolicy", env, seed=args.seed, verbose=1,
                         tensorboard_log=str(ROOT / "runs"), **extra)

every = args.video_every or max(1, args.steps // 10)
callbacks = [CheckpointCallback(save_freq=every, save_path=str(run_dir / "checkpoints"), name_prefix="model")]
if not args.no_video:
    callbacks.append(TrainingVideo(run_dir / "training.mp4", every, args.steps, fixed_goal=args.goal,
                                   control_wrist=not args.no_wrist))

log_interval = 1 if args.algo == "ppo" else 20  # PPO counts rollouts of 2048 steps, the others count episodes
model.learn(total_timesteps=args.steps, tb_log_name=run_name, log_interval=log_interval, callback=callbacks)
model.save(run_dir / "model")
print(f"Saved runs/{run_name}/model.zip")
