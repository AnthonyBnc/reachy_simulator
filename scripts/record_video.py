"""Record a video of ReachyReach-v0 (headless, no window).

    python scripts/record_video.py                                  # random actions
    python scripts/record_video.py --model runs/reach_sac_0/model.zip   # a trained policy
"""

import argparse
from pathlib import Path

import cv2
import gymnasium as gym

import reachy_sim  # noqa: F401  (registers ReachyReach-v0)

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="path to a trained SAC model (default: random actions)")
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--out", default=str(ROOT / "runs" / "reach.mp4"))
args = parser.parse_args()

env = gym.make("ReachyReach-v0", render_mode="rgb_array")
model = None
if args.model:
    from stable_baselines3 import SAC

    model = SAC.load(args.model)

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
writer = None
for episode in range(args.episodes):
    obs, _ = env.reset(seed=episode)
    done = False
    while not done:
        action = model.predict(obs, deterministic=True)[0] if model else env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        frame = env.render()  # numpy image (height, width, 3), RGB
        if writer is None:
            height, width = frame.shape[:2]
            fps = env.metadata["render_fps"]  # 20 -> the video plays at real-time speed
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        text = f"episode {episode + 1}  distance {100 * info['distance']:.1f} cm"
        cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # OpenCV expects BGR
    print(f"episode {episode + 1}: success={info['is_success']}  final distance={100 * info['distance']:.1f} cm")

writer.release()
env.close()
print(f"Saved {args.out}")