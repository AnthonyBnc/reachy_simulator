"""Plot episode reward and success rate against training steps.

    python scripts/plot_rewards.py runs/any_20261020_101500
    python scripts/plot_rewards.py runs/any_* --out runs/compare.png    # several runs on one graph
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("runs", nargs="+", type=Path, help="run folders containing monitor.monitor.csv")
parser.add_argument("--window", type=int, default=50, help="moving average over N episodes")
parser.add_argument("--out", type=Path, help="output PNG (default: <first run>/reward.png)")
args = parser.parse_args()

fig, (ax_reward, ax_success) = plt.subplots(2, 1, sharex=True, figsize=(9, 7))
for run in args.runs:
    if not (run / "monitor.monitor.csv").exists():  # e.g. the TensorBoard folders runs/<run>_1 matched by a *
        print(f"{run}: no monitor.monitor.csv, skipped")
        continue
    with open(run / "monitor.monitor.csv") as f:
        next(f)  # the first line is a JSON comment written by Monitor
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"{run}: no finished episodes yet, skipped")
        continue

    steps = np.cumsum([int(r["l"]) for r in rows])          # training step at the end of each episode
    reward = np.array([float(r["r"]) for r in rows])        # total reward of each episode
    success = np.array([r["is_success"] == "True" for r in rows], dtype=float)

    w = max(1, min(args.window, len(rows) // 5))  # short runs: smaller window, so there is still a curve
    smooth = lambda y: np.convolve(y, np.ones(w) / w, mode="valid")  # moving average

    raw, = ax_reward.plot(steps, reward, alpha=0.2)
    ax_reward.plot(steps[w - 1:], smooth(reward), color=raw.get_color(), label=f"{run.name} (avg of {w})")
    ax_success.plot(steps[w - 1:], 100 * smooth(success), color=raw.get_color(), label=run.name)

ax_reward.set_ylabel("episode reward")
ax_reward.set_title("Reward and success vs. training steps (faint: each episode, solid: moving average)")
ax_success.set_ylabel("success (%)")
ax_success.set_xlabel("training steps")
for ax in (ax_reward, ax_success):
    ax.grid(alpha=0.3)
    ax.legend()

out = args.out or args.runs[0] / "reward.png"
fig.tight_layout()
fig.savefig(out, dpi=120)
print(f"Saved {out}")
