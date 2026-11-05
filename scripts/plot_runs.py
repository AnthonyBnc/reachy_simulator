"""Plot simulation (dry run) vs real (live) results logged by run_real.py.

    python scripts/plot_runs.py
    python scripts/plot_runs.py --results results --speed 0.3
    python scripts/plot_runs.py --goal "0.35 -0.30 -0.25"     # one goal; PNGs go to results/goal_0.35_-0.30_-0.25/

Writes three PNGs into the results folder:
    compare_runs.png     final distance, time to reach and shield interventions per policy
    distance_curves.png  hand-to-goal distance over time, sim vs real, one panel per policy
    joint_curves.png     the 4 arm joint commands over time, sim vs real, one row per policy
Shield interventions are marked with an x.
Repeat runs of the same policy are averaged (bars) or all drawn (curves).
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--results", type=Path, default=Path("results"))
parser.add_argument("--speed", help="only runs at this speed, e.g. 0.3 (default: all)")
parser.add_argument("--goal", help='only runs at this goal, as logged, e.g. "0.35 -0.30 -0.25"')
args = parser.parse_args()

SIM, REAL = "#2a78d6", "#eb6834"   # categorical slots 1 and 2
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb"})

rows = [r for r in csv.DictReader(open(args.results / "runs.csv", newline=""))
        if r["mode"] in ("sim", "real") and (args.speed is None or r["speed"] == args.speed)
        and (args.goal is None or r["goal"] == args.goal)]
out_dir = args.results / ("goal_" + args.goal.replace(" ", "_") if args.goal else "")
out_dir.mkdir(parents=True, exist_ok=True)
by = defaultdict(lambda: defaultdict(list))  # policy -> mode -> runs
for r in rows:
    by[r["policy"]][r["mode"]].append(r)
policies = sorted(by)
labels = [p.replace("reach_", "").replace("_goal/", "\n").upper() for p in policies]  # FIXED\nSAC


JOINTS = ["shoulder pitch", "shoulder roll", "elbow yaw", "elbow pitch"]


def load_trace(run):
    """(time s, distance cm, shield used?, 7 joint commands deg) arrays of one run."""
    trace = list(csv.DictReader(open(args.results / "traces" / run["trace"], newline="")))
    t = np.array([float(s["t_s"]) for s in trace])
    d = np.array([float(s["dist_cm"]) for s in trace])
    shield = np.array([s["shield"] != "" for s in trace], dtype=bool)
    q = np.array([[float(s[f"cmd{i}_deg"]) for i in range(7)] for s in trace]).reshape(-1, 7)
    return t, d, shield, q


def mean(policy, mode, key):
    """Mean over successful runs; stopped runs (watchdog, ctrl+c) only count if none succeeded."""
    runs = [r for r in by[policy][mode] if r["result"] == "success"] or by[policy][mode]
    vals = [float(r[key]) for r in runs if r[key] != ""]
    return np.mean(vals) if vals else np.nan


# ---------- 1. Summary bars ----------
metrics = [("final_cm", "Final distance (cm)"), ("t_reach_s", "Time to reach 3 cm (s)"),
           ("interventions", "Shield interventions per run")]
fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
x = np.arange(len(policies))
w = 0.38
for ax, (key, title) in zip(axes, metrics):
    for i, (mode, color) in enumerate((("sim", SIM), ("real", REAL))):
        vals = [mean(p, mode, key) for p in policies]
        bars = ax.bar(x + (i - 0.5) * (w + 0.02), vals, w, color=color, label=mode,
                      hatch="//" if mode == "real" else None, edgecolor="#fcfcfb", linewidth=0)
        failed = []
        for p in policies:
            runs = by[p][mode]
            bad = [r["result"] for r in runs if r["result"] != "success"]
            ok = len(runs) - len(bad)
            failed.append(", ".join(sorted(set(bad))) if bad and not ok else "")  # only when no run succeeded
        texts = ["" if np.isnan(v) and not f else ("-" if np.isnan(v) else f"{v:.1f}") + (f"\n{f}" if f else "")
                 for v, f in zip(vals, failed)]
        ax.bar_label(bars, texts, fontsize=7, color=MUTED, padding=2)
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.margins(y=0.25)
    ax.set_ylim(bottom=0, top=max(ax.get_ylim()[1], 1))
axes[0].legend(["Simulation (dry run)", "Real robot (live)"], frameon=False, ncol=2,
               loc="upper left", bbox_to_anchor=(0, 1.35))
axes[-1].set_xticks(x, labels)
fig.suptitle("Reach policies: simulation vs real robot", x=0.01, ha="left", color=INK,
             fontsize=12, fontweight="bold")
fig.tight_layout()
out1 = out_dir / "compare_runs.png"
fig.savefig(out1, dpi=150)

# ---------- 2. Distance curves ----------
cols = 4
nrows = int(np.ceil(len(policies) / cols))
fig, axes = plt.subplots(nrows, cols, figsize=(12, 3 * nrows), sharex=True, sharey=True, squeeze=False)
for ax, policy, label in zip(axes.flat, policies, labels):
    for mode, color, style in (("sim", SIM, "-"), ("real", REAL, "--")):
        for r in by[policy][mode]:
            t, d, shield, _ = load_trace(r)
            ax.plot(t, d, style, color=color, linewidth=2, label=mode)
            ax.plot(t[shield], d[shield], "x", color=color, markersize=8, markeredgewidth=2)
    ax.axhline(3, color=MUTED, linewidth=0.8, linestyle=":")  # success threshold
    ax.set_title(label.replace("\n", " / "), loc="left", color=INK, fontsize=9)
    ax.grid(color=GRID, linewidth=0.8)
for ax in axes.flat[len(policies):]:
    ax.set_visible(False)
for ax in axes[-1]:
    ax.set_xlabel("time (s)")
for ax in axes[:, 0]:
    ax.set_ylabel("distance to goal (cm)")
handles = [plt.Line2D([], [], color=SIM, linewidth=2), plt.Line2D([], [], color=REAL, linewidth=2, linestyle="--"),
           plt.Line2D([], [], color=MUTED, linewidth=0.8, linestyle=":"),
           plt.Line2D([], [], color=MUTED, marker="x", linestyle="", markersize=8, markeredgewidth=2)]
LEGEND = ["Simulation (dry run)", "Real robot (live)", "3 cm success threshold", "shield intervention"]
fig.legend(handles, LEGEND, frameon=False, ncol=4, loc="upper right")
fig.suptitle("Hand-to-goal distance over time", x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
out2 = out_dir / "distance_curves.png"
fig.savefig(out2, dpi=150)

# ---------- 3. Joint commands ----------
fig, axes = plt.subplots(len(policies), len(JOINTS), figsize=(13, 1.9 * len(policies) + 0.8),
                         sharex=True, sharey="col", squeeze=False)
for row, (policy, label) in enumerate(zip(policies, labels)):
    for mode, color, style in (("sim", SIM, "-"), ("real", REAL, "--")):
        for r in by[policy][mode]:
            t, _, shield, q = load_trace(r)
            for j, ax in enumerate(axes[row]):
                ax.plot(t, q[:, j], style, color=color, linewidth=2)
                ax.plot(t[shield], q[shield, j], "x", color=color, markersize=7, markeredgewidth=2)
    axes[row, 0].set_ylabel(label.replace("\n", " / "), color=INK, fontsize=9)
for j, name in enumerate(JOINTS):
    axes[0, j].set_title(f"{name} (deg)", loc="left", color=INK, fontsize=10)
    axes[-1, j].set_xlabel("time (s)")
for ax in axes.flat:
    ax.grid(color=GRID, linewidth=0.8)
fig.legend([handles[0], handles[1], handles[3]], [LEGEND[0], LEGEND[1], LEGEND[3]],
           frameon=False, ncol=3, loc="upper right")
fig.suptitle("Joint commands over time", x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.97))
out3 = out_dir / "joint_curves.png"
fig.savefig(out3, dpi=150)

print(f"Saved {out1}\nSaved {out2}\nSaved {out3}")
