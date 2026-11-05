"""Compare simulation (dry run) and real (live) results logged by run_real.py.

    python scripts/compare_runs.py
    python scripts/compare_runs.py --results results --include-fake

One line per policy, goal and speed: sim next to real, averaged over all runs of each.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--results", default="results")
parser.add_argument("--include-fake", action="store_true", help="also show --fake test runs")
args = parser.parse_args()

rows = list(csv.DictReader(open(Path(args.results) / "runs.csv", newline="")))
modes = ["sim", "real"] + (["fake"] if args.include_fake else [])
groups = defaultdict(lambda: defaultdict(list))
for r in rows:
    if r["mode"] in modes:
        groups[(r["policy"], r["goal"], r["speed"])][r["mode"]].append(r)


def summary(runs):
    """'ok/n  final cm  reach s  shield  lag' for a list of runs, or '-' if none."""
    if not runs:
        return "-"
    ok = sum(r["result"] == "success" for r in runs)
    final = np.mean([float(r["final_cm"]) for r in runs])
    reach = [float(r["t_reach_s"]) for r in runs if r["t_reach_s"]]
    reach = f"{np.mean(reach):5.1f}" if reach else "    -"
    shield = np.mean([int(r["interventions"]) for r in runs])
    lag = max(float(r["max_lag_deg"]) for r in runs)
    return f"{ok}/{len(runs)}  {final:5.1f}  {reach}  {shield:5.1f}  {lag:4.1f}"


cols = "ok/n  final  reach  shield  lag"
print(f"{'policy':26} {'goal':18} {'speed':6}" + "   ".join(f"{m.upper():^{len(cols)}}" for m in modes))
print(f"{'':26} {'':18} {'':6}" + "   ".join(cols for _ in modes))
print("units: final cm, time to reach s, shield interventions per run, max lag deg")
print("-" * (53 + len(modes) * (len(cols) + 3)))
for (policy, goal, speed), by_mode in sorted(groups.items()):
    print(f"{policy:26} {goal:18} {speed:6}" + "   ".join(f"{summary(by_mode[m]):{len(cols)}}" for m in modes))
