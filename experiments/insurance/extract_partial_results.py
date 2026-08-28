"""Extract whatever insurance results we have from the live log file."""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

LOGS = Path(__file__).resolve().parents[2] / "logs" / "insurance"
RESULTS = Path(__file__).resolve().parents[2] / "results" / "insurance"


def parse_log(path: Path) -> dict:
    """Per-seed result: {iterations: [...], accuracy, mae, stability}."""
    iters = []
    last_iter = -1
    runs = []
    current = []
    for line in path.open():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("event") != "metrics":
            continue
        it = ev.get("iteration", 0)
        if it <= last_iter and current:
            runs.append(current)
            current = []
        current.append({
            "k": it,
            "accuracy": ev.get("accuracy", 0),
            "mae": ev.get("mae", 0),
            "stability": ev.get("stability", 0),
        })
        last_iter = it
    if current:
        runs.append(current)
    if not runs:
        return None
    last = runs[-1]
    return {
        "iterations": last,
        "accuracy": last[-1]["accuracy"],
        "mae": last[-1]["mae"],
        "stability": last[-1]["stability"],
    }


def main():
    by_method = defaultdict(dict)
    for log in sorted(LOGS.glob("*.jsonl")):
        m = re.match(r"^(.+)_s(\d+)$", log.stem)
        if not m:
            continue
        method, seed = m.group(1), m.group(2)
        r = parse_log(log)
        if r is not None:
            by_method[method][seed] = r

    out = {}
    for method, seeds in by_method.items():
        runs = list(seeds.values())
        if not runs:
            continue
        n = len(runs)
        out[method] = {
            "mean_accuracy": round(sum(r["accuracy"] for r in runs) / n, 4),
            "mean_mae": round(sum(r["mae"] for r in runs) / n, 4),
            "mean_stability": round(sum(r["stability"] for r in runs) / n, 4),
            "per_seed": runs,
            "n_seeds": n,
        }
    RESULTS.mkdir(exist_ok=True, parents=True)
    with open(RESULTS / "results.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"{'Method':<20} {'#seeds':>6} {'Acc':>6} {'MAE':>6} {'Stab':>6}")
    print("-" * 60)
    for m, d in out.items():
        print(f"{m:<20} {d['n_seeds']:>6} {d['mean_accuracy']:>6.3f} {d['mean_mae']:>6.3f} {d['mean_stability']:>6.3f}")


if __name__ == "__main__":
    main()
