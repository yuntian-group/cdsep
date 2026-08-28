"""Failure-mode analysis from existing JSONL experiment logs.

We extract per-iteration stability traces from the ``metrics`` events emitted
by each experiment, then plot:

* per-iteration stability (% of episodes that reached ``terminate`` cleanly
  with no parse / validation / routing errors) for naive vs ours, averaged
  over seeds;
* aggregate failure rates per (experiment, method) for the paper table.

Output:
    figures/failure_breakdown.pdf      -- per-iteration stability curves
    figures/failure_breakdown.png      -- same as PNG
    results/failure_breakdown.json     -- summary numbers for the paper

Stability for our framework is 100% by construction; this analysis is what
makes that visible in the data.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOGS_DIR = Path(__file__).resolve().parents[2] / "logs"
FIG_DIR = Path(__file__).resolve().parents[2] / "figures"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

COLORS = {
    "fixed": "#5B8DB8",
    "naive": "#E8853D",
    "ours": "#4DAF4A",
}
LABELS = {
    "fixed": "Fixed",
    "naive": "Naive TextGrad",
    "ours": "Ours (separated)",
}


def iter_jsonl(path: Path) -> Iterable[dict]:
    with open(path) as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def parse_log_name(stem: str) -> tuple[str, str, str] | None:
    """Filenames look like:  <task>_<method>_s<seed>  or  <method>_s<seed>."""
    m = re.match(r"^(.*)_(fixed|naive|ours|no_validation|editable_control|freeze)_s(\d+)$", stem)
    if m:
        return m.group(1), m.group(2), m.group(3)
    m = re.match(r"^(fixed|naive|ours|no_validation|editable_control|freeze)_s(\d+)$", stem)
    if m:
        return "_all", m.group(1), m.group(2)
    return None


def load_iter_stabilities(experiment: str) -> dict[str, dict[str, list[list[float]]]]:
    """Return ``{method: {task: [list of per-iter stability series, one per seed]}}``."""
    base = LOGS_DIR / experiment
    if not base.exists():
        return {}

    out: dict[str, dict[str, list[list[float]]]] = defaultdict(lambda: defaultdict(list))
    for log in sorted(base.glob("*.jsonl")):
        parsed = parse_log_name(log.stem)
        if parsed is None:
            continue
        task, method, seed = parsed

        # A log file may contain multiple appended runs (we sometimes re-ran).
        # Detect runs by noticing when the iteration counter resets to 0; keep
        # only the LAST run's per-iter stability series.
        runs: list[list[float]] = []
        current: list[float] = []
        last_iter = -1
        for ev in iter_jsonl(log):
            if ev.get("event") != "metrics":
                continue
            stab = ev.get("stability")
            it = ev.get("iteration", 0)
            if stab is None:
                continue
            if it <= last_iter and current:
                runs.append(current)
                current = []
            current.append(float(stab))
            last_iter = it
        if current:
            runs.append(current)
        if runs:
            out[method][task].append(runs[-1])
    return out


def aggregate_method_stability(per_method: dict[str, dict[str, list[list[float]]]]) -> dict[str, dict]:
    """Compute mean stability per method (final iter) and the averaged curve."""
    out: dict[str, dict] = {}
    for method, by_task in per_method.items():
        all_series: list[list[float]] = []
        finals = []
        for task, runs in by_task.items():
            for s in runs:
                all_series.append(s)
                finals.append(s[-1])
        if not all_series:
            continue

        # Pad each series to the longest length by repeating the last value
        max_len = max(len(s) for s in all_series)
        padded = [s + [s[-1]] * (max_len - len(s)) for s in all_series]
        arr = np.array(padded)
        out[method] = {
            "n_runs": len(all_series),
            "n_iters": int(max_len),
            "final_stability_mean": round(float(np.mean(finals)), 4),
            "final_stability_std": round(float(np.std(finals)), 4),
            "iter_mean": [round(float(x), 4) for x in arr.mean(axis=0)],
            "iter_std": [round(float(x), 4) for x in arr.std(axis=0)],
        }
    return out


def plot_stability_curves(summaries: dict[str, dict[str, dict]], output_path: Path) -> None:
    experiments = [e for e, s in summaries.items() if any(d.get("n_iters", 0) > 1 for d in s.values())]
    if not experiments:
        # Single-iteration experiments: collapse to bar chart of final stability
        plot_stability_bar(summaries, output_path)
        return

    n = len(experiments)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, exp in zip(axes, experiments):
        for method in ("fixed", "naive", "ours"):
            d = summaries[exp].get(method)
            if not d:
                continue
            mean = np.array(d["iter_mean"]) * 100
            std = np.array(d["iter_std"]) * 100
            x = np.arange(len(mean))
            if d["n_iters"] == 1:
                ax.axhline(y=mean[0], color=COLORS[method], linestyle=":", linewidth=1.2,
                            label=LABELS[method])
            else:
                ax.plot(x, mean, color=COLORS[method], linewidth=1.5, label=LABELS[method])
                ax.fill_between(x, np.maximum(mean - std, 0),
                                np.minimum(mean + std, 100), alpha=0.15, color=COLORS[method])
        ax.set_title(exp.capitalize(), fontweight="bold")
        ax.set_xlabel("Optimization iteration")
        ax.set_ylim(-3, 105)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    axes[0].set_ylabel("Episode stability (%)")
    axes[-1].legend(loc="lower right", fontsize=8, framealpha=0.95, edgecolor="none")

    plt.tight_layout(w_pad=0.7)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def plot_stability_bar(summaries: dict[str, dict[str, dict]], output_path: Path) -> None:
    """Fallback: simple bar chart of final stability per (experiment, method)."""
    experiments = list(summaries.keys())
    methods = ("fixed", "naive", "ours")

    fig, ax = plt.subplots(figsize=(1.6 + 0.8 * len(experiments) * len(methods), 2.5))
    width = 0.25
    x = np.arange(len(experiments))
    for i, m in enumerate(methods):
        vals = [summaries[exp].get(m, {}).get("final_stability_mean", 0) * 100 for exp in experiments]
        bars = ax.bar(x + i * width, vals, width, color=COLORS[m], label=LABELS[m])
        for b, v in zip(bars, vals):
            if v > 0:
                ax.text(b.get_x() + width / 2, v + 1, f"{v:.0f}%",
                         ha="center", fontsize=7, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels([e.capitalize() for e in experiments])
    ax.set_ylabel("Final stability (%)")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8, framealpha=0.95, edgecolor="none")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True, parents=True)
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    summaries: dict[str, dict[str, dict]] = {}
    for exp in ("synthetic", "bbh", "insurance", "review"):
        per_method = load_iter_stabilities(exp)
        if not per_method:
            continue
        summaries[exp] = aggregate_method_stability(per_method)

    print("\n=== STABILITY-FAILURE BREAKDOWN ===\n")
    print(f"{'Experiment':<12} {'Method':<10} {'#runs':>6} {'iters':>6} "
          f"{'final stab (mean +/- std)':>30}")
    print("-" * 70)
    for exp, by_method in summaries.items():
        for method in ("fixed", "naive", "ours"):
            d = by_method.get(method)
            if not d:
                continue
            print(
                f"{exp:<12} {method:<10} {d['n_runs']:>6} {d['n_iters']:>6}   "
                f"{d['final_stability_mean'] * 100:>6.1f}% +/- {d['final_stability_std'] * 100:>5.1f}%"
            )
        print()

    if summaries:
        plot_stability_curves(summaries, FIG_DIR / "failure_breakdown.pdf")
        plot_stability_curves(summaries, FIG_DIR / "failure_breakdown.png")

    out_path = RESULTS_DIR / "failure_breakdown.json"
    with open(out_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
