"""Prompt-drift analysis from ``optimization_step`` events in the JSONL logs.

For each (experiment, method) we compare the iteration-0 prompt with the
iteration-K prompt and measure:

* Total Levenshtein-style edit distance (chars added + removed via difflib).
* Whether the edit *touched* any control-related substrings: JSON, schema
  field names, the control-block instructions, etc.
* Drift over iterations: ratio of the prompt that has been replaced by step k.

The headline output is a small LaTeX table summarising how aggressively each
method's optimizer rewrites prompts and how much of that rewriting touches
the control surface (which is what causes naive collapse on review).
"""

from __future__ import annotations

import difflib
import json
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

# Substrings that touching = "control-relevant".
CONTROL_KEYWORDS = (
    "json",
    "JSON",
    "control block",
    "control_block",
    "control schema",
    "schema",
    "answer",
    "action",
    "target_agent",
    "preliminary_rating",
    "final_rating",
    "stop",
    "send",
    "Output format",
    "Example: {",
    '"',
    "{",
    "}",
)

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
    m = re.match(r"^(.*)_(fixed|naive|ours|no_validation|editable_control|freeze)_s(\d+)$", stem)
    if m:
        return m.group(1), m.group(2), m.group(3)
    m = re.match(r"^(fixed|naive|ours|no_validation|editable_control|freeze)_s(\d+)$", stem)
    if m:
        return "_all", m.group(1), m.group(2)
    return None


def edit_size(a: str, b: str) -> int:
    """Total chars added + removed in turning a into b."""
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    inserted = removed = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace":
            removed += i2 - i1
            inserted += j2 - j1
        elif tag == "delete":
            removed += i2 - i1
        elif tag == "insert":
            inserted += j2 - j1
    return inserted + removed


def line_edits(a: str, b: str) -> tuple[list[str], list[str]]:
    """Return (added_lines, removed_lines) at the line level."""
    a_lines = a.splitlines(keepends=False)
    b_lines = b.splitlines(keepends=False)
    diff = difflib.ndiff(a_lines, b_lines)
    added, removed = [], []
    for line in diff:
        if line.startswith("+ "):
            added.append(line[2:])
        elif line.startswith("- "):
            removed.append(line[2:])
    return added, removed


def line_touches_control(line: str) -> bool:
    return any(kw in line for kw in CONTROL_KEYWORDS)


def fraction_control_touching(a: str, b: str) -> float:
    added, removed = line_edits(a, b)
    touched = sum(1 for l in added + removed if line_touches_control(l))
    total = len(added) + len(removed)
    return touched / total if total > 0 else 0.0


def collect_optimization_runs(experiment: str) -> dict[str, list[dict]]:
    """Return ``{method: [run_summary, ...]}`` for an experiment.

    A run_summary is a dict with: agent, prompts (list of (k, prompt)
    chronologically, dedup'd by appended-runs heuristic), final_edit_chars,
    final_edit_lines, control_touch_fraction.
    """
    base = LOGS_DIR / experiment
    if not base.exists():
        return {}

    out: dict[str, list[dict]] = defaultdict(list)
    for log in sorted(base.glob("*.jsonl")):
        parsed = parse_log_name(log.stem)
        if parsed is None:
            continue
        task, method, seed = parsed

        # Group optimization steps by (run, agent). Detect new runs by
        # iteration counter resetting.
        runs_by_agent: dict[str, list[list[dict]]] = defaultdict(list)
        current_by_agent: dict[str, list[dict]] = defaultdict(list)
        last_iter_by_agent: dict[str, int] = defaultdict(lambda: -1)
        for ev in iter_jsonl(log):
            if ev.get("event") != "optimization_step":
                continue
            agent = ev.get("agent", "agent")
            it = ev.get("iteration", 0)
            if it <= last_iter_by_agent[agent] and current_by_agent[agent]:
                runs_by_agent[agent].append(current_by_agent[agent])
                current_by_agent[agent] = []
            current_by_agent[agent].append(ev)
            last_iter_by_agent[agent] = it
        for agent, cur in current_by_agent.items():
            if cur:
                runs_by_agent[agent].append(cur)

        for agent, runs in runs_by_agent.items():
            if not runs:
                continue
            steps = runs[-1]
            initial_prompt = steps[0].get("old_prompt", "")
            final_prompt = steps[-1].get("new_prompt", "")
            if not initial_prompt or not final_prompt:
                continue
            edit_chars = edit_size(initial_prompt, final_prompt)
            added, removed = line_edits(initial_prompt, final_prompt)
            edit_lines = len(added) + len(removed)
            ctrl_frac = fraction_control_touching(initial_prompt, final_prompt)

            out[method].append({
                "experiment": experiment,
                "task": task,
                "seed": seed,
                "agent": agent,
                "n_iters": len(steps),
                "initial_prompt_len": len(initial_prompt),
                "final_prompt_len": len(final_prompt),
                "edit_chars": edit_chars,
                "edit_lines_added": len(added),
                "edit_lines_removed": len(removed),
                "control_touch_fraction": ctrl_frac,
            })
    return out


def summarize(per_method: dict[str, list[dict]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for method, runs in per_method.items():
        if not runs:
            continue
        ec = np.array([r["edit_chars"] for r in runs])
        el = np.array([r["edit_lines_added"] + r["edit_lines_removed"] for r in runs])
        cf = np.array([r["control_touch_fraction"] for r in runs])
        ipl = np.array([r["initial_prompt_len"] for r in runs])
        replaced = ec / np.maximum(ipl * 2, 1)
        out[method] = {
            "n_runs": len(runs),
            "mean_edit_chars": round(float(ec.mean()), 1),
            "mean_edit_lines": round(float(el.mean()), 1),
            "mean_replaced_fraction": round(float(replaced.mean()), 3),
            "mean_control_touch_fraction": round(float(cf.mean()), 3),
        }
    return out


def render_tex_table(summaries: dict[str, dict[str, dict[str, float]]]) -> str:
    """A LaTeX table to drop into the paper as tab:prompt-edits."""
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Experiment & Method & \#runs & Edit (chars) & Edit (lines) & Control-touch \% \\",
        r"\midrule",
    ]
    for exp, by_method in summaries.items():
        first = True
        for method in ("naive", "ours"):
            d = by_method.get(method)
            if not d:
                continue
            exp_label = exp.capitalize() if first else ""
            lines.append(
                f"{exp_label} & {LABELS[method]} & {int(d['n_runs'])} & "
                f"{d['mean_edit_chars']:.0f} & {d['mean_edit_lines']:.1f} & "
                f"{d['mean_control_touch_fraction'] * 100:.1f}\\% \\\\"
            )
            first = False
        lines.append(r"\midrule")
    lines = lines[:-1]  # drop trailing midrule
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def plot_drift(summaries: dict[str, dict[str, dict[str, float]]], output_path: Path) -> None:
    experiments = list(summaries.keys())
    methods = ("naive", "ours")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.0, 2.4))

    n = len(experiments)
    x = np.arange(n)
    width = 0.35

    for i, m in enumerate(methods):
        edits = [summaries[exp].get(m, {}).get("mean_edit_lines", 0) for exp in experiments]
        ax1.bar(x + i * width, edits, width, color=COLORS[m], label=LABELS[m])
    ax1.set_xticks(x + width / 2)
    ax1.set_xticklabels([e.capitalize() for e in experiments], fontsize=8)
    ax1.set_ylabel("Lines edited")
    ax1.set_title("Prompt-edit volume", fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.legend(fontsize=8, framealpha=0.95, edgecolor="none")

    for i, m in enumerate(methods):
        ctrl = [summaries[exp].get(m, {}).get("mean_control_touch_fraction", 0) * 100 for exp in experiments]
        bars = ax2.bar(x + i * width, ctrl, width, color=COLORS[m], label=LABELS[m])
        for b, v in zip(bars, ctrl):
            if v > 0:
                ax2.text(b.get_x() + width / 2, v + 1, f"{v:.0f}%", ha="center", fontsize=7)
    ax2.set_xticks(x + width / 2)
    ax2.set_xticklabels([e.capitalize() for e in experiments], fontsize=8)
    ax2.set_ylabel("Control-touching edits (%)")
    ax2.set_title("Drift onto control surface", fontweight="bold")
    ax2.set_ylim(0, max(50, max([max([summaries[e].get(m, {}).get("mean_control_touch_fraction", 0) * 100
                                        for m in methods]) for e in experiments]) * 1.4))
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout(w_pad=1.0)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True, parents=True)
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    summaries: dict[str, dict[str, dict[str, float]]] = {}
    for exp in ("synthetic", "bbh", "insurance", "review"):
        per_method = collect_optimization_runs(exp)
        if not per_method:
            continue
        summaries[exp] = summarize(per_method)

    print("\n=== PROMPT DRIFT ANALYSIS ===\n")
    print(f"{'Exp':<12} {'Method':<8} {'#runs':>6} {'edit_ch':>9} {'edit_ln':>9} {'replaced':>9} {'ctrl_touch':>11}")
    print("-" * 65)
    for exp, by_method in summaries.items():
        for method in ("naive", "ours"):
            d = by_method.get(method)
            if not d:
                continue
            print(
                f"{exp:<12} {method:<8} {int(d['n_runs']):>6} "
                f"{d['mean_edit_chars']:>8.0f}  {d['mean_edit_lines']:>8.1f}  "
                f"{d['mean_replaced_fraction']:>8.2f}  {d['mean_control_touch_fraction']*100:>9.1f}%"
            )

    if summaries:
        plot_drift(summaries, FIG_DIR / "prompt_drift.pdf")
        plot_drift(summaries, FIG_DIR / "prompt_drift.png")

        tex = render_tex_table(summaries)
        tex_path = RESULTS_DIR / "prompt_edits_table.tex"
        with open(tex_path, "w") as f:
            f.write(tex)
        print(f"\nWrote {tex_path}")

    out_path = RESULTS_DIR / "prompt_drift.json"
    with open(out_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
