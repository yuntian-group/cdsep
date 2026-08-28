"""Pull experiment results from JSON files and replace the \\TODO placeholders
in the paper's experiments and summary tables with the actual numbers.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = ROOT / "sections"
RESULTS = ROOT / "results"

EXPERIMENTS_TEX = SECTIONS / "experiments.tex"


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def safe_pct(x: float | int) -> str:
    return f"{x * 100:.1f}"


def update_block(text: str, marker_lines: list[str], replacements: list[str]) -> str:
    """Replace a contiguous block of lines.

    marker_lines = the EXACT lines we expect to find (preserving indentation
    is up to the caller). replacements is the same length list.
    """
    block = "\n".join(marker_lines)
    new_block = "\n".join(replacements)
    if block not in text:
        print(f"WARNING: block not found:\n{block!r}", file=sys.stderr)
        return text
    return text.replace(block, new_block)


def fill_bbh(text: str) -> str:
    res = load_json(RESULTS / "bbh" / "results.json")
    if not res:
        print("BBH results not yet available.", file=sys.stderr)
        return text

    tasks = [
        "logical_deduction_three_objects",
        "tracking_shuffled_objects_three_objects",
        "causal_judgement",
        "word_sorting",
    ]

    def get(method: str, key: str = "mean_accuracy") -> str:
        all_v = []
        for t in tasks:
            d = res.get(t, {}).get(method)
            if not d:
                return r"\TODO"
            all_v.append(d[key])
        return ""  # noop sentinel

    def cell(method: str, task: str, key: str = "mean_accuracy") -> str:
        d = res.get(task, {}).get(method)
        if not d:
            return r"\TODO"
        return safe_pct(d[key])

    # Order matters: rows are Fixed, Naive, Ours
    new_rows = []
    for method, prefix, suffix in (
        ("fixed", "Fixed prompts       ", "\\\\"),
        ("naive", "Na\"{\\i}ve TextGrad ", "\\\\"),
        ("ours",  "\\textbf{Ours}       ", "\\\\"),
    ):
        cells = [cell(method, t) for t in tasks]
        if method == "ours":
            cells = [f"\\textbf{{{c}}}" for c in cells]
        new_rows.append(f"{prefix}& {' & '.join(cells)} {suffix}")

    stab_naive = [cell("naive", t, "mean_stability") for t in tasks]
    stab_ours = [cell("ours", t, "mean_stability") for t in tasks]

    # Convert "100.0" -> "100" etc for stability rows
    def short_pct(s):
        try:
            v = float(s)
            return str(int(round(v)))
        except Exception:
            return s

    stab_naive_str = [short_pct(s) for s in stab_naive]
    stab_ours_str = [short_pct(s) for s in stab_ours]

    block = (
        "Fixed prompts       & \\TODO & \\TODO & \\TODO & \\TODO \\\\\n"
        "Na\\\"{\\i}ve TextGrad & \\TODO & \\TODO & \\TODO & \\TODO \\\\\n"
        "\\textbf{Ours}       & \\textbf{\\TODO} & \\textbf{\\TODO} & \\textbf{\\TODO} & \\textbf{\\TODO} \\\\\n"
        "\\midrule\n"
        "Stability -- Na\\\"{\\i}ve & \\TODO & \\TODO & \\TODO & \\TODO \\\\\n"
        "Stability -- Ours       & \\textbf{100} & \\textbf{100} & \\textbf{100} & \\textbf{100} \\\\"
    )
    new_block = (
        f"{new_rows[0]}\n"
        f"{new_rows[1]}\n"
        f"{new_rows[2]}\n"
        f"\\midrule\n"
        f"Stability -- Na\\\"{{\\i}}ve & {' & '.join(stab_naive_str)} \\\\\n"
        f"Stability -- Ours       & \\textbf{{{stab_ours_str[0]}}} & \\textbf{{{stab_ours_str[1]}}} & \\textbf{{{stab_ours_str[2]}}} & \\textbf{{{stab_ours_str[3]}}} \\\\"
    )
    if block in text:
        text = text.replace(block, new_block)
        print("Filled BBH table.")
    else:
        print("WARN: BBH table marker not found.", file=sys.stderr)
    return text


def fill_insurance(text: str) -> str:
    res = load_json(RESULTS / "insurance" / "results.json")
    if not res:
        print("Insurance results not yet available.", file=sys.stderr)
        return text

    def cell(method: str, key: str) -> str:
        d = res.get(method)
        if not d:
            return r"\TODO"
        v = d.get(f"mean_{key}")
        if v is None:
            return r"\TODO"
        if key == "stability":
            return f"{v * 100:.0f}\\%"
        if key == "mae":
            return f"{v:.3f}"
        return f"{v * 100:.1f}"

    block = (
        "Fixed prompts       & \\TODO & \\TODO & 100\\% \\\\\n"
        "Na\\\"{\\i}ve TextGrad & \\TODO & \\TODO & \\TODO \\\\\n"
        "\\textbf{Ours}       & \\textbf{\\TODO} & \\textbf{\\TODO} & \\textbf{100\\%} \\\\"
    )
    new_block = (
        f"Fixed prompts       & {cell('fixed', 'accuracy')} & {cell('fixed', 'mae')} & {cell('fixed', 'stability')} \\\\\n"
        f"Na\\\"{{\\i}}ve TextGrad & {cell('naive', 'accuracy')} & {cell('naive', 'mae')} & {cell('naive', 'stability')} \\\\\n"
        f"\\textbf{{Ours}}       & \\textbf{{{cell('ours', 'accuracy')}}} & \\textbf{{{cell('ours', 'mae')}}} & \\textbf{{{cell('ours', 'stability')}}} \\\\"
    )
    if block in text:
        text = text.replace(block, new_block)
        print("Filled Insurance table.")
    else:
        print("WARN: Insurance table marker not found.", file=sys.stderr)
    return text


def fill_review(text: str) -> str:
    res = load_json(RESULTS / "review" / "results.json")
    if not res:
        print("Review results not yet available.", file=sys.stderr)
        return text

    def cell(method: str, key: str) -> str:
        d = res.get(method)
        if not d:
            return r"\TODO"
        v = d.get(f"mean_{key}")
        if v is None:
            return r"\TODO"
        if key == "n_comments":
            return f"{v:.1f}"
        return f"{v * 100:.1f}"

    block = (
        "Fixed prompts       & \\TODO & \\TODO & \\TODO & \\TODO \\\\\n"
        "Na\\\"{\\i}ve TextGrad & \\TODO & \\TODO & \\TODO & \\TODO \\\\\n"
        "\\textbf{Ours}       & \\textbf{\\TODO} & \\textbf{\\TODO} & \\textbf{\\TODO} & \\TODO \\\\"
    )
    new_block = (
        f"Fixed prompts       & {cell('fixed', 'recall')} & {cell('fixed', 'precision')} & {cell('fixed', 'jaccard')} & {cell('fixed', 'n_comments')} \\\\\n"
        f"Na\\\"{{\\i}}ve TextGrad & {cell('naive', 'recall')} & {cell('naive', 'precision')} & {cell('naive', 'jaccard')} & {cell('naive', 'n_comments')} \\\\\n"
        f"\\textbf{{Ours}}       & \\textbf{{{cell('ours', 'recall')}}} & \\textbf{{{cell('ours', 'precision')}}} & \\textbf{{{cell('ours', 'jaccard')}}} & {cell('ours', 'n_comments')} \\\\"
    )
    if block in text:
        text = text.replace(block, new_block)
        print("Filled Review table.")
    else:
        print("WARN: Review table marker not found.", file=sys.stderr)
    return text


def fill_summary(text: str) -> str:
    bbh = load_json(RESULTS / "bbh" / "results.json")
    insurance = load_json(RESULTS / "insurance" / "results.json")
    review = load_json(RESULTS / "review" / "results.json")

    def bbh_avg(method: str, key: str = "mean_accuracy") -> str:
        if not bbh:
            return r"\TODO"
        vals = []
        for t, by_method in bbh.items():
            d = by_method.get(method)
            if d and key in d:
                vals.append(d[key])
        if not vals:
            return r"\TODO"
        return f"{sum(vals) / len(vals) * 100:.1f}"

    def ins(method: str, key: str = "mean_accuracy") -> str:
        if not insurance:
            return r"\TODO"
        d = insurance.get(method)
        if not d or key not in d:
            return r"\TODO"
        return f"{d[key] * 100:.1f}"

    def rev(method: str, key: str = "mean_jaccard") -> str:
        if not review:
            return r"\TODO"
        d = review.get(method)
        if not d or key not in d:
            return r"\TODO"
        return f"{d[key] * 100:.1f}"

    def fmt_stab(v: float | None) -> str:
        if v is None:
            return r"\TODO"
        return f"{v * 100:.0f}\\%"

    rev_naive_stab = fmt_stab(review.get("naive", {}).get("mean_stability") if review else None)
    rev_ours_stab = fmt_stab(review.get("ours", {}).get("mean_stability") if review else None)
    ins_naive_stab = fmt_stab(insurance.get("naive", {}).get("mean_stability") if insurance else None)

    block = (
        "Fixed prompts       & \\TODO & 100\\% & \\TODO & 100\\% & \\TODO & 100\\% \\\\\n"
        "Na\\\"{\\i}ve TextGrad & \\TODO & 100\\%  & \\TODO & \\TODO  & \\TODO & \\TODO  \\\\\n"
        "\\textbf{Ours}       & \\textbf{\\TODO} & \\textbf{100\\%} & \\textbf{\\TODO} & \\textbf{100\\%} & \\textbf{\\TODO} & \\textbf{100\\%} \\\\"
    )
    new_block = (
        f"Fixed prompts       & {bbh_avg('fixed')} & 100\\% & {rev('fixed')} & 100\\% & {ins('fixed')} & 100\\% \\\\\n"
        f"Na\\\"{{\\i}}ve TextGrad & {bbh_avg('naive')} & 100\\%  & {rev('naive')} & {rev_naive_stab}  & {ins('naive')} & {ins_naive_stab}  \\\\\n"
        f"\\textbf{{Ours}}       & \\textbf{{{bbh_avg('ours')}}} & \\textbf{{100\\%}} & \\textbf{{{rev('ours')}}} & \\textbf{{{rev_ours_stab}}} & \\textbf{{{ins('ours')}}} & \\textbf{{100\\%}} \\\\"
    )
    if block in text:
        text = text.replace(block, new_block)
        print("Filled Summary table.")
    else:
        print("WARN: Summary table marker not found.", file=sys.stderr)
    return text


def main() -> None:
    text = EXPERIMENTS_TEX.read_text()
    text = fill_bbh(text)
    text = fill_insurance(text)
    text = fill_review(text)
    text = fill_summary(text)
    EXPERIMENTS_TEX.write_text(text)


if __name__ == "__main__":
    main()
