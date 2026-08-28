"""Metrics for collaborative review generation: Recall, Precision, Jaccard over atomic comments."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cdsep.llm import LLMClient

MATCH_PROMPT = """\
You are evaluating whether two review comments are semantically similar (covering the same point).

Comment A: {comment_a}
Comment B: {comment_b}

Are these two comments making the same or very similar point about the paper? 
Answer with only "yes" or "no"."""


def compute_match_matrix(
    system_comments: list[str],
    reference_comments: list[str],
    llm: LLMClient,
) -> list[list[bool]]:
    """Compute pairwise semantic match matrix using LLM-as-judge.

    Returns matrix[i][j] = True if system_comments[i] matches reference_comments[j].
    """
    matrix = []
    for sc in system_comments:
        row = []
        for rc in reference_comments:
            prompt = MATCH_PROMPT.format(comment_a=sc.strip(), comment_b=rc.strip())
            resp = llm.chat([{"role": "user", "content": prompt}], temperature=0.0)
            row.append(resp.strip().lower().startswith("yes"))
        matrix.append(row)
    return matrix


def compute_metrics(
    system_comments: list[str],
    reference_comments: list[str],
    llm: LLMClient,
) -> dict[str, float]:
    """Compute Recall, Precision, Jaccard over atomic comment sets."""
    if not system_comments or not reference_comments:
        return {"recall": 0.0, "precision": 0.0, "jaccard": 0.0, "n_comments": len(system_comments)}

    matrix = compute_match_matrix(system_comments, reference_comments, llm)

    ref_matched = set()
    sys_matched = set()

    for i, row in enumerate(matrix):
        for j, matched in enumerate(row):
            if matched:
                ref_matched.add(j)
                sys_matched.add(i)

    recall = len(ref_matched) / len(reference_comments) if reference_comments else 0
    precision = len(sys_matched) / len(system_comments) if system_comments else 0

    union_size = len(system_comments) + len(reference_comments) - len(ref_matched)
    jaccard = len(ref_matched) / union_size if union_size > 0 else 0

    return {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "jaccard": round(jaccard, 4),
        "n_comments": len(system_comments),
    }
