"""MARG-style atomic-comment alignment metrics.

This is a simplified faithful re-implementation of the alignment evaluation
described in Section 5 of D'Arcy et al. 2024 ("MARG: Multi-Agent Review
Generation for Scientific Papers"). Three steps:

1. **Comment extraction**: split each human review into atomic comments via
   an LLM call.
2. **Pairwise alignment**: for every (predicted_comment, reference_comment)
   pair, ask the LLM to score relatedness in {none, weak, medium, high} and
   relative specificity in {less, same, more}.
3. **Metrics**: directional intersections give Recall, Precision, Pseudo-Jaccard
   (see equations in MARG Sec.~5).

Notes
-----
Unlike MARG, we use ``gpt-5.4-nano`` for both extraction and alignment instead
of GPT-4. Numbers are therefore not bit-for-bit comparable to their reported
values; the extraction/alignment prompts are taken (lightly adapted) from their
released ``align_config.json``.
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterable

from cdsep.llm import LLMClient

# Parallelism for the pairwise alignment loop. Override with MARG_ALIGN_WORKERS.
ALIGN_WORKERS = int(os.environ.get("MARG_ALIGN_WORKERS", "64"))


# ---------------------------------------------------------------------------
# Prompts adapted from MARG's released align_config.json
# ---------------------------------------------------------------------------

EXTRACT_SYSTEM = """\
A user will give you a scientific paper review, and you must produce a list
of comments made by the reviewer. Each item should stand alone as a complete
comment, paraphrased only as needed to add context. Merge similar comments
together. Output a JSON object: {"comments": List[str]}. Focus on
SUBSTANTIVE comments (impact, correctness, clarity); skip pure praise.
"""

ALIGN_SYSTEM = """\
You compare two scientific review comments. One is the "reference" (from a
real reviewer) and one is the "predicted" comment. Decide whether they are
making essentially the same point, and rate the predicted comment's
specificity relative to the reference.

Output a JSON object on its own line:
{
  "relatedness": one of "none" | "weak" | "medium" | "high",
  "relative_specificity": one of "less" | "same" | "more" | null
}

Use "none" if the comments are about different things; "weak" if they share
a topic but not the same request; "medium" if an edit addressing one would
likely address the other; "high" if they're nearly the same comment.
"""

# A predicted/reference pair counts as "aligned" if relatedness >= this.
DEFAULT_ALIGN_THRESHOLD = "medium"

_RELATEDNESS_RANK = {"none": 0, "weak": 1, "medium": 2, "high": 3, "unknown": 0}


def _extract_json_object(text: str) -> dict | None:
    """Find the last well-formed JSON object in a model response."""
    text = text.strip()
    # Try fenced first
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Find LAST {...} balanced block
    candidates = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                candidates.append(text[start : i + 1])
                start = -1
    for s in reversed(candidates):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            continue
    return None


def extract_atomic_comments(review_text: str, llm: LLMClient) -> list[str]:
    """Use the LLM to split a review into a list of atomic comments."""
    if not review_text.strip():
        return []
    msgs = [
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": f"Review:\n\n{review_text}"},
    ]
    raw = llm.chat(msgs, temperature=0)
    obj = _extract_json_object(raw) or {}
    comments = obj.get("comments") or []
    return [c.strip() for c in comments if isinstance(c, str) and c.strip()]


def extract_reference_comments(
    reviews: Iterable[str], llm: LLMClient, max_reviews: int = 2
) -> list[str]:
    """Concatenate atomic comments extracted from human reviews.

    For cost control we use only the first ``max_reviews`` reviews per paper.
    Parallelised across reviews via ``ALIGN_WORKERS``.
    """
    reviews = list(reviews)[:max_reviews]
    if not reviews:
        return []
    workers = max(1, min(ALIGN_WORKERS, len(reviews)))
    if workers <= 1:
        lists = [extract_atomic_comments(r, llm) for r in reviews]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            lists = list(ex.map(lambda r: extract_atomic_comments(r, llm), reviews))
    out: list[str] = []
    for cs in lists:
        out.extend(cs)
    return out


def _align_pair(
    predicted: str, reference: str, llm: LLMClient
) -> tuple[str, str | None]:
    msgs = [
        {"role": "system", "content": ALIGN_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Reference comment:\n{reference}\n\n"
                f"Predicted comment:\n{predicted}"
            ),
        },
    ]
    raw = llm.chat(msgs, temperature=0)
    obj = _extract_json_object(raw) or {}
    rel = str(obj.get("relatedness", "none")).strip().lower()
    spec = obj.get("relative_specificity")
    if spec is not None:
        spec = str(spec).strip().lower()
    return rel, spec


@dataclass
class AlignmentResult:
    recall: float
    precision: float
    jaccard: float
    n_predicted: int
    n_reference: int
    n_predicted_aligned: int
    n_reference_aligned: int
    alignment_matrix: list[list[bool]] | None = None  # matrix[i][j] = pred_i aligned to ref_j


def alignment_metrics(
    predicted: list[str],
    reference: list[str],
    llm: LLMClient,
    threshold: str = DEFAULT_ALIGN_THRESHOLD,
    max_pred: int = 6,
    max_ref: int = 8,
) -> AlignmentResult:
    """Run pairwise alignment and compute Recall / Precision / Pseudo-Jaccard.

    For cost control, ``max_pred`` / ``max_ref`` cap the number of comments
    used in the all-pairs comparison; excess comments are truncated. The
    underlying counts (``n_predicted`` / ``n_reference``) reflect the
    truncated lists so that Recall and Precision are well-defined.
    """
    predicted = predicted[:max_pred]
    reference = reference[:max_ref]
    n_pred = len(predicted)
    n_ref = len(reference)
    if n_pred == 0 or n_ref == 0:
        return AlignmentResult(
            recall=0.0,
            precision=0.0,
            jaccard=0.0,
            n_predicted=n_pred,
            n_reference=n_ref,
            n_predicted_aligned=0,
            n_reference_aligned=0,
        )

    threshold_rank = _RELATEDNESS_RANK[threshold]
    pred_aligned: set[int] = set()
    ref_aligned: set[int] = set()
    matrix: list[list[bool]] = [[False] * n_ref for _ in range(n_pred)]

    pairs = [(i, j) for i in range(n_pred) for j in range(n_ref)]

    def _score(ij: tuple[int, int]) -> tuple[int, int, str]:
        i, j = ij
        rel, _ = _align_pair(predicted[i], reference[j], llm)
        return i, j, rel

    workers = max(1, min(ALIGN_WORKERS, len(pairs)))
    if workers <= 1:
        results = [_score(p) for p in pairs]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_score, pairs))
    for i, j, rel in results:
        if _RELATEDNESS_RANK.get(rel, 0) >= threshold_rank:
            pred_aligned.add(i)
            ref_aligned.add(j)
            matrix[i][j] = True

    recall = len(ref_aligned) / n_ref
    precision = len(pred_aligned) / n_pred
    intersection = (len(pred_aligned) + len(ref_aligned)) / 2
    union = n_pred + n_ref - intersection
    jaccard = intersection / union if union > 0 else 0.0

    return AlignmentResult(
        recall=recall,
        precision=precision,
        jaccard=jaccard,
        n_predicted=n_pred,
        n_reference=n_ref,
        n_predicted_aligned=len(pred_aligned),
        n_reference_aligned=len(ref_aligned),
        alignment_matrix=matrix,
    )


def compute_metrics_marg(
    system_comments: list[str],
    human_reviews: list[str],
    llm: LLMClient,
) -> dict[str, float]:
    """Adapter compatible with the existing review-experiment evaluate loop."""
    reference = extract_reference_comments(human_reviews, llm)
    res = alignment_metrics(system_comments, reference, llm)
    return {
        "recall": round(res.recall, 4),
        "precision": round(res.precision, 4),
        "jaccard": round(res.jaccard, 4),
        "n_comments": float(res.n_predicted),
        "n_reference": float(res.n_reference),
    }
