"""Load MARG (ARIES) ICLR papers + human reviews for the review-generation experiment.

Data files (in ``data/marg/``):
- ``split_ids.json``        - ARIES train/dev/test split (we use test).
- ``review_replies_test.jsonl`` - one record per OpenReview reply for our 42 test docs.
- ``s2orc/<paper_id>.json`` - parsed paper text for each PDF in the test set.
"""

from __future__ import annotations

import json
import os
import random
from functools import lru_cache
from typing import Iterable

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "marg",
)


@lru_cache(maxsize=1)
def _load_split_ids() -> dict:
    with open(os.path.join(DATA_DIR, "split_ids.json")) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_review_replies_by_doc() -> dict[str, list[dict]]:
    """Group review_replies records by their forum (= doc_id)."""
    by_doc: dict[str, list[dict]] = {}
    path = os.path.join(DATA_DIR, "review_replies_test.jsonl")
    with open(path) as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            forum = obj.get("forum")
            if not forum:
                continue
            by_doc.setdefault(forum, []).append(obj)
    return by_doc


def _is_official_review(rec: dict) -> bool:
    """Filter for top-level reviewer reviews (skip author replies, meta-reviews)."""
    inv = rec.get("invitation", "")
    return "Official_Review" in inv or inv.endswith("/Review")


def _extract_review_text(rec: dict) -> str:
    """Concatenate the relevant text fields of an OpenReview review record."""
    content = rec.get("content", {}) or {}
    parts = []
    for key in (
        "summary_of_the_paper",
        "main_review",
        "review",
        "summary_of_the_review",
        "strength_and_weaknesses",
        "strength_weakness",
        "strengths",
        "weaknesses",
        "questions",
    ):
        val = content.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    if not parts:
        # fallback: anything textual
        for v in content.values():
            if isinstance(v, str) and len(v) > 50:
                parts.append(v.strip())
    return "\n\n".join(parts)


def get_human_reviews(doc_id: str) -> list[str]:
    """Return list of full-text reviews for a paper (typically 3-5)."""
    by_doc = _load_review_replies_by_doc()
    out = []
    for rec in by_doc.get(doc_id, []):
        if not _is_official_review(rec):
            continue
        text = _extract_review_text(rec)
        if text.strip():
            out.append(text)
    return out


@lru_cache(maxsize=128)
def _load_s2orc(paper_id: str) -> dict | None:
    path = os.path.join(DATA_DIR, "s2orc", f"{paper_id}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _section_chunks(s2orc: dict, max_chars_per_section: int = 4000) -> dict[str, str]:
    """Group a paper's body_text by section heading, truncating very long sections."""
    sections: dict[str, list[str]] = {}
    pdf = s2orc.get("pdf_parse", {}) or {}
    for blk in pdf.get("body_text", []):
        sec_name = (blk.get("section") or "Body").strip() or "Body"
        sections.setdefault(sec_name, []).append(blk.get("text", ""))
    out: dict[str, str] = {}
    for name, texts in sections.items():
        joined = " ".join(t for t in texts if t)
        if len(joined) > max_chars_per_section:
            joined = joined[:max_chars_per_section] + " [...truncated]"
        out[name] = joined
    return out


def get_paper(doc_id: str) -> dict | None:
    """Return {title, abstract, sections: {name: text}, human_reviews: [...]}.

    Picks the source PDF (pre-rebuttal version) for the doc.
    """
    split = _load_split_ids()
    pdf_id = None
    for r in split["test"]:
        if r["doc_id"] == doc_id:
            pdf_id = r["source_pdf_id"]
            break
    if pdf_id is None:
        return None
    s2orc = _load_s2orc(pdf_id)
    if s2orc is None:
        return None
    return {
        "doc_id": doc_id,
        "title": s2orc.get("title", ""),
        "abstract": s2orc.get("abstract", ""),
        "sections": _section_chunks(s2orc),
        "human_reviews": get_human_reviews(doc_id),
    }


def list_test_doc_ids() -> list[str]:
    """All test doc_ids that have both a PDF and at least one official review."""
    out = []
    for r in _load_split_ids()["test"]:
        did = r["doc_id"]
        pdf_id = r["source_pdf_id"]
        if not os.path.exists(os.path.join(DATA_DIR, "s2orc", f"{pdf_id}.json")):
            continue
        if not get_human_reviews(did):
            continue
        out.append(did)
    return out


def load_papers(seed: int = 42, limit: int | None = None) -> list[dict]:
    """Load all eligible papers, shuffled by seed."""
    rng = random.Random(seed)
    doc_ids = list_test_doc_ids()
    rng.shuffle(doc_ids)
    if limit is not None:
        doc_ids = doc_ids[:limit]
    out: list[dict] = []
    for did in doc_ids:
        p = get_paper(did)
        if p is not None and p["sections"] and p["human_reviews"]:
            out.append(p)
    return out


def split_papers(papers: list[dict], n_train: int) -> tuple[list[dict], list[dict]]:
    return papers[:n_train], papers[n_train:]
