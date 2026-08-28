"""Smoke tests for review experiment pipeline."""

import os
import pytest

from experiments.review.agents import LeaderControl, WorkerControl, make_review_agents
from experiments.review.marg_data import DATA_DIR, list_test_doc_ids, split_papers
from experiments.review.run import extract_comments_from_message, route_review


HAS_MARG_DATA = os.path.exists(os.path.join(DATA_DIR, "split_ids.json"))


@pytest.mark.skipif(not HAS_MARG_DATA, reason="MARG/ARIES data not downloaded")
class TestReviewData:
    def test_eligible_docs(self):
        ids = list_test_doc_ids()
        assert len(ids) > 0

    def test_split(self):
        papers = [{"i": i} for i in range(10)]
        train, test = split_papers(papers, 6)
        assert len(train) == 6
        assert len(test) == 4


class TestReviewRouting:
    def test_leader_send(self):
        ctrl = LeaderControl(action="send", target_agent="worker_1", stop=False)
        assert route_review(ctrl) == "worker_1"

    def test_leader_stop(self):
        ctrl = LeaderControl(action="stop", target_agent="none", stop=True)
        assert route_review(ctrl) == "terminate"

    def test_worker_returns_to_leader(self):
        ctrl = WorkerControl(status="done", section="introduction")
        assert route_review(ctrl) == "leader"


class TestCommentExtraction:
    def test_numbered_list(self):
        msg = "1. The method is novel.\n2. Missing baselines.\n3. Writing is clear."
        comments = extract_comments_from_message(msg)
        assert len(comments) == 3
        assert "novel" in comments[0]

    def test_bullet_list(self):
        msg = "- The approach is well-motivated.\n- Experiments are thorough.\n- Minor typos exist."
        comments = extract_comments_from_message(msg)
        assert len(comments) == 3

    def test_agents_created(self):
        agents = make_review_agents()
        assert "leader" in agents
        assert "worker_1" in agents
        assert "worker_2" in agents
        assert "worker_3" in agents
