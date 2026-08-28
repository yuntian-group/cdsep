"""Smoke tests for BBH experiment pipeline."""

from unittest.mock import MagicMock

from experiments.bbh.agents import make_bbh_agent, make_control_schema
from experiments.bbh.data import is_correct, task_answer_space
from cdsep.episode import run_single_agent_episode
from cdsep.llm import LLMClient


def test_is_correct_choice():
    assert is_correct("A", "A", "choice")
    assert is_correct("(A)", "A", "choice")
    assert not is_correct("B", "A", "choice")


def test_is_correct_yesno():
    assert is_correct("Yes", "Yes", "yesno")
    assert is_correct("yes, definitely", "Yes", "yesno")
    assert not is_correct("Yes", "No", "yesno")


def test_is_correct_freeform():
    assert is_correct("apple banana", "apple banana", "freeform")
    assert is_correct("Apple Banana", "apple banana", "freeform")
    assert not is_correct("apple", "apple banana", "freeform")


def test_task_answer_space_choice():
    examples = [{"input": "Q.\nOptions:\n(A) one\n(B) two\n(C) three", "answer": "A"}]
    kind, opts = task_answer_space("logical_deduction_three_objects", examples)
    assert kind == "choice"
    assert opts == ["A", "B", "C"]


def test_task_answer_space_yesno():
    kind, opts = task_answer_space("causal_judgement", [])
    assert kind == "yesno"


def test_make_control_schema_choice():
    meta = {"task": "demo", "kind": "choice", "options": ["A", "B"]}
    cls = make_control_schema(meta)
    instance = cls(answer="A")
    assert instance.answer == "A"


def test_agent_with_mocked_llm():
    meta = {"task": "demo", "kind": "choice", "options": ["A", "B", "C"]}
    agent = make_bbh_agent("Q: example\nA: A", meta, separated=True)

    llm = MagicMock(spec=LLMClient)
    llm.chat = MagicMock(return_value='{"answer": "B"}\nReasoning here.')

    trace = run_single_agent_episode(agent, "Q: 2+2=?", llm)
    assert trace.outcome == "completed"
    assert trace.steps[0].control.answer == "B"
