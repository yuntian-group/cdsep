"""Smoke tests for synthetic experiment pipeline (mocked LLM)."""

from dataclasses import dataclass
from unittest.mock import MagicMock

from experiments.synthetic.agents import SyntheticControl, make_synthetic_agent
from experiments.synthetic.data import format_examples_as_table, format_query, generate_dataset
from cdsep.agent import Agent, AgentOutput
from cdsep.episode import run_single_agent_episode
from cdsep.llm import LLMClient


class TestSyntheticData:
    def test_generate_max(self):
        train, test = generate_dataset("Max", n_train=10, n_test=5, seed=0)
        assert len(train) == 10
        assert len(test) == 5
        for ex in train + test:
            assert ex["answer"] == max(ex["a"], ex["b"])

    def test_generate_modsum10(self):
        train, test = generate_dataset("ModSum10", n_train=10, n_test=5, seed=0)
        for ex in train + test:
            assert ex["answer"] == (ex["a"] + ex["b"]) % 10

    def test_format_table(self):
        examples = [{"a": 1, "b": 2, "answer": 3}]
        table = format_examples_as_table(examples)
        assert "1" in table and "2" in table and "3" in table

    def test_format_query(self):
        q = format_query({"a": 3, "b": 5, "answer": 8})
        assert "3" in q and "5" in q


class TestSyntheticPipeline:
    def test_end_to_end_with_mock(self):
        train, test = generate_dataset("Max", n_train=5, n_test=3, seed=0)
        few_shot_text = format_examples_as_table(train[:3])
        agent = make_synthetic_agent(few_shot_text)

        llm = MagicMock(spec=LLMClient)
        llm.chat = MagicMock(return_value='{"answer": 7}\nThe max is 7.')

        trace = run_single_agent_episode(agent, format_query(test[0]), llm)
        assert trace.outcome == "completed"
        assert trace.steps[0].control.answer == 7
