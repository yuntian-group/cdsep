"""Tests for the TextGrad optimizer."""

from dataclasses import dataclass
from typing import Literal
from unittest.mock import MagicMock

import pytest

from cdsep.agent import Agent
from cdsep.episode import EpisodeTrace, StepRecord
from cdsep.graph import ComputationGraph
from cdsep.llm import LLMClient
from cdsep.optimizer import TextGradOptimizer


@dataclass
class SimpleControl:
    answer: int


@dataclass
class LeaderControl:
    action: Literal["send", "stop"]
    target_agent: str
    stop: bool


def _mock_optimizer_llm(response: str) -> LLMClient:
    llm = MagicMock(spec=LLMClient)
    llm.chat = MagicMock(return_value=response)
    return llm


def _make_trace() -> EpisodeTrace:
    trace = EpisodeTrace()
    trace.steps.append(StepRecord(
        step=0, agent_name="solver",
        control=SimpleControl(answer=5), message="I think the answer is 5.",
        raw='{"answer":5}\nI think the answer is 5.',
        is_valid=True,
    ))
    trace.outcome = "completed"
    trace.final_message = "I think the answer is 5."
    return trace


class TestTextGradOptimizer:
    def test_separated_mode_calls_llm(self):
        new_prompt = "You are given examples of a function. Compute the sum modulo 10."
        llm = _mock_optimizer_llm(new_prompt)
        optimizer = TextGradOptimizer(llm)

        agent = Agent("solver", SimpleControl, "Solve the function.")
        graph = ComputationGraph()
        graph.add_from_trace(_make_trace(), {"solver": "Solve the function."})

        result = optimizer.optimize(agent, graph, loss=0.6, separated=True)
        assert result == new_prompt
        llm.chat.assert_called_once()

        call_args = llm.chat.call_args[0][0]
        prompt_text = call_args[0]["content"]
        assert "Solve the function." in prompt_text
        assert "ONLY modify" in prompt_text

    def test_naive_mode_includes_full_prompt(self):
        new_prompt = "Fully rewritten prompt with schema changes."
        llm = _mock_optimizer_llm(new_prompt)
        optimizer = TextGradOptimizer(llm)

        agent = Agent("solver", SimpleControl, "Solve the function.")
        graph = ComputationGraph()
        graph.add_from_trace(_make_trace(), {"solver": "Solve the function."})

        result = optimizer.optimize(agent, graph, loss=0.8, separated=False)
        assert result == new_prompt

        call_args = llm.chat.call_args[0][0]
        prompt_text = call_args[0]["content"]
        assert "Full Prompt" in prompt_text
        assert "You may modify any part" in prompt_text

    def test_strips_markdown_fences(self):
        fenced = "```\nClean prompt content.\n```"
        llm = _mock_optimizer_llm(fenced)
        optimizer = TextGradOptimizer(llm)

        agent = Agent("solver", SimpleControl, "Old prompt.")
        graph = ComputationGraph()
        result = optimizer.optimize(agent, graph, loss=0.5, separated=True)
        assert result == "Clean prompt content."
        assert "```" not in result

    def test_optimize_agents(self):
        llm = _mock_optimizer_llm("Better prompt.")
        optimizer = TextGradOptimizer(llm)

        agents = {
            "leader": Agent("leader", LeaderControl, "Lead."),
            "solver": Agent("solver", SimpleControl, "Solve."),
        }
        graph = ComputationGraph()

        results = optimizer.optimize_agents(agents, graph, loss=0.7, separated=True)
        assert "leader" in results
        assert "solver" in results
        assert llm.chat.call_count == 2
