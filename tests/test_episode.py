"""Tests for the episode runner."""

from dataclasses import dataclass
from typing import Literal
from unittest.mock import MagicMock

import pytest

from cdsep.agent import Agent, AgentOutput
from cdsep.episode import EpisodeTrace, run_episode, run_single_agent_episode
from cdsep.llm import LLMClient


@dataclass
class LeaderControl:
    action: Literal["send", "stop"]
    target_agent: str
    stop: bool


@dataclass
class WorkerControl:
    status: Literal["done"]


@dataclass
class SimpleControl:
    answer: int


def _make_agent_with_mock_call(name, schema, outputs: list[AgentOutput]) -> Agent:
    """Create an agent whose .call() returns predefined outputs in order."""
    agent = Agent(name, schema, "test prompt")
    call_iter = iter(outputs)
    agent.call = MagicMock(side_effect=lambda *a, **kw: next(call_iter))
    return agent


class TestRunEpisode:
    def test_single_step_terminate(self):
        leader_out = AgentOutput(
            control=LeaderControl(action="stop", target_agent="", stop=True),
            message="All done.",
            raw='{"action":"stop","target_agent":"","stop":true}\nAll done.',
        )
        leader = _make_agent_with_mock_call("leader", LeaderControl, [leader_out])

        def route(ctrl):
            if ctrl.stop:
                return "terminate"
            return ctrl.target_agent

        llm = MagicMock(spec=LLMClient)
        trace = run_episode(leader, {"leader": leader}, route, "task", llm)
        assert trace.outcome == "completed"
        assert trace.is_stable
        assert len(trace.steps) == 1

    def test_multi_agent_routing(self):
        leader_out1 = AgentOutput(
            control=LeaderControl(action="send", target_agent="worker", stop=False),
            message="Worker, do section 1.",
            raw='{"action":"send","target_agent":"worker","stop":false}\nWorker, do section 1.',
        )
        worker_out = AgentOutput(
            control=WorkerControl(status="done"),
            message="Section 1 looks good.",
            raw='{"status":"done"}\nSection 1 looks good.',
        )
        leader_out2 = AgentOutput(
            control=LeaderControl(action="stop", target_agent="", stop=True),
            message="Final review.",
            raw='{"action":"stop","target_agent":"","stop":true}\nFinal review.',
        )

        leader = _make_agent_with_mock_call("leader", LeaderControl, [leader_out1, leader_out2])
        worker = _make_agent_with_mock_call("worker", WorkerControl, [worker_out])

        def route(ctrl):
            if isinstance(ctrl, LeaderControl):
                if ctrl.stop:
                    return "terminate"
                return ctrl.target_agent
            if isinstance(ctrl, WorkerControl):
                return "leader"
            return "terminate"

        llm = MagicMock(spec=LLMClient)
        agents = {"leader": leader, "worker": worker}
        trace = run_episode(leader, agents, route, "Review paper", llm)

        assert trace.outcome == "completed"
        assert trace.is_stable
        assert len(trace.steps) == 3
        assert trace.steps[0].agent_name == "leader"
        assert trace.steps[1].agent_name == "worker"
        assert trace.steps[2].agent_name == "leader"

    def test_max_steps_prevents_infinite_loop(self):
        loop_out = AgentOutput(
            control=LeaderControl(action="send", target_agent="leader", stop=False),
            message="Looping...",
            raw='{"action":"send","target_agent":"leader","stop":false}\nLooping...',
        )
        outputs = [loop_out] * 10
        leader = _make_agent_with_mock_call("leader", LeaderControl, outputs)

        def route(ctrl):
            if ctrl.stop:
                return "terminate"
            return ctrl.target_agent

        llm = MagicMock(spec=LLMClient)
        trace = run_episode(leader, {"leader": leader}, route, "task", llm, max_steps=5)
        assert trace.outcome == "max_steps"
        assert trace.stability_errors["max_steps_hit"] == 1
        assert len(trace.steps) == 5

    def test_parse_failure_tracked(self):
        bad_out = AgentOutput(
            control=None,
            message="no json",
            raw="no json",
            parse_errors=["no JSON found"],
        )
        leader = _make_agent_with_mock_call("leader", LeaderControl, [bad_out])

        def route(ctrl):
            return "terminate"

        llm = MagicMock(spec=LLMClient)
        trace = run_episode(leader, {"leader": leader}, route, "task", llm)
        assert trace.outcome == "parse_failure"
        assert not trace.is_stable
        assert trace.stability_errors["parse_errors"] == 1

    def test_routing_error_tracked(self):
        out = AgentOutput(
            control=LeaderControl(action="send", target_agent="nonexistent", stop=False),
            message="Sending to nowhere.",
            raw='{"action":"send","target_agent":"nonexistent","stop":false}\nSending.',
        )
        leader = _make_agent_with_mock_call("leader", LeaderControl, [out])

        def route(ctrl):
            if ctrl.stop:
                return "terminate"
            return ctrl.target_agent

        llm = MagicMock(spec=LLMClient)
        trace = run_episode(leader, {"leader": leader}, route, "task", llm)
        assert trace.outcome == "routing_error"
        assert trace.stability_errors["routing_errors"] == 1


class TestSingleAgentEpisode:
    def test_successful_single_call(self):
        out = AgentOutput(
            control=SimpleControl(answer=7),
            message="The answer is 7.",
            raw='{"answer": 7}\nThe answer is 7.',
        )
        agent = _make_agent_with_mock_call("solver", SimpleControl, [out])
        llm = MagicMock(spec=LLMClient)

        trace = run_single_agent_episode(agent, "What is 3+4?", llm)
        assert trace.outcome == "completed"
        assert trace.is_stable
        assert trace.steps[0].control.answer == 7

    def test_trace_to_dict(self):
        out = AgentOutput(
            control=SimpleControl(answer=5),
            message="Five.",
            raw='{"answer":5}\nFive.',
        )
        agent = _make_agent_with_mock_call("solver", SimpleControl, [out])
        llm = MagicMock(spec=LLMClient)

        trace = run_single_agent_episode(agent, "2+3?", llm)
        d = trace.to_dict()
        assert d["outcome"] == "completed"
        assert d["num_steps"] == 1
        assert d["steps"][0]["agent"] == "solver"
