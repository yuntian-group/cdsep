"""Tests for the Agent class with mocked LLM."""

from dataclasses import dataclass
from typing import Literal
from unittest.mock import MagicMock

import pytest

from cdsep.agent import Agent, AgentOutput
from cdsep.llm import LLMClient


@dataclass
class SimpleControl:
    answer: int


@dataclass
class LeaderControl:
    action: Literal["send", "stop"]
    target_agent: str
    stop: bool


def _mock_llm(responses: list[str]) -> LLMClient:
    """Create a mock LLMClient that returns predefined responses in order."""
    llm = MagicMock(spec=LLMClient)
    llm.chat = MagicMock(side_effect=responses)
    return llm


class TestAgentCall:
    def test_valid_response(self):
        agent = Agent("solver", SimpleControl, "Solve the problem.")
        llm = _mock_llm(['{"answer": 42}\nThe answer is forty-two.'])

        out = agent.call("What is 6*7?", llm)
        assert out.is_valid
        assert out.control.answer == 42
        assert "forty-two" in out.message

    def test_retry_on_no_json(self):
        agent = Agent("solver", SimpleControl, "Solve the problem.", max_parse_retries=1)
        responses = [
            "The answer is 42.",
            '{"answer": 42}\nNow with proper format.',
        ]
        llm = _mock_llm(responses)

        out = agent.call("What is 6*7?", llm)
        assert out.is_valid
        assert out.control.answer == 42
        assert llm.chat.call_count == 2

    def test_retry_on_validation_error(self):
        agent = Agent("leader", LeaderControl, "Coordinate.", max_parse_retries=1)
        responses = [
            '{"action": "jump", "target_agent": "w1", "stop": false}\nBad action.',
            '{"action": "send", "target_agent": "w1", "stop": false}\nGood now.',
        ]
        llm = _mock_llm(responses)

        out = agent.call("Start work.", llm)
        assert out.is_valid
        assert out.control.action == "send"

    def test_all_retries_exhausted(self):
        agent = Agent("solver", SimpleControl, "Solve.", max_parse_retries=1)
        responses = [
            "No json here",
            "Still no json",
        ]
        llm = _mock_llm(responses)

        out = agent.call("Question?", llm)
        assert not out.is_valid
        assert out.control is None
        assert len(out.parse_errors) > 0

    def test_system_message_contains_schema(self):
        agent = Agent("test", SimpleControl, "My custom prompt.")
        sys_msg = agent.build_system_message()
        assert "My custom prompt" in sys_msg
        assert '"answer"' in sys_msg
        assert "JSON" in sys_msg

    def test_clone_independence(self):
        agent = Agent("solver", SimpleControl, "Original prompt.")
        clone = agent.clone()
        clone.system_prompt = "Modified prompt."
        assert agent.system_prompt == "Original prompt."
        assert clone.system_prompt == "Modified prompt."

    def test_conversation_history_forwarded(self):
        agent = Agent("solver", SimpleControl, "Solve.")
        llm = _mock_llm(['{"answer": 5}\nDone.'])

        history = [{"role": "user", "content": "Previous question"}]
        out = agent.call("New question", llm, conversation_history=history)
        assert out.is_valid
        call_args = llm.chat.call_args[0][0]
        assert any("Previous question" in m.get("content", "") for m in call_args)
