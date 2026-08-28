"""cdsep: Control-Data Flow Separation for multi-agent LLM systems.

Public API
----------
- :class:`Agent` -- LLM-backed agent with a typed control schema and an
  optimizable system prompt.
- :class:`AgentOutput` -- ``(control, message)`` pair returned by ``Agent.call``.
- :func:`run_episode` -- multi-agent interaction loop with developer-supplied
  routing function.
- :func:`run_single_agent_episode` -- shortcut for single-agent tasks.
- :class:`EpisodeTrace`, :class:`StepRecord` -- structured trace of an episode,
  with stability guarantees by construction.
- :class:`ComputationGraph`, :class:`TraceNode` -- DAG over episode traces,
  used by the optimizer to assemble per-agent feedback.
- :class:`TextGradOptimizer` -- TextGrad-style optimizer; supports both the
  *separated* (default) and *naive* modes used in our experiments.
- :class:`LLMClient` -- thin OpenAI wrapper with retry + disk caching.
- :class:`ExperimentLogger` -- JSONL logger for reproducible experiment runs.
- :func:`generate_scaffolding`, :func:`parse_response`, :func:`validate_control`
  -- low-level schema utilities (rarely needed directly).

Quick start
-----------
>>> from dataclasses import dataclass
>>> from typing import Literal
>>> from cdsep import Agent, run_single_agent_episode, LLMClient
>>>
>>> @dataclass
... class Answer:
...     answer: int
>>>
>>> agent = Agent("solver", Answer, "You solve arithmetic. Output the integer.")
>>> # llm = LLMClient(model="gpt-5.4-nano")
>>> # trace = run_single_agent_episode(agent, "What is 6 * 7?", llm)
>>> # print(trace.steps[0].control.answer)
"""

from cdsep.agent import Agent, AgentOutput
from cdsep.episode import (
    EpisodeTrace,
    StepRecord,
    run_episode,
    run_single_agent_episode,
)
from cdsep.graph import ComputationGraph, TraceNode
from cdsep.llm import LLMCache, LLMClient
from cdsep.logging_utils import ExperimentLogger
from cdsep.optimizer import TextGradOptimizer
from cdsep.schema import (
    generate_scaffolding,
    get_schema_fields,
    parse_response,
    validate_control,
)

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentOutput",
    "ComputationGraph",
    "EpisodeTrace",
    "ExperimentLogger",
    "LLMCache",
    "LLMClient",
    "StepRecord",
    "TextGradOptimizer",
    "TraceNode",
    "generate_scaffolding",
    "get_schema_fields",
    "parse_response",
    "run_episode",
    "run_single_agent_episode",
    "validate_control",
    "__version__",
]
