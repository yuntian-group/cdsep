"""Episode runner: executes multi-agent interaction loops with routing and termination."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from cdsep.agent import Agent, AgentOutput
from cdsep.llm import LLMClient


@dataclass
class StepRecord:
    """Record of a single step in an episode."""
    step: int
    agent_name: str
    control: Any
    message: str
    raw: str
    is_valid: bool
    errors: list[str] = field(default_factory=list)


@dataclass
class EpisodeTrace:
    """Complete trace of an episode execution."""
    steps: list[StepRecord] = field(default_factory=list)
    outcome: str = "incomplete"  # "completed", "max_steps", "parse_failure", "routing_error"
    final_message: str = ""
    stability_errors: dict[str, int] = field(default_factory=lambda: {
        "parse_errors": 0,
        "validation_errors": 0,
        "routing_errors": 0,
        "max_steps_hit": 0,
    })

    @property
    def is_stable(self) -> bool:
        """Protocol stability: True iff no schema/parsing/routing violations.

        Note: this does NOT include max_steps_hit, which is a task-level
        completion issue (the LLM did not decide to stop) rather than a
        protocol violation. Control--data flow separation guarantees protocol
        stability by construction; whether the agents choose to terminate
        is a separate, task-quality question.
        """
        return (
            self.stability_errors["parse_errors"] == 0
            and self.stability_errors["validation_errors"] == 0
            and self.stability_errors["routing_errors"] == 0
        )

    @property
    def completed_cleanly(self) -> bool:
        """True iff the episode terminated via an explicit `terminate` route
        (no protocol violations AND no max-step timeout)."""
        return self.is_stable and self.outcome == "completed"

    def to_dict(self) -> dict:
        return {
            "num_steps": len(self.steps),
            "outcome": self.outcome,
            "final_message": self.final_message[:500],
            "stability_errors": self.stability_errors,
            "steps": [
                {
                    "step": s.step,
                    "agent": s.agent_name,
                    "control": str(s.control) if s.control else None,
                    "message": s.message[:300],
                    "is_valid": s.is_valid,
                }
                for s in self.steps
            ],
        }

    def get_messages_for_agent(self, agent_name: str) -> list[str]:
        """Get all data-flow messages relevant to a specific agent."""
        return [s.message for s in self.steps if s.agent_name == agent_name]

    def get_all_messages(self) -> str:
        """Concatenated summary of all data-flow messages."""
        parts = []
        for s in self.steps:
            parts.append(f"[{s.agent_name}]: {s.message}")
        return "\n".join(parts)


def run_episode(
    entry_agent: Agent,
    agents: dict[str, Agent],
    route_fn: Callable,
    task_input: str,
    llm: LLMClient,
    max_steps: int = 20,
) -> EpisodeTrace:
    """Run a single multi-agent episode.

    Args:
        entry_agent: The first agent to call.
        agents: Dict mapping agent names to Agent instances (includes entry_agent).
        route_fn: Function(control) -> str. Returns next agent name or "terminate".
        task_input: The task description/input text.
        llm: LLM client for making calls.
        max_steps: Maximum interaction steps before forced termination.

    Returns:
        EpisodeTrace with full step history and outcome metadata.
    """
    trace = EpisodeTrace()
    current_agent = entry_agent
    context = task_input
    conversation_history: list[dict[str, str]] = []

    for step_idx in range(max_steps):
        output: AgentOutput = current_agent.call(context, llm, conversation_history=None)

        record = StepRecord(
            step=step_idx,
            agent_name=current_agent.name,
            control=output.control,
            message=output.message,
            raw=output.raw,
            is_valid=output.is_valid,
            errors=output.parse_errors + output.validation_errors,
        )
        trace.steps.append(record)

        if not output.is_valid:
            if output.parse_errors:
                trace.stability_errors["parse_errors"] += 1
            if output.validation_errors:
                trace.stability_errors["validation_errors"] += 1
            trace.outcome = "parse_failure"
            trace.final_message = output.message
            return trace

        conversation_history.append({"role": "assistant", "content": output.raw})

        try:
            next_action = route_fn(output.control)
        except Exception:
            trace.stability_errors["routing_errors"] += 1
            trace.outcome = "routing_error"
            trace.final_message = output.message
            return trace

        if next_action == "terminate":
            trace.outcome = "completed"
            trace.final_message = output.message
            return trace

        if next_action not in agents:
            trace.stability_errors["routing_errors"] += 1
            trace.outcome = "routing_error"
            trace.final_message = f"Unknown agent: {next_action}"
            return trace

        current_agent = agents[next_action]
        context = (
            f"Previous agent ({record.agent_name}) said:\n{output.message}\n\n"
            f"Original task:\n{task_input}"
        )

    trace.stability_errors["max_steps_hit"] = 1
    trace.outcome = "max_steps"
    trace.final_message = trace.steps[-1].message if trace.steps else ""
    return trace


def run_single_agent_episode(
    agent: Agent,
    task_input: str,
    llm: LLMClient,
) -> EpisodeTrace:
    """Simplified runner for single-agent tasks (no routing needed)."""
    trace = EpisodeTrace()
    output = agent.call(task_input, llm)

    record = StepRecord(
        step=0,
        agent_name=agent.name,
        control=output.control,
        message=output.message,
        raw=output.raw,
        is_valid=output.is_valid,
        errors=output.parse_errors + output.validation_errors,
    )
    trace.steps.append(record)

    if output.is_valid:
        trace.outcome = "completed"
    else:
        if output.parse_errors:
            trace.stability_errors["parse_errors"] += 1
        if output.validation_errors:
            trace.stability_errors["validation_errors"] += 1
        trace.outcome = "parse_failure"

    trace.final_message = output.message
    return trace
