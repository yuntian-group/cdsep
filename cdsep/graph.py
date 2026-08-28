"""Computation graph for tracking agent interaction traces and enabling backward traversal."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cdsep.episode import EpisodeTrace


@dataclass
class TraceNode:
    """A node in the computation graph representing one agent step."""
    node_id: int
    agent_name: str
    prompt_used: str
    control: Any
    message: str
    parent_ids: list[int] = field(default_factory=list)

    def summary(self, max_len: int = 300) -> str:
        msg = self.message[:max_len]
        return f"[{self.agent_name}] {msg}"


class ComputationGraph:
    """DAG of TraceNodes built from episode traces.

    Supports backward traversal for the optimizer to construct
    per-agent interaction summaries.
    """

    def __init__(self):
        self.nodes: list[TraceNode] = []
        self._next_id = 0

    def add_from_trace(self, trace: EpisodeTrace, prompts: dict[str, str]) -> None:
        """Add nodes from an episode trace.

        Args:
            trace: The completed episode trace.
            prompts: Dict mapping agent_name -> system_prompt used during the episode.
        """
        prev_id: int | None = None
        for step in trace.steps:
            node = TraceNode(
                node_id=self._next_id,
                agent_name=step.agent_name,
                prompt_used=prompts.get(step.agent_name, ""),
                control=step.control,
                message=step.message,
                parent_ids=[prev_id] if prev_id is not None else [],
            )
            self.nodes.append(node)
            prev_id = self._next_id
            self._next_id += 1

    def get_nodes_for_agent(self, agent_name: str) -> list[TraceNode]:
        return [n for n in self.nodes if n.agent_name == agent_name]

    def build_interaction_summary(self, agent_name: str, include_all: bool = False) -> str:
        """Build a summary of interactions relevant to a specific agent.

        If include_all is True, includes all messages in the trace (for naive TextGrad).
        Otherwise, includes only data-flow messages (for separated mode).
        """
        lines = []
        for node in self.nodes:
            if include_all or node.agent_name == agent_name:
                lines.append(node.summary())
            elif node.parent_ids:
                for pid in node.parent_ids:
                    parent = self.nodes[pid] if pid < len(self.nodes) else None
                    if parent and parent.agent_name == agent_name:
                        lines.append(f"  -> Response from [{node.agent_name}]: {node.message[:200]}")
        return "\n".join(lines)

    def build_full_trace_summary(self) -> str:
        """Build a full summary of the entire interaction trace."""
        lines = []
        for node in self.nodes:
            lines.append(f"Step {node.node_id} [{node.agent_name}]: {node.message[:300]}")
        return "\n".join(lines)

    def clear(self) -> None:
        self.nodes.clear()
        self._next_id = 0
