"""Structured JSONL experiment logging."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ExperimentLogger:
    """Logs experiment events to a JSONL file."""

    def __init__(self, experiment_name: str, run_id: str | None = None, log_dir: str = "logs"):
        self.experiment_name = experiment_name
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"{self.run_id}.jsonl"
        self._start_time = time.time()

    def log(self, event_type: str, data: dict[str, Any]) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": round(time.time() - self._start_time, 3),
            "event": event_type,
            **data,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def log_llm_call(
        self,
        model: str,
        messages: list[dict],
        response_text: str,
        usage: dict | None = None,
        latency_ms: float | None = None,
        **extra: Any,
    ) -> None:
        self.log("llm_call", {
            "model": model,
            "messages": messages,
            "response": response_text,
            "usage": usage,
            "latency_ms": latency_ms,
            **extra,
        })

    def log_parse(self, raw: str, control: dict | None, message: str, errors: list[str]) -> None:
        self.log("parse", {
            "raw": raw,
            "control": control,
            "message": message[:500],
            "errors": errors,
        })

    def log_episode(self, episode_id: int, trace: list[dict], outcome: dict) -> None:
        self.log("episode", {
            "episode_id": episode_id,
            "trace": trace,
            "outcome": outcome,
        })

    def log_optimization_step(self, iteration: int, agent_name: str, old_prompt: str, new_prompt: str, loss: float) -> None:
        self.log("optimization_step", {
            "iteration": iteration,
            "agent": agent_name,
            "old_prompt": old_prompt,
            "new_prompt": new_prompt,
            "loss": loss,
        })

    def log_metrics(self, iteration: int, metrics: dict[str, Any]) -> None:
        self.log("metrics", {"iteration": iteration, **metrics})
