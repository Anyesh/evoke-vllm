"""Shared request and result types for bench workloads.

Every workload loader (MemoryAgentBench, verdant-replay) builds a plain list
of ``ChatRequest`` objects; ``bench/runner.py`` is the only place that knows
how to turn one into an HTTP call. Keeping the workloads HTTP-free is what
makes them unit-testable without a server.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChatRequest:
    request_id: str
    messages: list[dict[str, str]]
    temperature: float
    max_tokens: int
    kv_transfer_params: dict[str, Any] | None = None
    ground_truths: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredResult:
    request_id: str
    completion: str
    score: float | None
    ttft_seconds: float | None = None
    latency_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Workload:
    """A resolved, ready-to-run workload: requests plus its scoring rule.

    ``score_fn`` is ``None`` for workloads that are quality-neutral by
    construction (verdant replay); ``aggregate`` reduces per-request scores
    to the one number reported per cell (mean, ignoring ``None`` entries).
    """

    workload_id: str
    requests: list[ChatRequest]
    score_fn: Callable[[ChatRequest, str], float | None] | None = None
    stats: dict[str, Any] = field(default_factory=dict)

    def score(self, request: ChatRequest, completion: str) -> float | None:
        if self.score_fn is None:
            return None
        return self.score_fn(request, completion)

    @staticmethod
    def aggregate(scores: list[float | None]) -> float | None:
        present = [s for s in scores if s is not None]
        if not present:
            return None
        return sum(present) / len(present)
