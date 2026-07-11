"""Runner: executes one (arm, workload, budget) cell against a server URL.

Scrapes ``/metrics`` before and after the workload, issues every request in
``workload.requests`` sequentially against ``/v1/chat/completions`` with
``stream=true`` so time-to-first-token is measurable client-side (spec
02a-workloads.md section 4: TTFT is "corroborated by client-side first-token
latency under stream=true"), scores each completion with the workload's own
scorer, and writes one JSON file per cell.

``httpx.Client`` is passed in rather than constructed here so tests can
inject ``httpx.MockTransport`` and exercise this module with no real server
and no network.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from bench.metrics import MetricsSnapshot, compute_cell_metrics
from bench.workloads.base import ChatRequest, ScoredResult, Workload


@dataclass
class CellResult:
    arm: str
    workload: str
    budget: str
    server_url: str
    wall_seconds: float
    quality_score: float | None
    metrics: dict[str, Any]
    workload_stats: dict[str, Any]
    requests: list[dict[str, Any]]

    def as_dict(self) -> dict:
        return asdict(self)


def issue_chat_request(
    client: httpx.Client, request: ChatRequest, model: str
) -> tuple[str, float | None, float]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": request.messages,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "stream": True,
    }
    if request.kv_transfer_params is not None:
        payload["kv_transfer_params"] = request.kv_transfer_params

    started = time.monotonic()
    ttft: float | None = None
    text_parts: list[str] = []
    with client.stream("POST", "/v1/chat/completions", json=payload) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            choices = chunk.get("choices") or []
            if not choices:
                continue
            content = choices[0].get("delta", {}).get("content")
            if content:
                if ttft is None:
                    ttft = time.monotonic() - started
                text_parts.append(content)
    latency = time.monotonic() - started
    return "".join(text_parts), ttft, latency


def run_cell(
    client: httpx.Client,
    *,
    arm: str,
    workload: Workload,
    budget: str,
    server_url: str,
    model: str,
    out_path: Path | None = None,
) -> CellResult:
    before = MetricsSnapshot.parse(client.get("/metrics").text)
    started = time.monotonic()

    scored: list[ScoredResult] = []
    for request in workload.requests:
        completion, ttft, latency = issue_chat_request(client, request, model)
        score = workload.score(request, completion)
        scored.append(
            ScoredResult(
                request_id=request.request_id,
                completion=completion,
                score=score,
                ttft_seconds=ttft,
                latency_seconds=latency,
                metadata=dict(request.metadata),
            )
        )

    wall_seconds = time.monotonic() - started
    after = MetricsSnapshot.parse(client.get("/metrics").text)
    cell_metrics = compute_cell_metrics(before, after, wall_seconds)
    quality = Workload.aggregate([r.score for r in scored])

    result = CellResult(
        arm=arm,
        workload=workload.workload_id,
        budget=budget,
        server_url=server_url,
        wall_seconds=wall_seconds,
        quality_score=quality,
        metrics=cell_metrics.as_dict(),
        workload_stats=dict(workload.stats),
        requests=[asdict(r) for r in scored],
    )

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result.as_dict(), indent=2))

    return result
