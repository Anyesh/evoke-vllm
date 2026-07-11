import json
from pathlib import Path

import httpx

from bench.runner import run_cell
from bench.workloads.base import ChatRequest, Workload

FIXTURES = Path(__file__).parent / "bench_fixtures"
BEFORE_TEXT = (FIXTURES / "metrics_before.txt").read_text()
AFTER_TEXT = (FIXTURES / "metrics_after.txt").read_text()


def _sse_body(content: str) -> str:
    chunk = json.dumps({"choices": [{"delta": {"content": content}}]})
    return f"data: {chunk}\n\ndata: [DONE]\n\n"


def make_handler(completions: list[str]):
    calls = {"metrics": 0, "completion": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            calls["metrics"] += 1
            text = BEFORE_TEXT if calls["metrics"] == 1 else AFTER_TEXT
            return httpx.Response(200, text=text)
        if request.url.path == "/v1/chat/completions":
            index = calls["completion"]
            calls["completion"] += 1
            return httpx.Response(
                200,
                text=_sse_body(completions[index]),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(404)

    return handler, calls


def _score_fn(request: ChatRequest, completion: str) -> float | None:
    if not request.ground_truths:
        return None
    return 1.0 if request.ground_truths[0].lower() in completion.lower() else 0.0


def test_run_cell_scores_and_writes_json(tmp_path):
    workload = Workload(
        workload_id="W1",
        requests=[
            ChatRequest(
                request_id="r1",
                messages=[{"role": "user", "content": "q1"}],
                temperature=0.0,
                max_tokens=8,
                ground_truths=["answer"],
            ),
            ChatRequest(
                request_id="r2",
                messages=[{"role": "user", "content": "q2"}],
                temperature=0.0,
                max_tokens=8,
                ground_truths=["other"],
            ),
        ],
        score_fn=_score_fn,
        stats={"n_requests": 2},
    )
    handler, calls = make_handler(["the answer is 42", "something unrelated"])
    out_path = tmp_path / "result.json"

    with httpx.Client(
        base_url="http://test", transport=httpx.MockTransport(handler)
    ) as client:
        result = run_cell(
            client,
            arm="A2",
            workload=workload,
            budget="B2",
            server_url="http://test",
            model="test-model",
            out_path=out_path,
        )

    assert calls["metrics"] == 2
    assert calls["completion"] == 2
    assert result.arm == "A2"
    assert result.workload == "W1"
    assert result.budget == "B2"
    assert result.requests[0]["score"] == 1.0
    assert result.requests[1]["score"] == 0.0
    assert result.quality_score == 0.5
    assert result.requests[0]["ttft_seconds"] is not None
    assert result.metrics["prefill_tokens_avoided"] == 220.0

    assert out_path.exists()
    on_disk = json.loads(out_path.read_text())
    assert on_disk["workload"] == "W1"
    assert on_disk["workload_stats"] == {"n_requests": 2}


def test_run_cell_quality_neutral_workload_has_none_score(tmp_path):
    workload = Workload(
        workload_id="W3",
        requests=[
            ChatRequest(
                request_id="r1",
                messages=[{"role": "user", "content": "replay"}],
                temperature=0.0,
                max_tokens=8,
                ground_truths=None,
            ),
        ],
        score_fn=None,
    )
    handler, _calls = make_handler(["some replayed completion"])

    with httpx.Client(
        base_url="http://test", transport=httpx.MockTransport(handler)
    ) as client:
        result = run_cell(
            client,
            arm="A0",
            workload=workload,
            budget="NA",
            server_url="http://test",
            model="test-model",
        )

    assert result.requests[0]["score"] is None
    assert result.quality_score is None


def test_run_cell_without_out_path_does_not_write_file():
    workload = Workload(workload_id="W1", requests=[], score_fn=None)
    handler, _calls = make_handler([])

    with httpx.Client(
        base_url="http://test", transport=httpx.MockTransport(handler)
    ) as client:
        result = run_cell(
            client,
            arm="A0",
            workload=workload,
            budget="NA",
            server_url="http://test",
            model="test-model",
        )

    assert result.requests == []
    assert result.quality_score is None
