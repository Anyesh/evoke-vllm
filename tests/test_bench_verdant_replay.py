from pathlib import Path

from bench.workloads.cas import CasReader
from bench.workloads.verdant_replay import (
    FileReadEvent,
    LlmCallEvent,
    PromptSegmentEvent,
    ToolCallEvent,
    build_replay_workload,
    drop_requests_over_token_budget,
    load_workload,
    parse_trace_jsonl,
)

FIXTURES = Path(__file__).parent / "bench_fixtures"
TRACE_PATH = FIXTURES / "trace" / "tiny_session.jsonl"
TRACE_CAS_ROOT = FIXTURES / "trace_cas"

SYS_DIGEST = "448aa4e01bbe7d9962cc1208fb8982f2d2efab53d43db4bba786df79d0d7a51b"
FILE_A_DIGEST = "de5e0884a80aba904a99f2ee41328218aea08086aae58b3267c3b0b713558224"


def test_parse_trace_jsonl_reads_all_kinds():
    events = parse_trace_jsonl(TRACE_PATH)
    assert len(events) == 6
    assert isinstance(events[0], ToolCallEvent)
    assert isinstance(events[1], FileReadEvent)
    assert isinstance(events[2], LlmCallEvent)
    assert isinstance(events[3], PromptSegmentEvent)
    assert isinstance(events[5], LlmCallEvent)

    first_call = events[2]
    assert first_call.system_prompt_hash == SYS_DIGEST
    assert first_call.upstream_seqs == (0, 1)
    assert first_call.completion_bytes == 17


def test_replay_with_real_cas_mixes_hits_and_misses():
    events = parse_trace_jsonl(TRACE_PATH)
    cas = CasReader(TRACE_CAS_ROOT)
    workload = build_replay_workload(events, cas, trace_id="trace-x", max_tokens=64)

    assert workload.workload_id == "W3"
    assert len(workload.requests) == 2
    assert workload.stats["llm_calls_seen"] == 2
    assert workload.stats["requests_built"] == 2
    assert workload.stats["segments_resolved"] == 6
    assert workload.stats["cas_hits"] == 3
    assert workload.stats["cas_hit_rate"] == 0.5


def test_first_call_uses_real_hits_and_filler_together():
    events = parse_trace_jsonl(TRACE_PATH)
    cas = CasReader(TRACE_CAS_ROOT)
    workload = build_replay_workload(events, cas, trace_id="trace-x")

    first = workload.requests[0]
    content = first.messages[0]["content"]
    assert "system prompt v1" in content
    assert "file A content" in content
    parts = content.split("\n")
    assert len(parts) == 3
    tool_a_filler = parts[1]
    assert len(tool_a_filler) == 13
    assert tool_a_filler != "system prompt v1"


def test_zero_length_upstream_is_skipped():
    events = parse_trace_jsonl(TRACE_PATH)
    cas = CasReader(TRACE_CAS_ROOT)
    workload = build_replay_workload(events, cas, trace_id="trace-x")

    second = workload.requests[1]
    assert second.metadata["n_segments"] == 3


def test_source_type_is_system_for_first_call_then_user():
    events = parse_trace_jsonl(TRACE_PATH)
    cas = CasReader(TRACE_CAS_ROOT)
    workload = build_replay_workload(events, cas, trace_id="trace-x")

    assert workload.requests[0].kv_transfer_params["evoke"]["source_type"] == "system"
    assert workload.requests[1].kv_transfer_params["evoke"]["source_type"] == "user"
    for request in workload.requests:
        assert request.kv_transfer_params["evoke"]["evoke_session"] == "trace-x"


def test_no_cas_root_falls_back_entirely_to_filler():
    events = parse_trace_jsonl(TRACE_PATH)
    cas = CasReader(None)
    workload = build_replay_workload(events, cas, trace_id="trace-x")

    assert workload.stats["cas_hits"] == 0
    assert workload.stats["cas_hit_rate"] == 0.0
    for request in workload.requests:
        assert request.messages[0]["content"]


def test_quality_neutral_by_construction():
    events = parse_trace_jsonl(TRACE_PATH)
    cas = CasReader(None)
    workload = build_replay_workload(events, cas, trace_id="trace-x")
    assert workload.score_fn is None
    assert workload.score(workload.requests[0], "anything") is None


def test_load_workload_reads_from_disk():
    workload = load_workload(
        TRACE_PATH, TRACE_CAS_ROOT, trace_id="trace-y", max_tokens=32
    )
    assert len(workload.requests) == 2
    assert workload.requests[0].max_tokens == 32
    assert workload.requests[0].temperature == 0.0


class WordTokenizer:
    def __call__(self, text, add_special_tokens=False):
        class Out:
            def __init__(self, ids):
                self.input_ids = ids

        return Out(text.split())


def test_drop_requests_over_token_budget_keeps_short_calls_in_order():
    # The real trace's reconstructed prompts run from 3 tokens to 2.8M
    # (p50 3,788, p90 182k), so calls that cannot fit the serving window
    # are dropped rather than truncated: cutting a replayed prompt would
    # rewrite its prefix lineage, while dropping keeps every surviving
    # call byte-identical across arms.
    events = parse_trace_jsonl(TRACE_PATH)
    cas = CasReader(TRACE_CAS_ROOT)
    workload = build_replay_workload(events, cas, trace_id="trace-z")
    lengths = [
        len("\n".join(m["content"] for m in r.messages).split())
        for r in workload.requests
    ]
    budget = sorted(lengths)[0]
    kept = drop_requests_over_token_budget(workload.requests, budget, WordTokenizer())
    assert 0 < len(kept) < len(workload.requests)
    kept_ids = [r.request_id for r in kept]
    original_order = [r.request_id for r in workload.requests if r in kept]
    assert kept_ids == original_order
