from pathlib import Path

import pytest
from gate_lib import (
    FidelityFailure,
    OffloadMetricsSummary,
    RecordedResponse,
    WorkloadConfig,
    build_completion_payload,
    build_session_prefixes,
    build_turn_requests,
    compare_responses,
    metrics_delta,
    parse_prometheus_text,
    summarize_offload_metrics,
)

# No vLLM import anywhere in this file: gate_lib.py is stdlib-only, so this
# lane runs on the same pure-unit boundary as tests/test_config.py.

FIXTURES = Path(__file__).parent / "fixtures"


def _config(**overrides) -> WorkloadConfig:
    base = {
        "session_count": 2,
        "growth_turns": 2,
        "filler_words_per_turn": 5,
        "seed": 1,
        "max_tokens": 16,
        "top_logprobs": 3,
        "model": "test-model",
    }
    base.update(overrides)
    return WorkloadConfig(**base)


def test_build_session_prefixes_grows_by_appending_only():
    config = _config()
    prefixes, _codes = build_session_prefixes(config)
    segments = prefixes["session-0"]
    assert len(segments) == config.growth_turns + 1
    cumulative_before = segments[0]
    cumulative_after = " ".join(segments[:2])
    assert cumulative_after.startswith(cumulative_before)


def test_build_session_prefixes_are_deterministic_for_same_seed():
    first, first_codes = build_session_prefixes(_config(seed=42))
    second, second_codes = build_session_prefixes(_config(seed=42))
    assert first == second
    assert first_codes == second_codes


def test_build_session_prefixes_differ_across_seeds():
    first, _ = build_session_prefixes(_config(seed=1))
    second, _ = build_session_prefixes(_config(seed=2))
    assert first != second


def test_build_session_prefixes_are_session_unique():
    _prefixes, codes = build_session_prefixes(_config(session_count=4))
    assert len(set(codes.values())) == 4


def test_build_turn_requests_counts_growth_then_replay():
    config = _config(session_count=3, growth_turns=2)
    turns = build_turn_requests(config)
    growth = [t for t in turns if t.phase == "growth"]
    replay = [t for t in turns if t.phase == "replay"]
    assert len(growth) == config.session_count * config.growth_turns
    assert len(replay) == config.session_count
    assert len(turns) == len(growth) + len(replay)


def test_build_turn_requests_interleaves_round_robin():
    config = _config(session_count=3, growth_turns=2)
    turns = build_turn_requests(config)
    growth = [t for t in turns if t.phase == "growth"]
    session_order = [t.session_id for t in growth]
    assert session_order == [
        "session-0",
        "session-1",
        "session-2",
        "session-0",
        "session-1",
        "session-2",
    ]


def test_build_turn_requests_replay_uses_full_grown_prefix():
    config = _config(session_count=1, growth_turns=2)
    turns = build_turn_requests(config)
    replay = next(t for t in turns if t.phase == "replay")
    growth_turn_1 = next(t for t in turns if t.phase == "growth" and t.turn_index == 1)
    assert growth_turn_1.prompt.split("\n\n")[0] in replay.prompt


def test_build_turn_requests_tags_carry_session_id():
    turns = build_turn_requests(_config())
    for turn in turns:
        assert turn.kv_transfer_params["evoke"]["evoke_session"] == turn.session_id
        assert turn.kv_transfer_params["evoke"]["source_type"] == "user"


def test_build_completion_payload_shape():
    config = _config(model="qwen2.5-1.5b-instruct", max_tokens=8, top_logprobs=4)
    turn = build_turn_requests(config)[0]
    payload = build_completion_payload(turn, config)
    assert payload["model"] == "qwen2.5-1.5b-instruct"
    assert payload["prompt"] == turn.prompt
    assert payload["max_tokens"] == 8
    assert payload["temperature"] == 0
    assert payload["logprobs"] == 4
    assert payload["kv_transfer_params"] == turn.kv_transfer_params


def test_build_completion_payload_is_json_serializable():
    import json

    config = _config()
    turn = build_turn_requests(config)[0]
    payload = build_completion_payload(turn, config)
    json.dumps(payload)


def _response(session_id="s", turn_index=0, phase="growth", tokens=None, lps=None):
    return RecordedResponse(
        session_id=session_id,
        turn_index=turn_index,
        phase=phase,
        tokens=tokens if tokens is not None else ["a", "b"],
        token_logprobs=lps if lps is not None else [-0.1, -0.2],
    )


def test_compare_responses_identical_runs_pass():
    baseline = [_response()]
    evoke = [_response()]
    assert compare_responses(baseline, evoke) == []


def test_compare_responses_detects_token_mismatch():
    baseline = [_response(tokens=["a", "b"])]
    evoke = [_response(tokens=["a", "c"])]
    failures = compare_responses(baseline, evoke)
    assert len(failures) == 1
    assert isinstance(failures[0], FidelityFailure)
    assert "token mismatch" in failures[0].reason


def test_compare_responses_tolerates_small_logprob_noise():
    baseline = [_response(lps=[-0.100])]
    evoke = [_response(lps=[-0.101])]
    assert compare_responses(baseline, evoke, logprob_atol=0.05) == []


def test_compare_responses_flags_logprob_divergence_beyond_tolerance():
    baseline = [_response(lps=[-0.10])]
    evoke = [_response(lps=[-0.90])]
    failures = compare_responses(baseline, evoke, logprob_atol=0.05)
    assert len(failures) == 1
    assert "logprob mismatch" in failures[0].reason


def test_compare_responses_flags_missing_evoke_response():
    baseline = [_response(session_id="only-in-baseline")]
    failures = compare_responses(baseline, [])
    assert len(failures) == 1
    assert "no matching evoke-run response" in failures[0].reason


def test_compare_responses_ignores_none_logprobs():
    baseline = [_response(lps=[None, -0.1])]
    evoke = [_response(lps=[None, -0.1])]
    assert compare_responses(baseline, evoke) == []


def test_parse_prometheus_text_reads_labeled_counter():
    text = 'vllm:external_prefix_cache_hits{model_name="m"} 42.0\n'
    samples = parse_prometheus_text(text)
    assert samples["vllm:external_prefix_cache_hits"][0].value == pytest.approx(42.0)
    assert samples["vllm:external_prefix_cache_hits"][0].labels == {"model_name": "m"}


def test_parse_prometheus_text_ignores_comments_and_blank_lines():
    text = "# HELP x y\n# TYPE x counter\n\nvllm:foo 1.0\n"
    samples = parse_prometheus_text(text)
    assert list(samples.keys()) == ["vllm:foo"]


def test_parse_prometheus_text_reads_unlabeled_gauge():
    samples = parse_prometheus_text("vllm:kv_cache_usage_perc 0.75\n")
    assert samples["vllm:kv_cache_usage_perc"][0].value == pytest.approx(0.75)
    assert samples["vllm:kv_cache_usage_perc"][0].labels == {}


def test_parse_prometheus_text_handles_inf_values():
    samples = parse_prometheus_text('x_bucket{le="+Inf"} 9.0\n')
    assert samples["x_bucket"][0].value == pytest.approx(9.0)


def test_summarize_offload_metrics_healthy_fixture_is_not_vacuous():
    text = (FIXTURES / "metrics_healthy.prom").read_text()
    summary = summarize_offload_metrics(text)
    assert summary.external_prefix_cache_hits == pytest.approx(384.0)
    assert summary.external_prefix_cache_queries == pytest.approx(512.0)
    assert summary.cpu_to_gpu_bytes == pytest.approx(6291456.0)
    assert summary.gpu_to_cpu_bytes == pytest.approx(9437184.0)
    assert summary.kv_cache_usage_perc == pytest.approx(0.83)


def test_summarize_offload_metrics_matches_transfer_type_case_insensitively():
    # vllm==0.24.0 emits "CPU_to_GPU"/"GPU_to_CPU", not the lowercase
    # "cpu_to_gpu" spelling used in spec 02a's prose; the fixture uses the
    # real mixed-case labels vLLM actually emits.
    text = (FIXTURES / "metrics_healthy.prom").read_text()
    assert 'transfer_type="CPU_to_GPU"' in text
    summary = summarize_offload_metrics(text)
    assert summary.cpu_to_gpu_bytes > 0


def test_summarize_offload_metrics_vacuous_fixture_is_vacuous():
    text = (FIXTURES / "metrics_vacuous.prom").read_text()
    summary = summarize_offload_metrics(text)
    assert summary.external_prefix_cache_hits == 0
    assert summary.cpu_to_gpu_bytes == 0


def test_summarize_offload_metrics_reads_real_openmetrics_total_suffix():
    # Captured verbatim from a live vllm==0.24.0 /metrics endpoint on the
    # RTX 2060 fidelity run (2026-07-11). prometheus_client's exposition
    # layer appends "_total" to every counter name, so the served names
    # are vllm:kv_offload_total_bytes_total and
    # vllm:external_prefix_cache_hits_total, not the bare names that
    # appear in vLLM's own metrics.py source. The "_created" gauges that
    # accompany each counter must not be summed into the counter values.
    text = (FIXTURES / "metrics_live_vllm_0_24_0.prom").read_text()
    summary = summarize_offload_metrics(text)
    assert summary.external_prefix_cache_hits == pytest.approx(1232.0)
    assert summary.external_prefix_cache_queries == pytest.approx(70106.0)
    assert summary.cpu_to_gpu_bytes == pytest.approx(35323904.0)
    assert summary.gpu_to_cpu_bytes == pytest.approx(40370176.0)
    assert summary.kv_cache_usage_perc == pytest.approx(0.0)


def test_summarize_offload_metrics_falls_back_to_flat_metric_names():
    text = (FIXTURES / "metrics_flat_fallback.prom").read_text()
    summary = summarize_offload_metrics(text)
    assert summary.cpu_to_gpu_bytes == pytest.approx(2048.0)
    assert summary.gpu_to_cpu_bytes == pytest.approx(4096.0)
    assert summary.external_prefix_cache_hits == pytest.approx(150.0)


def test_metrics_delta_attributes_growth_to_the_run():
    before = summarize_offload_metrics(
        'vllm:external_prefix_cache_hits{model_name="m"} 100.0\n'
        'vllm:kv_offload_total_bytes{model_name="m",'
        'transfer_type="CPU_to_GPU"} 1000.0\n'
    )
    after = summarize_offload_metrics(
        'vllm:external_prefix_cache_hits{model_name="m"} 137.0\n'
        'vllm:kv_offload_total_bytes{model_name="m",'
        'transfer_type="CPU_to_GPU"} 5192.0\n'
    )
    delta = metrics_delta(before, after)
    assert delta.external_prefix_cache_hits == pytest.approx(37.0)
    assert delta.cpu_to_gpu_bytes == pytest.approx(4192.0)


def test_metrics_delta_zero_growth_is_vacuous_signal():
    stable = summarize_offload_metrics((FIXTURES / "metrics_vacuous.prom").read_text())
    delta = metrics_delta(stable, stable)
    assert delta.external_prefix_cache_hits == 0
    assert delta.cpu_to_gpu_bytes == 0


def test_offload_metrics_summary_round_trips_through_dict():
    original = OffloadMetricsSummary(
        external_prefix_cache_hits=1.0,
        external_prefix_cache_queries=2.0,
        cpu_to_gpu_bytes=3.0,
        gpu_to_cpu_bytes=4.0,
        kv_cache_usage_perc=0.5,
    )
    assert OffloadMetricsSummary.from_dict(original.as_dict()) == original
