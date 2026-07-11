from pathlib import Path

import pytest

from bench.metrics import (
    MetricsSnapshot,
    compute_cell_metrics,
    diff_buckets,
    parse_prometheus_text,
    quantile_from_buckets,
)

FIXTURES = Path(__file__).parent / "bench_fixtures"
BEFORE_TEXT = (FIXTURES / "metrics_before.txt").read_text()
AFTER_TEXT = (FIXTURES / "metrics_after.txt").read_text()


def test_parse_prometheus_text_reads_labels_and_value():
    samples = parse_prometheus_text('vllm:prefix_cache_hits{model_name="qwen"} 100.0\n')
    assert len(samples) == 1
    assert samples[0].name == "vllm:prefix_cache_hits"
    assert dict(samples[0].labels) == {"model_name": "qwen"}
    assert samples[0].value == 100.0


def test_parse_prometheus_text_skips_comments_and_blank_lines():
    text = "# a comment\n\nvllm:generation_tokens 5.0\n"
    samples = parse_prometheus_text(text)
    assert len(samples) == 1
    assert samples[0].value == 5.0


def test_parse_prometheus_text_handles_unlabeled_metric():
    samples = parse_prometheus_text("vllm:generation_tokens 5.0\n")
    assert samples[0].labels == ()


def test_snapshot_sum_across_matching_series():
    snapshot = MetricsSnapshot.parse('vllm:x{a="1"} 3.0\nvllm:x{a="2"} 4.0\n')
    assert snapshot.sum("vllm:x") == 7.0
    assert snapshot.sum("vllm:x", a="1") == 3.0


def test_snapshot_case_insensitive_label_match():
    snapshot = MetricsSnapshot.parse(
        'vllm:kv_offload_total_bytes{transfer_type="CPU_to_GPU"} 42.0\n'
    )
    assert (
        snapshot.sum("vllm:kv_offload_total_bytes", transfer_type="cpu_to_gpu") == 0.0
    )
    assert (
        snapshot.sum(
            "vllm:kv_offload_total_bytes",
            case_insensitive_labels=True,
            transfer_type="cpu_to_gpu",
        )
        == 42.0
    )


def test_snapshot_last_returns_most_recent_gauge_sample():
    snapshot = MetricsSnapshot.parse("vllm:g 1.0\nvllm:g 2.0\n")
    assert snapshot.last("vllm:g") == 2.0
    assert snapshot.last("vllm:missing") is None


def test_diff_buckets_and_quantile_from_fixture():
    before = MetricsSnapshot.parse(BEFORE_TEXT)
    after = MetricsSnapshot.parse(AFTER_TEXT)
    buckets = diff_buckets(before, after, "vllm:time_to_first_token_seconds")
    as_dict = dict(buckets)
    assert as_dict[0.01] == 0
    assert as_dict[0.05] == 7
    assert as_dict[0.1] == 16
    assert as_dict[float("inf")] == 20

    p50 = quantile_from_buckets(buckets, 0.5)
    assert p50 is not None
    assert 0.05 < p50 <= 0.1


def test_quantile_from_buckets_empty_or_zero_total():
    assert quantile_from_buckets([], 0.5) is None
    assert quantile_from_buckets([(0.1, 0.0), (float("inf"), 0.0)], 0.5) is None


def test_compute_cell_metrics_end_to_end():
    before = MetricsSnapshot.parse(BEFORE_TEXT)
    after = MetricsSnapshot.parse(AFTER_TEXT)
    metrics = compute_cell_metrics(before, after, wall_seconds=10.0)

    assert metrics.prefix_cache_hits == 160.0
    assert metrics.prefix_cache_queries == 200.0
    assert metrics.external_prefix_cache_hits == 60.0
    assert metrics.external_prefix_cache_queries == 50.0
    assert metrics.restore_hit_rate == pytest.approx(60.0 / 50.0)
    assert metrics.prefill_tokens_avoided == 220.0
    assert metrics.generation_tokens == 800.0
    assert metrics.decode_tokens_per_second == pytest.approx(80.0)
    assert metrics.kv_cache_usage_perc_end == pytest.approx(0.42)
    assert metrics.offload_store_bytes == 10000.0
    assert metrics.offload_load_bytes == 7000.0
    assert metrics.ttft_p50_seconds is not None


def test_compute_cell_metrics_reads_real_openmetrics_total_suffix():
    # Captured verbatim from a live vllm==0.24.0 /metrics endpoint on the
    # RTX 2060 (2026-07-11): "before" right after server ready, "after"
    # following the 70-request fidelity-gate workload. prometheus_client's
    # exposition layer appends "_total" to every counter, so the served
    # names are vllm:prefix_cache_hits_total, vllm:generation_tokens_total,
    # vllm:kv_offload_load_bytes_total etc., never the bare names vLLM's
    # own source declares. The before-scrape has no kv_offload_* counters
    # at all (they first appear after the first transfer), which is the
    # real shape the has()-based dispatch in _offload_transfer_delta sees.
    before = MetricsSnapshot.parse((FIXTURES / "metrics_live_before.prom").read_text())
    after = MetricsSnapshot.parse((FIXTURES / "metrics_live_after.prom").read_text())
    metrics = compute_cell_metrics(before, after, wall_seconds=100.0)

    assert metrics.prefix_cache_hits == 8336.0
    assert metrics.prefix_cache_queries == 78442.0
    assert metrics.external_prefix_cache_hits == 1232.0
    assert metrics.external_prefix_cache_queries == 70106.0
    assert metrics.restore_hit_rate == pytest.approx(1232.0 / 70106.0)
    assert metrics.prefill_tokens_avoided == 9568.0
    assert metrics.generation_tokens == 2230.0
    assert metrics.decode_tokens_per_second == pytest.approx(22.30)
    assert metrics.kv_cache_usage_perc_end == pytest.approx(0.0)
    assert metrics.offload_store_bytes == 40370176.0
    assert metrics.offload_load_bytes == 35323904.0
    assert metrics.ttft_p50_seconds is not None


def test_compute_cell_metrics_falls_back_to_deprecated_transfer_type_metric():
    before = MetricsSnapshot.parse(
        'vllm:kv_offload_total_bytes{transfer_type="GPU_to_CPU"} 100.0\n'
        'vllm:kv_offload_total_bytes{transfer_type="CPU_to_GPU"} 40.0\n'
    )
    after = MetricsSnapshot.parse(
        'vllm:kv_offload_total_bytes{transfer_type="GPU_to_CPU"} 900.0\n'
        'vllm:kv_offload_total_bytes{transfer_type="CPU_to_GPU"} 140.0\n'
    )
    metrics = compute_cell_metrics(before, after, wall_seconds=1.0)
    assert metrics.offload_store_bytes == 800.0
    assert metrics.offload_load_bytes == 100.0


def test_compute_cell_metrics_zero_queries_gives_none_hit_rate():
    before = MetricsSnapshot.parse("vllm:external_prefix_cache_queries 0.0\n")
    after = MetricsSnapshot.parse("vllm:external_prefix_cache_queries 0.0\n")
    metrics = compute_cell_metrics(before, after, wall_seconds=1.0)
    assert metrics.restore_hit_rate is None
