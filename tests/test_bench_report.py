import json
from pathlib import Path

from bench.report import load_hot_examples, render_report


def _cell(arm, workload, budget, **metrics):
    return {
        "arm": arm,
        "workload": workload,
        "budget": budget,
        "wall_seconds": metrics.pop("wall_seconds", 100.0),
        "quality_score": metrics.pop("quality_score", 0.5),
        "metrics": metrics,
        "workload_stats": {"n_examples": 2, "n_requests": 4},
        "requests": [],
    }


def _write(tmp_path: Path, cell: dict) -> None:
    name = f"{cell['arm']}_{cell['workload']}_{cell['budget']}.json"
    (tmp_path / name).write_text(json.dumps(cell))


def test_report_renders_one_table_per_workload_with_formatted_metrics(tmp_path):
    _write(
        tmp_path,
        _cell(
            "A1",
            "W1S",
            "B1",
            wall_seconds=96.63,
            quality_score=0.5,
            restore_hit_rate=0.32362,
            external_prefix_cache_hits=226304.0,
            ttft_p50_seconds=1.3461,
            prefill_tokens_avoided=257360.0,
        ),
    )
    _write(
        tmp_path,
        _cell(
            "A2",
            "W1S",
            "B1",
            wall_seconds=76.61,
            quality_score=0.48,
            restore_hit_rate=0.58,
            external_prefix_cache_hits=405248.0,
            ttft_p50_seconds=0.225,
            prefill_tokens_avoided=436304.0,
        ),
    )
    report = render_report(tmp_path)

    assert "## W1S" in report
    a1_row = next(line for line in report.splitlines() if "| A1 " in line)
    assert "32.4%" in a1_row
    assert "96.6" in a1_row
    assert "1.346" in a1_row
    a2_row = next(line for line in report.splitlines() if "| A2 " in line)
    assert "58.0%" in a2_row


def test_report_marks_absent_metrics_and_sorts_stock_first(tmp_path):
    # A0 (stock, no offload) reports no restore_hit_rate at all; the report
    # must render absence as "-" rather than a fake 0, so parity rows and
    # no-signal rows stay distinguishable.
    _write(tmp_path, _cell("A2", "W1", "B1", restore_hit_rate=0.076))
    _write(tmp_path, _cell("A0", "W1", "NA", quality_score=0.47))
    report = render_report(tmp_path)

    lines = [line for line in report.splitlines() if line.startswith("| A")]
    assert lines[0].startswith("| A0 ")
    assert " - " in lines[0]
    assert "7.6%" in lines[1]


def _hot_col(row: str) -> str:
    cols = [c.strip() for c in row.strip().strip("|").split("|")]
    return cols[7]


def test_report_renders_hot_request_ttft_mean_for_skewed_workloads(tmp_path):
    # The README's headline hot-request TTFT must be readable off the
    # rendered report, not re-derived by hand from the raw request records.
    cell = _cell("A2", "W1S", "B1", restore_hit_rate=0.58)
    cell["requests"] = [
        {"ttft_seconds": 0.2, "metadata": {"example_index": 0}},
        {"ttft_seconds": 0.6, "metadata": {"example_index": 1}},
        {"ttft_seconds": 2.0, "metadata": {"example_index": 5}},
    ]
    _write(tmp_path, cell)
    report = render_report(tmp_path, hot_examples={"W1S": 2})

    assert "ttft hot mean s" in report
    a2_row = next(line for line in report.splitlines() if "| A2 " in line)
    assert _hot_col(a2_row) == "0.400"


def test_report_marks_hot_ttft_absent_without_a_hot_split(tmp_path):
    cell = _cell("A2", "W1", "B1", restore_hit_rate=0.076)
    cell["requests"] = [{"ttft_seconds": 0.5, "metadata": {"example_index": 0}}]
    _write(tmp_path, cell)
    report = render_report(tmp_path, hot_examples={"W1S": 2})

    a2_row = next(line for line in report.splitlines() if "| A2 " in line)
    assert _hot_col(a2_row) == "-"


def test_load_hot_examples_reads_the_published_skew_matrix():
    hot = load_hot_examples(Path("bench/matrix.toml"), Path("bench/matrix-skew.toml"))
    assert hot == {"W1S": 2}
