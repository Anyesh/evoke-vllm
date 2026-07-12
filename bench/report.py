"""Render the benchmark report from per-cell result JSONs.

The report is generated, never hand-edited: every number in it is read
back from the JSON a `run-cell` invocation wrote, so the published tables
cannot drift from the published data. Regenerate with
``python -m bench report --results bench/results-published --out bench/REPORT.md``.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

PREAMBLE = """\
# Benchmark report

Every table below is rendered from the per-cell result JSONs next to this
file by `python -m bench report`; see `bench/README.md` for the harness,
workload definitions (`matrix.toml`, `matrix-skew.toml`), and how to rerun
any cell. Arms: A0 stock vLLM (no offload), A1 stock CPU offload with the
LRU policy, A2 CPU offload with the EVOKE policy from this package, A3
EVOKE composed with LMCache through MultiConnector.

W3 replays a recorded agent session for latency-overhead measurement; its
quality score is not meaningful and its content resolves to deterministic
filler. A `-` cell means the metric does not exist for that arm (stock
vLLM has no restore path), which is different from measuring zero.

`ttft hot mean s` is the mean client-side TTFT over requests that revisit
the hot set; it is defined only for workloads with a hot/cold split
(`hot_examples` in `matrix-skew.toml`) and rendered `-` everywhere else.
"""

COLUMNS = [
    ("arm", "arm"),
    ("budget", "budget"),
    ("wall s", "wall"),
    ("quality", "quality"),
    ("hit rate", "hit_rate"),
    ("hit tokens", "hit_tokens"),
    ("ttft p50 s", "ttft_p50"),
    ("ttft hot mean s", "ttft_hot"),
    ("prefill avoided", "avoided"),
]


def _fmt(value: float | None, pattern: str) -> str:
    if value is None:
        return "-"
    return pattern.format(value)


def _hot_ttft_mean(cell: dict, hot_n: int | None) -> float | None:
    if hot_n is None:
        return None
    ttfts = [
        req["ttft_seconds"]
        for req in cell.get("requests", [])
        if req.get("ttft_seconds") is not None
        and req.get("metadata", {}).get("example_index", hot_n) < hot_n
    ]
    if not ttfts:
        return None
    return sum(ttfts) / len(ttfts)


def _row(cell: dict, hot_n: int | None = None) -> dict[str, str]:
    metrics = cell.get("metrics", {})
    hit_rate = metrics.get("restore_hit_rate")
    return {
        "arm": cell["arm"],
        "budget": cell["budget"],
        "wall": _fmt(cell.get("wall_seconds"), "{:.1f}"),
        "quality": _fmt(cell.get("quality_score"), "{:.2f}"),
        "hit_rate": "-" if hit_rate is None else f"{hit_rate * 100:.1f}%",
        "hit_tokens": _fmt(metrics.get("external_prefix_cache_hits"), "{:.0f}"),
        "ttft_p50": _fmt(metrics.get("ttft_p50_seconds"), "{:.3f}"),
        "ttft_hot": _fmt(_hot_ttft_mean(cell, hot_n), "{:.3f}"),
        "avoided": _fmt(metrics.get("prefill_tokens_avoided"), "{:.0f}"),
    }


def load_hot_examples(*matrix_paths: Path) -> dict[str, int]:
    hot: dict[str, int] = {}
    for path in matrix_paths:
        if not path.is_file():
            continue
        data = tomllib.loads(path.read_text())
        for workload in data.get("workloads", []):
            if "hot_examples" in workload:
                hot[workload["id"]] = int(workload["hot_examples"])
    return hot


def load_cells(results_dir: Path) -> list[dict]:
    cells = []
    for path in sorted(results_dir.glob("*.json")):
        data = json.loads(path.read_text())
        if {"arm", "workload", "budget"} <= data.keys():
            cells.append(data)
    return cells


def render_report(results_dir: Path, hot_examples: dict[str, int] | None = None) -> str:
    cells = load_cells(results_dir)
    hot_examples = hot_examples or {}
    by_workload: dict[str, list[dict]] = {}
    for cell in cells:
        by_workload.setdefault(cell["workload"], []).append(cell)

    parts = [PREAMBLE]
    for workload in sorted(by_workload):
        rows = sorted(by_workload[workload], key=lambda c: (c["arm"], c["budget"]))
        parts.append(f"\n## {workload}\n")
        header = " | ".join(label for label, _ in COLUMNS)
        divider = " | ".join("---" for _ in COLUMNS)
        parts.append(f"| {header} |")
        parts.append(f"| {divider} |")
        for cell in rows:
            row = _row(cell, hot_examples.get(workload))
            values = " | ".join(row[key] for _, key in COLUMNS)
            parts.append(f"| {values} |")
    return "\n".join(parts) + "\n"
