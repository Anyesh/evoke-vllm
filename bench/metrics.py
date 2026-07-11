"""Prometheus ``/metrics`` scraping and per-cell delta computation.

Hand-rolled text parser rather than ``prometheus_client``'s own (vLLM's
transitive dependency, not a declared one here), matching the same choice
``scripts/gate_lib.py`` already made for the GPU correctness gate. This
module is not imported from ``scripts/`` and does not import it either
(``pyproject.toml``'s ``pythonpath = ["scripts"]`` is pytest-only, and
``scripts/`` is deliberately not a package); the two Prometheus parsers stay
independent by design, but the semantics documented below were verified
against the same installed ``vllm==0.24.0`` build gate_lib.py checks.

Verified against the installed vllm package
(``vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py``):
the deprecated ``vllm:kv_offload_total_bytes`` counter's ``transfer_type``
label is ``CPU_to_GPU`` / ``GPU_to_CPU`` (mixed case), not the lowercase
``cpu_to_gpu``/``gpu_to_cpu`` spelling spec 02a-workloads.md's prose uses;
matching here is case-insensitive so a future vLLM release normalizing the
casing does not silently break this parser. The non-deprecated flat counters
(``vllm:kv_offload_load_bytes`` / ``_store_bytes``, direction in the name,
no label) are preferred when present.

Source-level verification alone is not enough here: prometheus_client's
text exposition appends ``_total`` to counter names, so a live ``/metrics``
endpoint serves ``vllm:generation_tokens_total`` where the source declares
``vllm:generation_tokens``. ``MetricsSnapshot._counter_samples`` resolves
both spellings; the ``tests/bench_fixtures/metrics_live_*.prom`` fixtures
are verbatim scrapes from a real vllm==0.24.0 server and pin the served
names.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

_METRIC_LINE_RE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)"
    r"(\{(?P<labels>[^}]*)\})?"
    r"\s+(?P<value>\S+)\s*$"
)
_LABEL_PAIR_RE = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')

LOAD_BYTES_METRIC = "vllm:kv_offload_load_bytes"
STORE_BYTES_METRIC = "vllm:kv_offload_store_bytes"
DEPRECATED_TOTAL_BYTES_METRIC = "vllm:kv_offload_total_bytes"


@dataclass(frozen=True)
class Sample:
    name: str
    labels: tuple[tuple[str, str], ...]
    value: float


def parse_prometheus_text(text: str) -> list[Sample]:
    samples: list[Sample] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _METRIC_LINE_RE.match(line)
        if not match:
            continue
        raw_labels = match.group("labels") or ""
        labels = tuple(sorted(_LABEL_PAIR_RE.findall(raw_labels)))
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        samples.append(Sample(name=match.group("name"), labels=labels, value=value))
    return samples


def _labels_match(
    labels: dict[str, str],
    filters: dict[str, str],
    case_insensitive: bool,
    ignore_keys: tuple[str, ...] = (),
) -> bool:
    for key, wanted in filters.items():
        if key in ignore_keys:
            continue
        actual = labels.get(key)
        if actual is None:
            return False
        if case_insensitive:
            if actual.lower() != wanted.lower():
                return False
        elif actual != wanted:
            return False
    return True


class MetricsSnapshot:
    def __init__(self, samples: list[Sample]) -> None:
        self.samples = samples
        self._by_name: dict[str, list[Sample]] = {}
        for sample in samples:
            self._by_name.setdefault(sample.name, []).append(sample)

    @classmethod
    def parse(cls, text: str) -> MetricsSnapshot:
        return cls(parse_prometheus_text(text))

    def _counter_samples(self, name: str) -> list[Sample]:
        """Looks up a counter under both its source name and its exposition name.

        vLLM declares counters with bare names (vllm:generation_tokens,
        vllm:kv_offload_load_bytes, ...), but prometheus_client's text
        exposition appends "_total" to every counter that does not already
        end in it, so the live /metrics endpoint serves the suffixed
        spelling. Only these two exact spellings are consulted; prefix
        matching would also sweep in the "_created" timestamp gauges that
        accompany each counter.
        """
        return self._by_name.get(name + "_total") or self._by_name.get(name) or []

    def has(self, name: str) -> bool:
        return bool(self._counter_samples(name))

    def sum(
        self, name: str, case_insensitive_labels: bool = False, **label_filters: str
    ) -> float:
        total = 0.0
        for sample in self._counter_samples(name):
            if _labels_match(
                dict(sample.labels), label_filters, case_insensitive_labels
            ):
                total += sample.value
        return total

    def last(self, name: str) -> float | None:
        matches = self._by_name.get(name, [])
        if not matches:
            return None
        return matches[-1].value

    def buckets(
        self, base_name: str, **label_filters: str
    ) -> list[tuple[float, float]]:
        out: dict[float, float] = {}
        for sample in self._by_name.get(base_name + "_bucket", []):
            labels = dict(sample.labels)
            if not _labels_match(labels, label_filters, False, ignore_keys=("le",)):
                continue
            le_raw = labels.get("le")
            if le_raw is None:
                continue
            le = float("inf") if le_raw == "+Inf" else float(le_raw)
            out[le] = out.get(le, 0.0) + sample.value
        return sorted(out.items())


def diff_buckets(
    before: MetricsSnapshot,
    after: MetricsSnapshot,
    base_name: str,
    **label_filters: str,
) -> list[tuple[float, float]]:
    before_map = dict(before.buckets(base_name, **label_filters))
    after_map = dict(after.buckets(base_name, **label_filters))
    les = sorted(set(before_map) | set(after_map))
    return [
        (le, max(after_map.get(le, 0.0) - before_map.get(le, 0.0), 0.0)) for le in les
    ]


def quantile_from_buckets(buckets: list[tuple[float, float]], q: float) -> float | None:
    if not buckets:
        return None
    total = buckets[-1][1]
    if total <= 0:
        return None
    target = q * total
    prev_le, prev_count = 0.0, 0.0
    for le, count in buckets:
        if count >= target:
            if le == float("inf"):
                return prev_le
            if le == prev_le:
                return le
            frac = (target - prev_count) / max(count - prev_count, 1e-9)
            return prev_le + frac * (le - prev_le)
        prev_le, prev_count = le, count
    return buckets[-1][0]


def _offload_transfer_delta(
    before: MetricsSnapshot,
    after: MetricsSnapshot,
    preferred_name: str,
    deprecated_transfer_type: str,
) -> float:
    if after.has(preferred_name) or before.has(preferred_name):
        return after.sum(preferred_name) - before.sum(preferred_name)
    return after.sum(
        DEPRECATED_TOTAL_BYTES_METRIC,
        case_insensitive_labels=True,
        transfer_type=deprecated_transfer_type,
    ) - before.sum(
        DEPRECATED_TOTAL_BYTES_METRIC,
        case_insensitive_labels=True,
        transfer_type=deprecated_transfer_type,
    )


@dataclass(frozen=True)
class CellMetrics:
    wall_seconds: float
    prefix_cache_hits: float
    prefix_cache_queries: float
    external_prefix_cache_hits: float
    external_prefix_cache_queries: float
    restore_hit_rate: float | None
    prefill_tokens_avoided: float
    ttft_p50_seconds: float | None
    ttft_p99_seconds: float | None
    generation_tokens: float
    decode_tokens_per_second: float | None
    kv_cache_usage_perc_end: float | None
    offload_store_bytes: float
    offload_load_bytes: float

    def as_dict(self) -> dict:
        return asdict(self)


def compute_cell_metrics(
    before: MetricsSnapshot, after: MetricsSnapshot, wall_seconds: float
) -> CellMetrics:
    prefix_hits = after.sum("vllm:prefix_cache_hits") - before.sum(
        "vllm:prefix_cache_hits"
    )
    prefix_queries = after.sum("vllm:prefix_cache_queries") - before.sum(
        "vllm:prefix_cache_queries"
    )
    ext_hits = after.sum("vllm:external_prefix_cache_hits") - before.sum(
        "vllm:external_prefix_cache_hits"
    )
    ext_queries = after.sum("vllm:external_prefix_cache_queries") - before.sum(
        "vllm:external_prefix_cache_queries"
    )
    restore_hit_rate = ext_hits / ext_queries if ext_queries > 0 else None

    generation_tokens = after.sum("vllm:generation_tokens") - before.sum(
        "vllm:generation_tokens"
    )
    decode_tokens_per_second = (
        generation_tokens / wall_seconds if wall_seconds > 0 else None
    )

    ttft_buckets = diff_buckets(before, after, "vllm:time_to_first_token_seconds")
    ttft_p50 = quantile_from_buckets(ttft_buckets, 0.5)
    ttft_p99 = quantile_from_buckets(ttft_buckets, 0.99)

    return CellMetrics(
        wall_seconds=wall_seconds,
        prefix_cache_hits=prefix_hits,
        prefix_cache_queries=prefix_queries,
        external_prefix_cache_hits=ext_hits,
        external_prefix_cache_queries=ext_queries,
        restore_hit_rate=restore_hit_rate,
        prefill_tokens_avoided=prefix_hits + ext_hits,
        ttft_p50_seconds=ttft_p50,
        ttft_p99_seconds=ttft_p99,
        generation_tokens=generation_tokens,
        decode_tokens_per_second=decode_tokens_per_second,
        kv_cache_usage_perc_end=after.last("vllm:kv_cache_usage_perc"),
        offload_store_bytes=_offload_transfer_delta(
            before, after, STORE_BYTES_METRIC, "GPU_to_CPU"
        ),
        offload_load_bytes=_offload_transfer_delta(
            before, after, LOAD_BYTES_METRIC, "CPU_to_GPU"
        ),
    )
