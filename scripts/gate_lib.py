"""Pure-Python core of the GPU correctness gate: workload construction,
OpenAI-completions request building, and Prometheus /metrics parsing.

Kept stdlib-only and vLLM-free so tests/test_gate_lib.py runs on the same
pure-unit lane as tests/test_config.py, with no server and no GPU. All
network I/O lives in fidelity_gate.py; this module only builds and parses
data.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

FILLER_WORDS = (
    "ledger",
    "quarter",
    "runway",
    "vendor",
    "migration",
    "cutover",
    "rollback",
    "budget",
    "audit",
    "backlog",
    "sprint",
    "cadence",
    "stakeholder",
    "pipeline",
    "throughput",
    "latency",
    "escalation",
    "incident",
    "postmortem",
    "rollout",
    "canary",
    "staging",
    "regression",
    "sandbox",
    "provisioning",
    "quota",
    "tenant",
    "shard",
    "replica",
    "checkpoint",
    "snapshot",
    "retention",
    "archive",
    "compliance",
    "onboarding",
    "handoff",
    "dependency",
    "interface",
    "contract",
    "schema",
)

SYSTEM_PREAMBLE = "You are a careful assistant tracking a running project brief."


@dataclass(frozen=True)
class WorkloadConfig:
    """Parameters for the multi-session growing-prefix workload.

    session_count and growth_turns control how many distinct hashed
    prefixes compete for GPU prefix-cache space; filler_words_per_turn
    controls how many KV blocks each growth turn adds. All three are
    operator-tunable from the CLI because the actual GPU pressure needed to
    force eviction depends on gpu_memory_utilization and VRAM the workload
    author cannot see from here.
    """

    session_count: int = 3
    growth_turns: int = 3
    filler_words_per_turn: int = 220
    seed: int = 20260711
    max_tokens: int = 32
    top_logprobs: int = 5
    model: str = "unset"


@dataclass(frozen=True)
class TurnRequest:
    session_id: str
    turn_index: int
    phase: str  # "growth" or "replay"
    prompt: str
    kv_transfer_params: dict = field(default_factory=dict)


def _session_ids(config: WorkloadConfig) -> list[str]:
    return [f"session-{i}" for i in range(config.session_count)]


def _filler_block(rng: random.Random, n_words: int) -> str:
    return " ".join(rng.choice(FILLER_WORDS) for _ in range(n_words)) + "."


def _session_code(rng: random.Random) -> str:
    return f"{rng.randint(100000, 999999)}"


def _tags(session_id: str) -> dict:
    return {
        "evoke": {
            "source_type": "user",
            "priority": 1.0,
            "evoke_session": session_id,
        }
    }


def build_session_prefixes(
    config: WorkloadConfig,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Grows each session's prefix turn by turn, by APPENDING text only.

    Appending only (never rewriting earlier text) keeps every earlier
    turn's token prefix byte-identical across turns, which is what makes
    vLLM's content-addressed block hashing treat turn k+1's prefix as a
    cache hit against turn k's stored blocks instead of a fresh sequence
    (a session is a chain of requests sharing a growing hashed prefix).

    Each session also gets a unique planted "reference code" in its first
    segment, a passkey-recall probe: it gives fidelity_gate.py's
    replay phase something concrete and human-legible to check for besides
    raw token equality.
    """
    prefixes: dict[str, list[str]] = {}
    codes: dict[str, str] = {}
    for session_index, session_id in enumerate(_session_ids(config)):
        rng = random.Random(config.seed * 1009 + session_index)
        code = _session_code(rng)
        codes[session_id] = code
        segments = [
            f"{SYSTEM_PREAMBLE} Session tag: {session_id}. Reference code: {code}."
        ]
        for turn in range(config.growth_turns):
            segments.append(
                f"Turn {turn} update for {session_id}: "
                + _filler_block(rng, config.filler_words_per_turn)
            )
        prefixes[session_id] = segments
        codes[session_id] = code
    return prefixes, codes


def build_turn_requests(config: WorkloadConfig) -> list[TurnRequest]:
    """Interleaves session turns round-robin, then replays each session.

    Growth phase: for each turn index, one request per session, in session
    order. A session's turn k+1 always arrives after every other session's
    turn k, so under a memory-constrained profile the other sessions'
    blocks compete for the small GPU prefix-cache pool while this session
    is idle between its own turns.

    Replay phase: one more request per session, after every growth turn for
    every session has already run. By this point a session's earlier
    blocks have had the maximum plausible amount of competing traffic to be
    evicted from the GPU pool; if the CPU offload tier still holds them
    (cpu_bytes_to_use sized generously relative to this workload, see
    profiles/*.env), the replay request is what should trigger a
    cpu_to_gpu restore.
    """
    prefixes, _codes = build_session_prefixes(config)
    session_ids = _session_ids(config)
    requests: list[TurnRequest] = []

    for turn in range(config.growth_turns):
        for session_id in session_ids:
            cumulative = " ".join(prefixes[session_id][: turn + 2])
            prompt = (
                cumulative
                + f"\n\nContinue the {session_id} brief in one short sentence."
            )
            requests.append(
                TurnRequest(
                    session_id=session_id,
                    turn_index=turn,
                    phase="growth",
                    prompt=prompt,
                    kv_transfer_params=_tags(session_id),
                )
            )

    for session_id in session_ids:
        cumulative = " ".join(prefixes[session_id])
        prompt = (
            cumulative
            + "\n\nQuestion: state this session's reference code, then "
            + "summarize the brief in one short sentence.\nAnswer:"
        )
        requests.append(
            TurnRequest(
                session_id=session_id,
                turn_index=config.growth_turns,
                phase="replay",
                prompt=prompt,
                kv_transfer_params=_tags(session_id),
            )
        )
    return requests


def build_completion_payload(turn: TurnRequest, config: WorkloadConfig) -> dict:
    """Builds the exact JSON body posted to POST /v1/completions.

    temperature=0 for determinism; logprobs=config.top_logprobs asks the
    OpenAI-compatible completions endpoint for a top-k logprob dict at each
    generated position (vllm.entrypoints.openai.completion.protocol.
    CompletionRequest.logprobs). kv_transfer_params is a first-class field
    on that same request model, which is how per-request EVOKE tags
    (source_type, priority, evoke_session) reach
    evoke_vllm.manager.EvokeOffloadingManager.prepare_store without going
    through a chat template.
    """
    return {
        "model": config.model,
        "prompt": turn.prompt,
        "max_tokens": config.max_tokens,
        "temperature": 0,
        "logprobs": config.top_logprobs,
        "kv_transfer_params": turn.kv_transfer_params,
    }


@dataclass(frozen=True)
class RecordedResponse:
    session_id: str
    turn_index: int
    phase: str
    tokens: list[str]
    token_logprobs: list[float | None]
    text: str = ""


@dataclass(frozen=True)
class FidelityFailure:
    session_id: str
    turn_index: int
    phase: str
    reason: str


def compare_responses(
    baseline: list[RecordedResponse],
    evoke: list[RecordedResponse],
    logprob_atol: float = 0.05,
) -> list[FidelityFailure]:
    """Diffs two recorded runs of the same workload token-for-token.

    A token mismatch at any position is a hard failure. When tokens match,
    logprobs are compared with a tolerance rather than exact equality,
    because prefix-cache/continuous-batching reduction order can introduce
    small floating-point noise between two runs even at temperature=0; that
    noise is not the fidelity property this gate is checking for.
    """
    failures: list[FidelityFailure] = []
    evoke_by_key = {(r.session_id, r.turn_index, r.phase): r for r in evoke}
    for base in baseline:
        key = (base.session_id, base.turn_index, base.phase)
        other = evoke_by_key.get(key)
        if other is None:
            failures.append(
                FidelityFailure(*key, reason="no matching evoke-run response")
            )
            continue
        if base.tokens != other.tokens:
            failures.append(
                FidelityFailure(
                    *key,
                    reason=(
                        f"token mismatch: baseline={base.tokens!r} "
                        f"evoke={other.tokens!r}"
                    ),
                )
            )
            continue
        # strict=False: token equality is already checked above, but the
        # logprobs lists are independently-sourced JSON fields that are not
        # worth crashing the gate over if a server response is malformed.
        for pos, (base_lp, evoke_lp) in enumerate(
            zip(base.token_logprobs, other.token_logprobs, strict=False)
        ):
            if base_lp is None or evoke_lp is None:
                continue
            if abs(base_lp - evoke_lp) > logprob_atol:
                failures.append(
                    FidelityFailure(
                        *key,
                        reason=(
                            f"logprob mismatch at position {pos}: "
                            f"baseline={base_lp} evoke={evoke_lp} "
                            f"(atol={logprob_atol})"
                        ),
                    )
                )
                break
    return failures


_METRIC_LINE_RE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)"
    r"(\{(?P<labels>[^}]*)\})?"
    r"\s+(?P<value>\S+)\s*$"
)
_LABEL_PAIR_RE = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


@dataclass(frozen=True)
class MetricSample:
    labels: dict[str, str]
    value: float


def parse_prometheus_text(text: str) -> dict[str, list[MetricSample]]:
    """Parses Prometheus text-exposition-format /metrics output.

    Hand-rolled rather than borrowed from prometheus_client's own parser
    (a transitive vLLM dependency, not a declared one here) so this gate
    has no dependency on a library whose parser version and behavior can
    drift independently of vLLM's own emitted format.
    """
    samples: dict[str, list[MetricSample]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _METRIC_LINE_RE.match(line)
        if not match:
            continue
        name = match.group("name")
        raw_labels = match.group("labels") or ""
        labels = dict(_LABEL_PAIR_RE.findall(raw_labels))
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        samples.setdefault(name, []).append(MetricSample(labels=labels, value=value))
    return samples


def _counter_samples(
    samples: dict[str, list[MetricSample]], name: str
) -> list[MetricSample]:
    """Looks up a counter under both its source name and its exposition name.

    vLLM's metrics.py declares counters with bare names such as
    vllm:kv_offload_total_bytes, but prometheus_client's text exposition
    appends "_total" to every counter that does not already end in it, so
    the /metrics endpoint serves vllm:kv_offload_total_bytes_total. Only
    these two exact spellings are consulted; prefix matching would also
    sweep in the "_created" timestamp gauges that accompany each counter.
    """
    return samples.get(name + "_total") or samples.get(name) or []


def _sum(samples: dict[str, list[MetricSample]], name: str) -> float:
    return sum(sample.value for sample in _counter_samples(samples, name))


def _sum_by_transfer_type(
    samples: dict[str, list[MetricSample]], name: str, transfer_type: str
) -> float:
    return sum(
        sample.value
        for sample in _counter_samples(samples, name)
        if sample.labels.get("transfer_type", "").lower() == transfer_type.lower()
    )


@dataclass(frozen=True)
class OffloadMetricsSummary:
    external_prefix_cache_hits: float
    external_prefix_cache_queries: float
    cpu_to_gpu_bytes: float
    gpu_to_cpu_bytes: float
    kv_cache_usage_perc: float | None

    def as_dict(self) -> dict:
        return {
            "external_prefix_cache_hits": self.external_prefix_cache_hits,
            "external_prefix_cache_queries": self.external_prefix_cache_queries,
            "cpu_to_gpu_bytes": self.cpu_to_gpu_bytes,
            "gpu_to_cpu_bytes": self.gpu_to_cpu_bytes,
            "kv_cache_usage_perc": self.kv_cache_usage_perc,
        }

    @classmethod
    def from_dict(cls, data: dict) -> OffloadMetricsSummary:
        return cls(
            external_prefix_cache_hits=data["external_prefix_cache_hits"],
            external_prefix_cache_queries=data["external_prefix_cache_queries"],
            cpu_to_gpu_bytes=data["cpu_to_gpu_bytes"],
            gpu_to_cpu_bytes=data["gpu_to_cpu_bytes"],
            kv_cache_usage_perc=data["kv_cache_usage_perc"],
        )


def summarize_offload_metrics(text: str) -> OffloadMetricsSummary:
    """Reduces a /metrics scrape to the counters the gate reasons about.

    vllm==0.24.0 (verified against the installed
    vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py)
    emits the deprecated vllm:kv_offload_total_bytes counter with a
    transfer_type label valued "CPU_to_GPU" / "GPU_to_CPU" (mixed case).
    Matching is case-insensitive here so this gate does not silently
    break on a future vLLM release that normalizes the casing.
    The flat, non-deprecated vllm:kv_offload_load_bytes /
    vllm:kv_offload_store_bytes counters (direction encoded in the metric
    name, no label at all) are the fallback for once the deprecated metric
    is removed upstream; load = restores = cpu_to_gpu, store = offloads =
    gpu_to_cpu.

    Source-level verification alone was not enough: prometheus_client's
    exposition appends "_total" to every counter name, so the live
    /metrics endpoint serves vllm:kv_offload_total_bytes_total and
    vllm:external_prefix_cache_hits_total. _counter_samples resolves both
    spellings; tests/fixtures/metrics_live_vllm_0_24_0.prom is a verbatim
    scrape from a real vllm==0.24.0 server and pins the served names.
    """
    samples = parse_prometheus_text(text)

    if _counter_samples(samples, "vllm:kv_offload_total_bytes"):
        cpu_to_gpu = _sum_by_transfer_type(
            samples, "vllm:kv_offload_total_bytes", "cpu_to_gpu"
        )
        gpu_to_cpu = _sum_by_transfer_type(
            samples, "vllm:kv_offload_total_bytes", "gpu_to_cpu"
        )
    else:
        cpu_to_gpu = _sum(samples, "vllm:kv_offload_load_bytes")
        gpu_to_cpu = _sum(samples, "vllm:kv_offload_store_bytes")

    kv_cache_usage_samples = samples.get("vllm:kv_cache_usage_perc")
    kv_cache_usage_perc = (
        kv_cache_usage_samples[-1].value if kv_cache_usage_samples else None
    )

    return OffloadMetricsSummary(
        external_prefix_cache_hits=_sum(samples, "vllm:external_prefix_cache_hits"),
        external_prefix_cache_queries=_sum(
            samples, "vllm:external_prefix_cache_queries"
        ),
        cpu_to_gpu_bytes=cpu_to_gpu,
        gpu_to_cpu_bytes=gpu_to_cpu,
        kv_cache_usage_perc=kv_cache_usage_perc,
    )


def metrics_delta(
    before: OffloadMetricsSummary, after: OffloadMetricsSummary
) -> OffloadMetricsSummary:
    """Attributes counter growth to one workload run.

    /metrics counters accumulate for the whole server lifetime, so a
    nonzero absolute value in "after" alone does not prove this run
    produced any restores; only the delta over "before" does.
    kv_cache_usage_perc is a gauge, reported as the latest snapshot rather
    than differenced.
    """
    return OffloadMetricsSummary(
        external_prefix_cache_hits=(
            after.external_prefix_cache_hits - before.external_prefix_cache_hits
        ),
        external_prefix_cache_queries=(
            after.external_prefix_cache_queries - before.external_prefix_cache_queries
        ),
        cpu_to_gpu_bytes=after.cpu_to_gpu_bytes - before.cpu_to_gpu_bytes,
        gpu_to_cpu_bytes=after.gpu_to_cpu_bytes - before.gpu_to_cpu_bytes,
        kv_cache_usage_perc=after.kv_cache_usage_perc,
    )
