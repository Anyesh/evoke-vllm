"""Verdant session replay (W3): rebuild real agent prompts from a trace.

The mechanism follows what the recorded trace schema actually
carries: every ``llm_call`` event in
the recorded session trace has an empty
``prefix_segment_hashes`` list (checked programmatically across all 247
calls), so the real reuse lineage lives in ``upstream_seqs``, which points at
prior ``file_read``/``tool_call``/``prompt_segment``/``llm_call`` events.
This module resolves both: ``prefix_segment_hashes`` first (for forward
compatibility, if a future trace populates it) and then ``upstream_seqs`` in
ascending order, so a call's reconstructed prompt is system prompt, tool
defs, prefix segments, then every upstream event's content, in that order.

``seq`` is not a session-wide unique event id in the real trace: only 691 of
its 6918 events have a distinct ``seq`` value, because the file is a
concatenation of many sub-session recordings (subagents, compactions) that
each restart their own local counter near 0 (confirmed by 131 backward jumps
in per-kind ``seq`` sequence when scanning the file in order). Building one
global ``seq -> event`` map from the whole file and looking up
``upstream_seqs`` against it would silently resolve to whichever sub-session
happened to write that ``seq`` value last, which is not necessarily the
sub-session the referencing call belongs to. This module instead builds the
map incrementally while walking events in file order, so a call only ever
resolves ``upstream_seqs`` against events that appeared at or before it, most
recent write wins; that matches the observed structure (``seq`` is locally
unique and increasing within one sub-session block). ``llm_call`` events are
likewise emitted as requests in file order, not sorted by ``seq``, since
``seq`` order and chronological order are the same thing only within a block.

Neither ``system_prompt_hash``/``tool_def_hash`` nor ``prefix_segment_hashes``
carry a paired byte length in the trace schema (only ``prompt_segment``,
``file_read``, ``tool_call``, and ``llm_call`` events record one). For those,
and for a ``subagent_spawn`` upstream (which records no length at all),
``DEFAULT_FILLER_LENGTH`` is used when the CAS lookup misses; a real CAS hit
always uses the actual stored length regardless.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import transformers

from bench.workloads.base import ChatRequest, Workload
from bench.workloads.cas import ZERO_DIGEST, CasReader

DEFAULT_FILLER_LENGTH = 256


@dataclass(frozen=True)
class FileReadEvent:
    seq: int
    content_hash: str
    byte_length: int


@dataclass(frozen=True)
class ToolCallEvent:
    seq: int
    result_content_hash: str
    result_bytes: int
    upstream_seqs: tuple[int, ...]


@dataclass(frozen=True)
class LlmCallEvent:
    seq: int
    system_prompt_hash: str
    prefix_segment_hashes: tuple[str, ...]
    tool_def_hash: str
    upstream_seqs: tuple[int, ...]
    completion_hash: str
    completion_bytes: int


@dataclass(frozen=True)
class SubAgentSpawnEvent:
    seq: int
    parent_seq: int
    completion_hash: str


@dataclass(frozen=True)
class PromptSegmentEvent:
    seq: int
    text_hash: str
    text_bytes: int


TraceEvent = (
    FileReadEvent
    | ToolCallEvent
    | LlmCallEvent
    | SubAgentSpawnEvent
    | PromptSegmentEvent
)


class TraceParseError(ValueError):
    pass


def _parse_event(raw: dict, line_no: int) -> TraceEvent:
    kind = raw.get("kind")
    try:
        if kind == "file_read":
            return FileReadEvent(
                seq=raw["seq"],
                content_hash=raw["content_hash"],
                byte_length=raw["bytes"],
            )
        if kind == "tool_call":
            return ToolCallEvent(
                seq=raw["seq"],
                result_content_hash=raw["result_content_hash"],
                result_bytes=raw["result_bytes"],
                upstream_seqs=tuple(raw.get("upstream_seqs", [])),
            )
        if kind == "llm_call":
            return LlmCallEvent(
                seq=raw["seq"],
                system_prompt_hash=raw["system_prompt_hash"],
                prefix_segment_hashes=tuple(raw.get("prefix_segment_hashes", [])),
                tool_def_hash=raw["tool_def_hash"],
                upstream_seqs=tuple(raw.get("upstream_seqs", [])),
                completion_hash=raw["completion_hash"],
                completion_bytes=raw["completion_bytes"],
            )
        if kind == "subagent_spawn":
            return SubAgentSpawnEvent(
                seq=raw["seq"],
                parent_seq=raw["parent_seq"],
                completion_hash=raw["completion_hash"],
            )
        if kind == "prompt_segment":
            return PromptSegmentEvent(
                seq=raw["seq"], text_hash=raw["text_hash"], text_bytes=raw["text_bytes"]
            )
    except KeyError as exc:
        raise TraceParseError(
            f"line {line_no}: missing field {exc} for kind {kind!r}"
        ) from exc
    raise TraceParseError(f"line {line_no}: unknown trace event kind {kind!r}")


def parse_trace_jsonl(path: Path) -> list[TraceEvent]:
    events: list[TraceEvent] = []
    with Path(path).open() as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            events.append(_parse_event(json.loads(line), line_no))
    return events


@dataclass(frozen=True)
class Segment:
    role: str
    digest: str
    length: int | None


def _describe_upstream(event: TraceEvent) -> tuple[str, str, int | None] | None:
    if isinstance(event, FileReadEvent):
        return "document", event.content_hash, event.byte_length
    if isinstance(event, ToolCallEvent):
        return "document", event.result_content_hash, event.result_bytes
    if isinstance(event, PromptSegmentEvent):
        return "user", event.text_hash, event.text_bytes
    if isinstance(event, LlmCallEvent):
        return "assistant", event.completion_hash, event.completion_bytes
    if isinstance(event, SubAgentSpawnEvent):
        return "document", event.completion_hash, None
    return None


def _call_segments(call: LlmCallEvent, by_seq: dict[int, TraceEvent]) -> list[Segment]:
    segments: list[Segment] = []
    if call.system_prompt_hash and call.system_prompt_hash != ZERO_DIGEST:
        segments.append(Segment("system", call.system_prompt_hash, None))
    if call.tool_def_hash and call.tool_def_hash != ZERO_DIGEST:
        segments.append(Segment("tool_def", call.tool_def_hash, None))
    for digest in call.prefix_segment_hashes:
        if digest and digest != ZERO_DIGEST:
            segments.append(Segment("prefix", digest, None))
    for seq in sorted(call.upstream_seqs):
        upstream = by_seq.get(seq)
        if upstream is None:
            continue
        described = _describe_upstream(upstream)
        if described is None:
            continue
        role, digest, length = described
        if not digest or digest == ZERO_DIGEST or length == 0:
            continue
        segments.append(Segment(role, digest, length))
    return segments


@dataclass
class ReplayStats:
    llm_calls_seen: int = 0
    requests_built: int = 0
    segments_resolved: int = 0
    cas_hits: int = 0

    @property
    def cas_hit_rate(self) -> float | None:
        if self.segments_resolved == 0:
            return None
        return self.cas_hits / self.segments_resolved


def build_replay_workload(
    events: list[TraceEvent],
    cas: CasReader,
    trace_id: str,
    max_tokens: int = 64,
    temperature: float = 0.0,
) -> Workload:
    by_seq: dict[int, TraceEvent] = {}
    requests: list[ChatRequest] = []
    stats = ReplayStats()

    for event in events:
        if isinstance(event, LlmCallEvent):
            stats.llm_calls_seen += 1
            parts: list[str] = []
            segment_meta: list[dict] = []
            for segment in _call_segments(event, by_seq):
                fallback_length = (
                    segment.length
                    if segment.length is not None
                    else DEFAULT_FILLER_LENGTH
                )
                resolution = cas.resolve(segment.digest, fallback_length)
                parts.append(resolution.data.decode("utf-8", errors="replace"))
                segment_meta.append({"role": segment.role, "status": resolution.status})
                stats.segments_resolved += 1
                if resolution.status == "hit":
                    stats.cas_hits += 1

            prompt_text = "\n".join(parts)
            if prompt_text:
                source_type = "system" if stats.requests_built == 0 else "user"
                requests.append(
                    ChatRequest(
                        request_id=f"{trace_id}-call{stats.requests_built}-seq{event.seq}",
                        messages=[{"role": "user", "content": prompt_text}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        kv_transfer_params={
                            "evoke": {
                                "evoke_session": trace_id,
                                "source_type": source_type,
                                "priority": 1.0,
                                "segments": segment_meta,
                            }
                        },
                        ground_truths=None,
                        metadata={"seq": event.seq, "n_segments": len(segment_meta)},
                    )
                )
                stats.requests_built += 1

        by_seq[event.seq] = event

    return Workload(
        workload_id="W3",
        requests=requests,
        score_fn=None,
        stats={
            "llm_calls_seen": stats.llm_calls_seen,
            "requests_built": stats.requests_built,
            "segments_resolved": stats.segments_resolved,
            "cas_hits": stats.cas_hits,
            "cas_hit_rate": stats.cas_hit_rate,
        },
    )


def drop_requests_over_token_budget(
    requests: Sequence[ChatRequest], budget: int, tokenizer
) -> list[ChatRequest]:
    """Filters out calls whose reconstructed prompt exceeds the budget.

    Dropping rather than truncating, because cutting a replayed prompt
    would rewrite its prefix lineage (the content-addressed reuse this
    replay exists to exercise), while dropping keeps every surviving call
    byte-identical across benchmark arms. On the real trace a 14k budget
    keeps 150 of 246 calls, the session's natural early-to-mid growth.
    """
    kept = []
    for request in requests:
        text = "\n".join(m["content"] for m in request.messages)
        if len(tokenizer(text, add_special_tokens=False).input_ids) <= budget:
            kept.append(request)
    return kept


def load_workload(
    trace_path: Path,
    cas_root: Path | None,
    trace_id: str,
    max_tokens: int = 64,
    temperature: float = 0.0,
    prompt_token_budget: int | None = None,
    tokenizer_id: str | None = None,
) -> Workload:
    if prompt_token_budget is not None and tokenizer_id is None:
        raise ValueError(
            "prompt_token_budget requires tokenizer_id so the cut is made "
            "with the same tokenizer the served model uses"
        )
    events = parse_trace_jsonl(trace_path)
    cas = CasReader(cas_root)
    workload = build_replay_workload(
        events, cas, trace_id, max_tokens=max_tokens, temperature=temperature
    )
    if prompt_token_budget is not None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_id)
        workload = replace(
            workload,
            requests=drop_requests_over_token_budget(
                workload.requests, prompt_token_budget, tokenizer
            ),
        )
    return workload
