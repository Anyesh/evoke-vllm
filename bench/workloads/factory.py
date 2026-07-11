from __future__ import annotations

from pathlib import Path

from bench.matrix import WorkloadSpec
from bench.workloads.base import Workload
from bench.workloads.memory_agent_bench import (
    ISSUE_ORDER_EXAMPLE_MAJOR,
    MabWorkloadConfig,
)
from bench.workloads.memory_agent_bench import load_workload as load_mab_workload
from bench.workloads.verdant_replay import load_workload as load_verdant_workload

KIND_MEMORY_AGENT_BENCH = "memory_agent_bench"
KIND_VERDANT_REPLAY = "verdant_replay"


def mab_config_from_spec(spec: WorkloadSpec) -> MabWorkloadConfig:
    raw_budget = spec.options.get("context_token_budget")
    raw_question_cap = spec.options.get("max_questions_per_example")
    raw_cold_cap = spec.options.get("cold_questions_per_example")
    return MabWorkloadConfig(
        workload_id=spec.id,
        track=spec.options["track"],
        sources=tuple(spec.options["sources"]),
        scorer=spec.options["scorer"],
        n_examples=int(spec.options.get("n_examples", 8)),
        seed=int(spec.options.get("seed", 42)),
        max_tokens=int(spec.options.get("max_tokens", 64)),
        context_token_budget=int(raw_budget) if raw_budget is not None else None,
        max_questions_per_example=(
            int(raw_question_cap) if raw_question_cap is not None else None
        ),
        tokenizer_id=spec.options.get("tokenizer_id"),
        issue_order=str(spec.options.get("issue_order", ISSUE_ORDER_EXAMPLE_MAJOR)),
        hot_examples=int(spec.options.get("hot_examples", 2)),
        cold_questions_per_example=(
            int(raw_cold_cap) if raw_cold_cap is not None else None
        ),
    )


def build_workload(spec: WorkloadSpec, default_trace_id: str) -> Workload:
    if spec.kind == KIND_MEMORY_AGENT_BENCH:
        return load_mab_workload(mab_config_from_spec(spec))

    if spec.kind == KIND_VERDANT_REPLAY:
        cas_root_raw = spec.options.get("cas_root") or None
        raw_prompt_budget = spec.options.get("prompt_token_budget")
        return load_verdant_workload(
            trace_path=Path(spec.options["trace_path"]),
            cas_root=Path(cas_root_raw) if cas_root_raw else None,
            trace_id=default_trace_id,
            max_tokens=int(spec.options.get("max_tokens", 64)),
            temperature=float(spec.options.get("temperature", 0.0)),
            prompt_token_budget=(
                int(raw_prompt_budget) if raw_prompt_budget is not None else None
            ),
            tokenizer_id=spec.options.get("tokenizer_id"),
        )

    raise ValueError(f"unknown workload kind {spec.kind!r}")
