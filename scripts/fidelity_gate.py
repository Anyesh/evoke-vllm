"""GPU correctness gate: offload-then-restore fidelity against a baseline.

Two-phase, sequential-server workflow (a single 6-16GB GPU cannot usually
hold a baseline server and an EVOKE-connector server at once):

    python scripts/fidelity_gate.py record --base-url http://localhost:8000 \\
        --model qwen2.5-1.5b-instruct --run-label baseline --out results/baseline.json
    # stop that server, start the other config (see scripts/serve.sh)
    python scripts/fidelity_gate.py record --base-url http://localhost:8000 \\
        --model qwen2.5-1.5b-instruct --run-label evoke --out results/evoke.json
    python scripts/fidelity_gate.py compare --baseline results/baseline.json \\
        --evoke results/evoke.json --out results/fidelity_result.json

"record" drives a temperature-0 multi-session workload (scripts/gate_lib.py)
against a running server and scrapes /metrics before and after. "compare"
is pure offline JSON diffing: no server or GPU needed. It exits nonzero
either on a fidelity mismatch or when the evoke run produced zero restores
(the "gate is vacuous" case), so a CI-style caller can gate on exit status
alone.

--dry-run on "record" builds the full request plan and prints it (method,
URL, JSON body) without making any HTTP calls, so the workload and request
shapes can be validated without a server or GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from gate_lib import (
    OffloadMetricsSummary,
    RecordedResponse,
    WorkloadConfig,
    build_completion_payload,
    build_session_prefixes,
    build_turn_requests,
    compare_responses,
    metrics_delta,
    summarize_offload_metrics,
)

DEFAULT_TIMEOUT_SECONDS = 120.0


def _http_get(url: str, timeout: float) -> str:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"could not reach {url} ({exc}). Is the server running?"
        ) from exc


def _http_post_json(url: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} failed: {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"could not reach {url} ({exc}). Is the server running?"
        ) from exc


def _workload_config_from_args(args: argparse.Namespace) -> WorkloadConfig:
    return WorkloadConfig(
        session_count=args.sessions,
        growth_turns=args.growth_turns,
        filler_words_per_turn=args.filler_words,
        seed=args.seed,
        max_tokens=args.max_tokens,
        top_logprobs=args.top_logprobs,
        model=args.model,
    )


def _print_dry_run_plan(config: WorkloadConfig, base_url: str) -> None:
    _prefixes, codes = build_session_prefixes(config)
    turns = build_turn_requests(config)
    print(f"# workload: {len(turns)} requests over {config.session_count} sessions")
    print(f"# growth_turns={config.growth_turns} seed={config.seed}")
    print("# planted reference codes (informational, checked at record time):")
    for session_id, code in codes.items():
        print(f"#   {session_id}: {code}")
    print()
    for turn in turns:
        payload = build_completion_payload(turn, config)
        print(f"# {turn.session_id} turn={turn.turn_index} phase={turn.phase}")
        print(f"POST {base_url}/v1/completions")
        print(json.dumps(payload, indent=2))
        print()


def cmd_record(args: argparse.Namespace) -> int:
    config = _workload_config_from_args(args)
    base_url = args.base_url.rstrip("/")

    if args.dry_run:
        _print_dry_run_plan(config, base_url)
        return 0

    print(f"scraping {base_url}/metrics (before)")
    before_text = _http_get(f"{base_url}/metrics", args.timeout)
    before_summary = summarize_offload_metrics(before_text)

    turns = build_turn_requests(config)
    _prefixes, codes = build_session_prefixes(config)
    responses: list[RecordedResponse] = []
    passkey_recall: dict[str, bool] = {}

    for index, turn in enumerate(turns):
        payload = build_completion_payload(turn, config)
        print(
            f"[{index + 1}/{len(turns)}] {turn.session_id} "
            f"turn={turn.turn_index} phase={turn.phase}"
        )
        result = _http_post_json(f"{base_url}/v1/completions", payload, args.timeout)
        choice = result["choices"][0]
        logprobs = choice.get("logprobs") or {}
        tokens = logprobs.get("tokens", [])
        token_logprobs = logprobs.get("token_logprobs", [])
        text = choice.get("text", "")
        responses.append(
            RecordedResponse(
                session_id=turn.session_id,
                turn_index=turn.turn_index,
                phase=turn.phase,
                tokens=tokens,
                token_logprobs=token_logprobs,
                text=text,
            )
        )
        if turn.phase == "replay":
            passkey_recall[turn.session_id] = codes[turn.session_id] in text

    print(f"scraping {base_url}/metrics (after)")
    after_text = _http_get(f"{base_url}/metrics", args.timeout)
    after_summary = summarize_offload_metrics(after_text)
    delta = metrics_delta(before_summary, after_summary)

    if args.run_label == "evoke":
        if delta.external_prefix_cache_hits <= 0 or delta.cpu_to_gpu_bytes <= 0:
            print(
                "WARNING: this evoke-run recording shows zero external "
                "prefix cache hits or zero cpu_to_gpu offload bytes so far, "
                "and the gate will fail as vacuous at compare time unless "
                "this changes, so consider lowering "
                "EVOKE_GPU_MEMORY_UTILIZATION or increasing "
                "--sessions/--growth-turns/--filler-words.",
                file=sys.stderr,
            )

    record = {
        "run_label": args.run_label,
        "base_url": base_url,
        "model": config.model,
        "recorded_at": datetime.now(UTC).isoformat(),
        "workload": asdict(config),
        "responses": [asdict(response) for response in responses],
        "passkey_recall": passkey_recall,
        "metrics_before": before_summary.as_dict(),
        "metrics_after": after_summary.as_dict(),
        "metrics_delta": delta.as_dict(),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))
    print(f"wrote {out_path}")
    print(f"metrics delta over this run: {delta.as_dict()}")
    return 0


def _load_record(path: Path) -> dict:
    return json.loads(path.read_text())


def _responses_from_record(record: dict) -> list[RecordedResponse]:
    return [
        RecordedResponse(
            session_id=item["session_id"],
            turn_index=item["turn_index"],
            phase=item["phase"],
            tokens=item["tokens"],
            token_logprobs=item["token_logprobs"],
            text=item.get("text", ""),
        )
        for item in record["responses"]
    ]


def cmd_compare(args: argparse.Namespace) -> int:
    baseline_path = Path(args.baseline)
    evoke_path = Path(args.evoke)
    baseline_record = _load_record(baseline_path)
    evoke_record = _load_record(evoke_path)

    baseline_responses = _responses_from_record(baseline_record)
    evoke_responses = _responses_from_record(evoke_record)
    failures = compare_responses(
        baseline_responses, evoke_responses, logprob_atol=args.logprob_atol
    )

    evoke_delta = OffloadMetricsSummary.from_dict(evoke_record["metrics_delta"])
    vacuous = (
        evoke_delta.external_prefix_cache_hits <= 0 or evoke_delta.cpu_to_gpu_bytes <= 0
    )

    passed = not failures and not vacuous
    result = {
        "generated_at": datetime.now(UTC).isoformat(),
        "baseline_record": str(baseline_path),
        "evoke_record": str(evoke_path),
        "workload": evoke_record.get("workload"),
        "vacuous_check": {
            "external_prefix_cache_hits_delta": evoke_delta.external_prefix_cache_hits,
            "external_prefix_cache_queries_delta": (
                evoke_delta.external_prefix_cache_queries
            ),
            "cpu_to_gpu_bytes_delta": evoke_delta.cpu_to_gpu_bytes,
            "gpu_to_cpu_bytes_delta": evoke_delta.gpu_to_cpu_bytes,
            "vacuous": vacuous,
        },
        "fidelity": {
            "total_requests": len(baseline_responses),
            "failed": len(failures),
            "passed": len(baseline_responses) - len(failures),
            "failures": [asdict(failure) for failure in failures],
        },
        "passkey_recall": {
            "baseline": baseline_record.get("passkey_recall", {}),
            "evoke": evoke_record.get("passkey_recall", {}),
        },
        "verdict": "PASS" if passed else "FAIL",
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")

    if failures:
        print(
            f"FIDELITY FAIL: {len(failures)} of {len(baseline_responses)} "
            f"requests diverged from baseline",
            file=sys.stderr,
        )
        for failure in failures[:10]:
            print(
                f"  {failure.session_id} turn={failure.turn_index} "
                f"phase={failure.phase}: {failure.reason}",
                file=sys.stderr,
            )

    if vacuous:
        print(
            "GATE FAIL: no restores happened, gate is vacuous: "
            "external_prefix_cache_hits and cpu_to_gpu offload bytes were "
            "both non-positive over the evoke run. Raise GPU pressure "
            "(lower EVOKE_GPU_MEMORY_UTILIZATION) or grow the workload "
            "(--sessions/--growth-turns/--filler-words) and re-record.",
            file=sys.stderr,
        )

    if passed:
        print("GATE PASS")
        return 0
    return 1


def _add_workload_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--growth-turns", type=int, default=3)
    parser.add_argument("--filler-words", type=int, default=220)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--top-logprobs", type=int, default=5)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    record_parser = subparsers.add_parser(
        "record", help="run the workload against a live server and save results"
    )
    record_parser.add_argument("--base-url", default="http://localhost:8000")
    record_parser.add_argument("--model", required=True)
    record_parser.add_argument(
        "--run-label", choices=["baseline", "evoke"], required=True
    )
    record_parser.add_argument("--out", required=True)
    record_parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    record_parser.add_argument("--dry-run", action="store_true")
    _add_workload_args(record_parser)
    record_parser.set_defaults(func=cmd_record)

    compare_parser = subparsers.add_parser(
        "compare", help="diff two recorded runs and emit the gate verdict"
    )
    compare_parser.add_argument("--baseline", required=True)
    compare_parser.add_argument("--evoke", required=True)
    compare_parser.add_argument("--out", required=True)
    compare_parser.add_argument("--logprob-atol", type=float, default=0.05)
    compare_parser.set_defaults(func=cmd_compare)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
