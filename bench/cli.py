"""Command-line entry point: ``python -m bench <matrix|run-cell|prefetch>``.

``matrix --dry-run`` prints the resolved run matrix and the exact serve and
run-cell commands, grouped to minimize server restarts, with no server, no
GPU, and no network required. ``run-cell`` and ``prefetch`` do real IO and
are meant for the GPU box (or a connected machine, for prefetch).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

from bench.arms import load_profile
from bench.matrix import build_dry_run_plan, load_matrix, render_dry_run
from bench.runner import run_cell
from bench.workloads.factory import build_workload, mab_config_from_spec
from bench.workloads.memory_agent_bench import fetch_rows

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
DEFAULT_MATRIX_PATH = BENCH_DIR / "matrix.toml"
DEFAULT_RESULTS_DIR = BENCH_DIR / "results"
DEFAULT_PROFILES_DIR = REPO_ROOT / "profiles"


def _resolve_profile_path(profile: str) -> Path:
    candidate = Path(profile)
    if candidate.exists():
        return candidate
    return DEFAULT_PROFILES_DIR / f"{profile}.env"


def cmd_matrix(args: argparse.Namespace) -> int:
    matrix_path = Path(args.matrix)
    config = load_matrix(matrix_path)
    profile = load_profile(_resolve_profile_path(args.profile))
    plan = build_dry_run_plan(config)

    if not args.dry_run:
        print(
            "real orchestration (starting/stopping vllm serve for each "
            "group) is not implemented in this CPU-only harness; rerun with "
            "--dry-run and copy the printed commands onto the GPU box, or "
            "run `run-cell` directly against an already-running server.",
            file=sys.stderr,
        )
        return 1

    print(
        render_dry_run(
            plan,
            config,
            profile,
            repo_root=str(REPO_ROOT),
            matrix_path=str(matrix_path),
            results_dir=args.results_dir,
        )
    )
    return 0


def cmd_run_cell(args: argparse.Namespace) -> int:
    config = load_matrix(Path(args.matrix))
    spec = config.workloads.get(args.workload)
    if spec is None:
        print(f"unknown workload id {args.workload!r}", file=sys.stderr)
        return 2

    model = args.model
    if model is None:
        profile = load_profile(_resolve_profile_path(args.profile))
        model = profile.served_model_name

    workload = build_workload(spec, config.trace_id)
    out_path = Path(args.out) if args.out else None

    with httpx.Client(base_url=args.server_url, timeout=args.timeout) as client:
        result = run_cell(
            client,
            arm=args.arm,
            workload=workload,
            budget=args.budget,
            server_url=args.server_url,
            model=model,
            out_path=out_path,
        )

    print(json.dumps(result.as_dict(), indent=2))
    return 0


def cmd_prefetch(args: argparse.Namespace) -> int:
    config = load_matrix(Path(args.matrix))
    wanted_ids = args.workload or [
        spec.id
        for spec in config.workloads.values()
        if spec.kind == "memory_agent_bench"
    ]

    exit_code = 0
    for workload_id in wanted_ids:
        spec = config.workloads.get(workload_id)
        if spec is None or spec.kind != "memory_agent_bench":
            print(
                f"skip {workload_id}: not a memory_agent_bench workload",
                file=sys.stderr,
            )
            exit_code = 1
            continue
        mab_config = mab_config_from_spec(spec)
        sources = ", ".join(mab_config.sources)
        print(f"prefetching {workload_id} ({mab_config.track}: {sources})")
        rows = fetch_rows(mab_config)
        print(f"  cached {len(rows)} rows")
    return exit_code


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    matrix_parser = subparsers.add_parser(
        "matrix", help="resolve and print (or, on the GPU box, execute) the run matrix"
    )
    matrix_parser.add_argument("--matrix", default=str(DEFAULT_MATRIX_PATH))
    matrix_parser.add_argument("--profile", default="local-2060")
    matrix_parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    matrix_parser.add_argument("--dry-run", action="store_true")
    matrix_parser.set_defaults(func=cmd_matrix)

    run_cell_parser = subparsers.add_parser(
        "run-cell", help="run one (arm, workload, budget) cell against a live server"
    )
    run_cell_parser.add_argument("--matrix", default=str(DEFAULT_MATRIX_PATH))
    run_cell_parser.add_argument("--arm", required=True)
    run_cell_parser.add_argument("--workload", required=True)
    run_cell_parser.add_argument("--budget", required=True)
    run_cell_parser.add_argument("--server-url", required=True)
    run_cell_parser.add_argument("--profile", default="local-2060")
    run_cell_parser.add_argument("--model", default=None)
    run_cell_parser.add_argument("--out", default=None)
    run_cell_parser.add_argument("--timeout", type=float, default=120.0)
    run_cell_parser.set_defaults(func=cmd_run_cell)

    prefetch_parser = subparsers.add_parser(
        "prefetch", help="warm the HF dataset cache for memory_agent_bench workloads"
    )
    prefetch_parser.add_argument("--matrix", default=str(DEFAULT_MATRIX_PATH))
    prefetch_parser.add_argument("--workload", action="append", default=None)
    prefetch_parser.set_defaults(func=cmd_prefetch)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return args.func(args)
