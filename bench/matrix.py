"""Matrix driver: resolves matrix.toml into an ordered list of cells.

Reads the arms x workloads x budgets run matrix from spec
02a-workloads.md section 6 (27 primary cells plus 6 composition-or-fallback
cells, 33 total) and, for each cell, knows which server config (arm plus
budget) it needs. Cells are resolved budget-major within each arm so every
run that shares a server config is contiguous, which is what
``group_by_server`` needs to minimize restarts: one ``vllm serve`` per
distinct (arm, budget) pair, covering every workload before the next
restart.
"""

from __future__ import annotations

import shlex
import tomllib
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Any

from bench.arms import ServeProfile, base_url, serve_command

BUDGET_NA = "NA"


class MatrixError(ValueError):
    pass


@dataclass(frozen=True)
class WorkloadSpec:
    id: str
    kind: str
    options: dict[str, Any]


@dataclass(frozen=True)
class BudgetSpec:
    id: str
    cpu_bytes_to_use: int
    label: str = ""


@dataclass(frozen=True)
class CellRule:
    arm: str
    workloads: tuple[str, ...]
    budgets: tuple[str, ...]
    fallback_arm: str | None = None


@dataclass(frozen=True)
class SmokeTestSpec:
    arm: str
    workload: str
    budget: str
    note: str = ""


@dataclass(frozen=True)
class MatrixConfig:
    trace_id: str
    workloads: dict[str, WorkloadSpec]
    budgets: dict[str, BudgetSpec]
    cells: tuple[CellRule, ...]
    smoke_test: SmokeTestSpec | None
    expected_core_runs: int | None = None


@dataclass(frozen=True)
class ResolvedCell:
    arm: str
    workload: str
    budget: str

    def result_filename(self) -> str:
        return f"{self.arm}_{self.workload}_{self.budget}.json"


@dataclass(frozen=True)
class ServerGroup:
    arm: str
    budget: str
    cells: tuple[ResolvedCell, ...]


def load_matrix(path: Path) -> MatrixConfig:
    raw = tomllib.loads(Path(path).read_text())

    meta = raw.get("meta", {})
    trace_id = meta.get("trace_id", "verdant-session")
    expected_core_runs = meta.get("expected_core_runs")

    workloads = {}
    for entry in raw.get("workloads", []):
        wid = entry["id"]
        options = {k: v for k, v in entry.items() if k not in ("id", "kind")}
        workloads[wid] = WorkloadSpec(id=wid, kind=entry["kind"], options=options)

    budgets = {}
    for entry in raw.get("budgets", []):
        bid = entry["id"]
        budgets[bid] = BudgetSpec(
            id=bid,
            cpu_bytes_to_use=entry["cpu_bytes_to_use"],
            label=entry.get("label", ""),
        )

    cells = []
    for entry in raw.get("cells", []):
        cells.append(
            CellRule(
                arm=entry["arm"],
                workloads=tuple(entry["workloads"]),
                budgets=tuple(entry["budgets"]),
                fallback_arm=entry.get("fallback_arm"),
            )
        )

    smoke_raw = raw.get("smoke_test")
    smoke_test = (
        SmokeTestSpec(
            arm=smoke_raw["arm"],
            workload=smoke_raw["workload"],
            budget=smoke_raw["budget"],
            note=smoke_raw.get("note", ""),
        )
        if smoke_raw
        else None
    )

    return MatrixConfig(
        trace_id=trace_id,
        workloads=workloads,
        budgets=budgets,
        cells=tuple(cells),
        smoke_test=smoke_test,
        expected_core_runs=expected_core_runs,
    )


def resolve_budget(config: MatrixConfig, budget_id: str) -> BudgetSpec | None:
    if budget_id == BUDGET_NA:
        return None
    if budget_id not in config.budgets:
        raise MatrixError(f"unknown budget id {budget_id!r}")
    return config.budgets[budget_id]


def resolve_cells(config: MatrixConfig) -> list[ResolvedCell]:
    resolved: list[ResolvedCell] = []
    for rule in config.cells:
        for workload_id in rule.workloads:
            if workload_id not in config.workloads:
                raise MatrixError(
                    f"cell for arm {rule.arm} references unknown workload "
                    f"{workload_id!r}"
                )
        for budget_id in rule.budgets:
            if budget_id != BUDGET_NA and budget_id not in config.budgets:
                raise MatrixError(
                    f"cell for arm {rule.arm} references unknown budget {budget_id!r}"
                )
        for budget_id in rule.budgets:
            for workload_id in rule.workloads:
                resolved.append(
                    ResolvedCell(arm=rule.arm, workload=workload_id, budget=budget_id)
                )
    return resolved


def group_by_server(cells: list[ResolvedCell]) -> list[ServerGroup]:
    groups: list[ServerGroup] = []
    for (arm, budget), members in groupby(cells, key=lambda c: (c.arm, c.budget)):
        groups.append(ServerGroup(arm=arm, budget=budget, cells=tuple(members)))
    return groups


@dataclass
class DryRunPlan:
    smoke_test: SmokeTestSpec | None
    groups: list[ServerGroup]
    total_cells: int
    expected_core_runs: int | None


def build_dry_run_plan(config: MatrixConfig) -> DryRunPlan:
    cells = resolve_cells(config)
    groups = group_by_server(cells)
    return DryRunPlan(
        smoke_test=config.smoke_test,
        groups=groups,
        total_cells=len(cells),
        expected_core_runs=config.expected_core_runs,
    )


def render_run_cell_command(
    cell: ResolvedCell,
    *,
    server_url: str,
    matrix_path: str,
    results_dir: str,
    model: str,
) -> list[str]:
    # --model is passed explicitly because run-cell's --profile flag
    # defaults to local-2060; a rendered command that omitted the model
    # would request the wrong served_model_name and the server would
    # answer 404 "model not found".
    return [
        "uv",
        "run",
        "python",
        "-m",
        "bench",
        "run-cell",
        "--arm",
        cell.arm,
        "--workload",
        cell.workload,
        "--budget",
        cell.budget,
        "--server-url",
        server_url,
        "--matrix",
        matrix_path,
        "--model",
        model,
        "--out",
        f"{results_dir.rstrip('/')}/{cell.result_filename()}",
    ]


def render_dry_run(
    plan: DryRunPlan,
    config: MatrixConfig,
    profile: ServeProfile,
    *,
    repo_root: str = ".",
    matrix_path: str = "bench/matrix.toml",
    results_dir: str = "bench/results",
) -> str:
    lines: list[str] = []
    url = base_url(profile)

    if plan.smoke_test is not None:
        st = plan.smoke_test
        budget_spec = resolve_budget(config, st.budget)
        cpu_bytes = budget_spec.cpu_bytes_to_use if budget_spec else None
        lines.append("=== smoke test (must pass before composition rows run) ===")
        if st.note:
            lines.append(f"# {st.note}")
        lines.append(
            shlex.join(serve_command(profile, st.arm, cpu_bytes, repo_root=repo_root))
        )
        lines.append(f"# wait for {url}/health")
        smoke_cell = ResolvedCell(arm=st.arm, workload=st.workload, budget=st.budget)
        lines.append(
            shlex.join(
                render_run_cell_command(
                    smoke_cell,
                    server_url=url,
                    matrix_path=matrix_path,
                    results_dir=results_dir,
                    model=profile.served_model_name,
                )
            )
        )
        lines.append("# stop server")
        lines.append("")

    for group_index, group in enumerate(plan.groups, start=1):
        budget_spec = resolve_budget(config, group.budget)
        cpu_bytes = budget_spec.cpu_bytes_to_use if budget_spec else None
        label = budget_spec.label if budget_spec else "N/A"
        lines.append(
            f"=== group {group_index}: arm={group.arm} budget={group.budget} "
            f"({label}), {len(group.cells)} run(s) ==="
        )
        lines.append(
            shlex.join(
                serve_command(profile, group.arm, cpu_bytes, repo_root=repo_root)
            )
        )
        lines.append(f"# wait for {url}/health")
        for cell in group.cells:
            lines.append(
                shlex.join(
                    render_run_cell_command(
                        cell,
                        server_url=url,
                        matrix_path=matrix_path,
                        results_dir=results_dir,
                        model=profile.served_model_name,
                    )
                )
            )
        lines.append("# stop server")
        lines.append("")

    lines.append(f"# total runs: {plan.total_cells}")
    if plan.expected_core_runs is not None:
        status = "OK" if plan.total_cells == plan.expected_core_runs else "MISMATCH"
        lines.append(f"# expected core runs: {plan.expected_core_runs} ({status})")
    return "\n".join(lines)
