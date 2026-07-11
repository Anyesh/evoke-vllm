import json
import shlex
from pathlib import Path

import pytest

from bench.arms import ServeProfile
from bench.matrix import (
    BUDGET_NA,
    MatrixError,
    build_dry_run_plan,
    group_by_server,
    load_matrix,
    render_dry_run,
    resolve_budget,
    resolve_cells,
)
from bench.workloads.factory import mab_config_from_spec
from bench.workloads.memory_agent_bench import ISSUE_ORDER_ROUND_ROBIN

REAL_MATRIX_PATH = Path(__file__).resolve().parents[1] / "bench" / "matrix.toml"


@pytest.fixture
def profile():
    return ServeProfile(
        model="test/model",
        served_model_name="test-model",
        host="0.0.0.0",
        port=8000,
        dtype="float16",
        max_model_len=4096,
        block_size=16,
        gpu_memory_utilization=0.6,
    )


def test_real_matrix_resolves_to_33_core_runs():
    config = load_matrix(REAL_MATRIX_PATH)
    cells = resolve_cells(config)
    assert len(cells) == 33
    assert config.expected_core_runs == 33


def test_real_matrix_arm_breakdown():
    config = load_matrix(REAL_MATRIX_PATH)
    cells = resolve_cells(config)
    by_arm: dict[str, int] = {}
    for cell in cells:
        by_arm[cell.arm] = by_arm.get(cell.arm, 0) + 1
    assert by_arm == {"A0": 3, "A1": 12, "A2": 12, "A3": 6}


def test_real_matrix_groups_minimize_restarts():
    config = load_matrix(REAL_MATRIX_PATH)
    cells = resolve_cells(config)
    groups = group_by_server(cells)
    assert len(groups) == 11
    for group in groups:
        assert len({cell.workload for cell in group.cells}) == len(group.cells)


def test_real_matrix_has_smoke_test_for_a3():
    config = load_matrix(REAL_MATRIX_PATH)
    assert config.smoke_test is not None
    assert config.smoke_test.arm == "A3"


def test_real_matrix_w1_w2_issue_round_robin():
    config = load_matrix(REAL_MATRIX_PATH)
    for workload_id in ("W1", "W2"):
        mab_config = mab_config_from_spec(config.workloads[workload_id])
        assert mab_config.issue_order == ISSUE_ORDER_ROUND_ROBIN


def test_resolve_budget_na_is_none():
    config = load_matrix(REAL_MATRIX_PATH)
    assert resolve_budget(config, BUDGET_NA) is None
    assert resolve_budget(config, "B2").cpu_bytes_to_use == 1610612736


def test_resolve_budget_unknown_raises():
    config = load_matrix(REAL_MATRIX_PATH)
    with pytest.raises(MatrixError):
        resolve_budget(config, "B99")


def test_dry_run_plan_matches_resolved_cells():
    config = load_matrix(REAL_MATRIX_PATH)
    plan = build_dry_run_plan(config)
    assert plan.total_cells == 33
    assert plan.expected_core_runs == 33
    assert sum(len(g.cells) for g in plan.groups) == 33


def test_render_dry_run_contains_every_arm_and_flags_ok(profile):
    config = load_matrix(REAL_MATRIX_PATH)
    plan = build_dry_run_plan(config)
    text = render_dry_run(plan, config, profile)
    for arm in ("A0", "A1", "A2", "A3"):
        assert f"arm={arm}" in text
    assert "total runs: 33" in text
    assert "(OK)" in text
    assert "smoke test" in text


def test_render_dry_run_commands_are_shell_safe(profile):
    # The dry-run plan is the copy-paste script for the GPU box, so every
    # printed command must survive a shell round trip. The first real
    # sitting failed here: the kv-transfer-config JSON was printed bare,
    # bash stripped its inner double quotes, and every connector arm's
    # server died on "--kv-transfer-config: 1 validation error".
    config = load_matrix(REAL_MATRIX_PATH)
    plan = build_dry_run_plan(config)
    text = render_dry_run(plan, config, profile)
    serve_lines = [line for line in text.splitlines() if "vllm serve" in line]
    assert serve_lines
    saw_kv_config = False
    for line in serve_lines:
        argv = shlex.split(line)
        if "--kv-transfer-config" in argv:
            saw_kv_config = True
            payload = argv[argv.index("--kv-transfer-config") + 1]
            json.loads(payload)
    assert saw_kv_config


def test_render_dry_run_cell_commands_carry_the_profiles_model(profile):
    # run-cell's --profile flag defaults to local-2060, so a rendered
    # command that omits the model requests the wrong served_model_name
    # and the server answers 404 "model not found". The second real
    # sitting failed exactly this way: an A3 server serving the 7B FP8
    # model got asked for qwen2.5-1.5b-instruct.
    config = load_matrix(REAL_MATRIX_PATH)
    plan = build_dry_run_plan(config)
    text = render_dry_run(plan, config, profile)
    cell_lines = [line for line in text.splitlines() if "run-cell" in line]
    assert cell_lines
    for line in cell_lines:
        argv = shlex.split(line)
        assert "--model" in argv
        assert argv[argv.index("--model") + 1] == profile.served_model_name


def test_cell_rule_referencing_unknown_workload_raises(tmp_path, profile):
    bad_toml = tmp_path / "bad.toml"
    bad_toml.write_text(
        """
[[workloads]]
id = "W1"
kind = "verdant_replay"
trace_path = "x"

[[cells]]
arm = "A0"
workloads = ["W_MISSING"]
budgets = ["NA"]
"""
    )
    config = load_matrix(bad_toml)
    with pytest.raises(MatrixError):
        resolve_cells(config)
