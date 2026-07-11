import json
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "scripts" / "serve.sh"


def _run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )


def _extract_json_block(stdout: str) -> dict:
    lines = stdout.splitlines()
    start = lines.index("# kv-transfer-config JSON:") + 1
    return json.loads(lines[start])


def test_dry_run_evoke_mode_prints_valid_kv_transfer_config():
    result = _run("--profile", "local-2060", "--dry-run")
    assert result.returncode == 0, result.stderr
    config = _extract_json_block(result.stdout)
    assert config["kv_connector"] == "OffloadingConnector"
    assert config["kv_role"] == "kv_both"
    extra = config["kv_connector_extra_config"]
    assert extra["spec_name"] == "EvokeOffloadingSpec"
    assert extra["spec_module_path"] == "evoke_vllm.spec"
    assert extra["cpu_bytes_to_use"] == 4294967296
    assert extra["offload_prompt_only"] is True
    assert "evoke" in extra


def test_dry_run_evoke_mode_command_includes_kv_transfer_config_flag():
    result = _run("--profile", "local-2060", "--dry-run")
    assert "--kv-transfer-config" in result.stdout
    assert "Qwen/Qwen2.5-1.5B-Instruct" in result.stdout


def test_dry_run_baseline_mode_omits_kv_transfer_config():
    result = _run("--profile", "local-2060", "--baseline", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "# mode: baseline (no connector)" in result.stdout
    assert "--kv-transfer-config" not in result.stdout


def test_dry_run_wsl2_profile_uses_fp8_model():
    result = _run("--profile", "wsl2-4070ti", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic" in result.stdout


def test_dry_run_extra_arg_is_appended():
    result = _run(
        "--profile", "local-2060", "--dry-run", "--extra-arg", "--enforce-eager"
    )
    assert result.returncode == 0, result.stderr
    assert "--enforce-eager" in result.stdout


def test_missing_profile_fails_clearly():
    result = _run("--profile", "does-not-exist", "--dry-run")
    assert result.returncode == 1
    assert "profile not found" in result.stderr


def test_missing_required_vars_without_profile_fails_clearly():
    result = _run("--dry-run", env={"PATH": "/usr/bin:/bin"})
    assert result.returncode == 1
    assert "missing required config" in result.stderr


def test_offload_block_size_not_a_multiple_of_block_size_fails_fast():
    env = {
        "PATH": "/usr/bin:/bin",
        "EVOKE_MODEL": "x",
        "EVOKE_HOST": "0.0.0.0",
        "EVOKE_PORT": "8000",
        "EVOKE_DTYPE": "auto",
        "EVOKE_MAX_MODEL_LEN": "1024",
        "EVOKE_BLOCK_SIZE": "16",
        "EVOKE_GPU_MEMORY_UTILIZATION": "0.5",
        "EVOKE_CPU_BYTES_TO_USE": "1000",
        "EVOKE_OFFLOAD_BLOCK_SIZE": "50",
        "EVOKE_STORE_THRESHOLD": "2",
    }
    result = _run("--dry-run", env=env)
    assert result.returncode == 1
    assert "must be a multiple of" in result.stderr
