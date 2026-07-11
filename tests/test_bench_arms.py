import json
from pathlib import Path

import pytest

from bench.arms import (
    ProfileError,
    ServeProfile,
    base_url,
    kv_transfer_config_for_arm,
    load_profile,
    parse_env_file,
    serve_command,
)

PROFILES_DIR = Path(__file__).resolve().parents[1] / "profiles"


@pytest.fixture
def profile():
    return ServeProfile(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        served_model_name="qwen2.5-1.5b-instruct",
        host="0.0.0.0",
        port=8000,
        dtype="float16",
        max_model_len=4096,
        block_size=16,
        gpu_memory_utilization=0.6,
    )


def test_real_profiles_parse_cleanly():
    for env_file in PROFILES_DIR.glob("*.env"):
        loaded = load_profile(env_file)
        assert loaded.model
        assert loaded.port == 8000


def test_wsl2_profile_carries_the_benchmark_pressure_recipe():
    # The 2026-07-11 matrix ran at 0.68 / threshold 2 and was vacuous:
    # the 26k-token pool never evicted, and threshold 2 stores a block
    # only on its second computation, so stores lag eviction. The
    # benchmark must inherit the gate's validated sizing or offload
    # traffic never flows.
    loaded = load_profile(PROFILES_DIR / "wsl2-4070ti.env")
    assert loaded.gpu_memory_utilization == 0.65
    assert loaded.store_threshold == 1


def test_parse_env_file_strips_quotes_and_comments(tmp_path):
    env_path = tmp_path / "x.env"
    env_path.write_text('# a comment\nEVOKE_MODEL="some/model"\nEVOKE_PORT=8000\n')
    env = parse_env_file(env_path)
    assert env == {"EVOKE_MODEL": "some/model", "EVOKE_PORT": "8000"}


def test_offload_block_size_must_be_multiple_of_block_size():
    with pytest.raises(ProfileError):
        ServeProfile(
            model="m",
            served_model_name="m",
            host="0.0.0.0",
            port=8000,
            dtype="float16",
            max_model_len=4096,
            block_size=16,
            gpu_memory_utilization=0.6,
            offload_block_size=50,
        )


def test_arm_a0_has_no_kv_transfer_config(profile):
    assert kv_transfer_config_for_arm("A0", profile, None) is None


def test_arm_a1_is_stock_offloading_connector_no_spec_name(profile):
    config = kv_transfer_config_for_arm("A1", profile, 1_000_000)
    assert config["kv_connector"] == "OffloadingConnector"
    assert "spec_name" not in config["kv_connector_extra_config"]
    assert config["kv_connector_extra_config"]["cpu_bytes_to_use"] == 1_000_000


def test_arm_a2_selects_evoke_spec(profile):
    config = kv_transfer_config_for_arm("A2", profile, 2_000_000)
    extra = config["kv_connector_extra_config"]
    assert extra["spec_name"] == "EvokeOffloadingSpec"
    assert extra["spec_module_path"] == "evoke_vllm.spec"
    assert "evoke" in extra


def test_arm_a3_is_multiconnector_evoke_first_then_lmcache(profile):
    config = kv_transfer_config_for_arm("A3", profile, 3_000_000)
    assert config["kv_connector"] == "MultiConnector"
    connectors = config["kv_connector_extra_config"]["connectors"]
    assert len(connectors) == 2
    assert (
        connectors[0]["kv_connector_extra_config"]["spec_name"] == "EvokeOffloadingSpec"
    )
    assert connectors[1]["kv_connector"] == "LMCacheConnectorV1"


def test_arm_a4_is_lmcache_alone(profile):
    config = kv_transfer_config_for_arm("A4", profile, None)
    assert config == {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}


def test_budgeted_arm_without_cpu_bytes_raises(profile):
    with pytest.raises(ValueError):
        kv_transfer_config_for_arm("A1", profile, None)


def test_unknown_arm_raises(profile):
    with pytest.raises(ValueError):
        kv_transfer_config_for_arm("A99", profile, None)


def test_serve_command_includes_kv_transfer_config_json(profile):
    cmd = serve_command(profile, "A2", 1_500_000_000, repo_root="/repo")
    assert cmd[:2] == ["uv", "run"]
    assert "vllm" in cmd
    assert "serve" in cmd
    idx = cmd.index("--kv-transfer-config")
    payload = json.loads(cmd[idx + 1])
    assert payload["kv_connector_extra_config"]["cpu_bytes_to_use"] == 1_500_000_000


def test_serve_command_baseline_omits_kv_transfer_config(profile):
    cmd = serve_command(profile, "A0", None)
    assert "--kv-transfer-config" not in cmd


def test_base_url_maps_wildcard_host_to_loopback(profile):
    assert base_url(profile) == "http://127.0.0.1:8000"
