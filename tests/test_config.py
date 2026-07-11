import pytest

from evoke_vllm.config import (
    DEFAULT_SOURCE_FLOORS,
    SOURCE_ASSISTANT,
    SOURCE_SYSTEM,
    SOURCE_USER,
    EvokeRequestTags,
    EvokeScoringConfig,
)

# No vLLM import anywhere in this file, so that this lane keeps passing in
# environments where the ~280MB vllm wheel is unavailable or impractical to
# fetch (see tests/test_smoke.py for the vLLM-gated lane).


def test_scoring_config_defaults():
    config = EvokeScoringConfig()
    assert config.w_recency == pytest.approx(0.5)
    assert config.w_reuse == pytest.approx(0.5)
    assert config.w_attention == pytest.approx(0.0)
    assert config.w_coherence == pytest.approx(0.0)
    assert config.recency_half_life == 64
    assert config.source_floors == DEFAULT_SOURCE_FLOORS


def test_scoring_config_source_floors_are_independent_copies():
    first = EvokeScoringConfig()
    second = EvokeScoringConfig()
    first.source_floors[SOURCE_USER] = 0.9
    assert second.source_floors[SOURCE_USER] == DEFAULT_SOURCE_FLOORS[SOURCE_USER]


def test_default_source_floors_cover_backbone_roles():
    assert SOURCE_SYSTEM in DEFAULT_SOURCE_FLOORS
    assert SOURCE_USER in DEFAULT_SOURCE_FLOORS
    assert SOURCE_ASSISTANT in DEFAULT_SOURCE_FLOORS


def test_request_tags_defaults():
    tags = EvokeRequestTags()
    assert tags.source_type is None
    assert tags.priority == pytest.approx(1.0)
    assert tags.evoke_session is None


def test_scoring_config_from_extra_config_reads_evoke_subkey():
    config = EvokeScoringConfig.from_extra_config(
        {
            "evoke": {
                "w_recency": 0.7,
                "w_reuse": 0.3,
                "recency_half_life": 32,
                "source_floors": {"system": 0.8},
            }
        }
    )
    assert config.w_recency == pytest.approx(0.7)
    assert config.w_reuse == pytest.approx(0.3)
    assert config.recency_half_life == 32
    assert config.source_floors == {"system": 0.8}


def test_scoring_config_from_extra_config_absent_evoke_uses_defaults():
    config = EvokeScoringConfig.from_extra_config({})
    assert config.w_recency == pytest.approx(0.5)
    assert config.w_reuse == pytest.approx(0.5)
    assert config.source_floors == DEFAULT_SOURCE_FLOORS


def test_scoring_config_from_extra_config_ignores_non_dict_subkey():
    config = EvokeScoringConfig.from_extra_config({"evoke": "not-a-dict"})
    assert config.w_recency == pytest.approx(0.5)


def test_env_overrides_take_precedence(monkeypatch):
    monkeypatch.setenv("EVOKE_W_RECENCY", "0.9")
    monkeypatch.setenv("EVOKE_RECENCY_HALF_LIFE", "16")
    monkeypatch.setenv("EVOKE_FLOOR_DOCUMENT", "0.4")
    config = EvokeScoringConfig.from_extra_config({"evoke": {"w_recency": 0.2}})
    assert config.w_recency == pytest.approx(0.9)
    assert config.recency_half_life == 16
    assert config.source_floors["document"] == pytest.approx(0.4)


def test_apply_env_overrides_standalone(monkeypatch):
    monkeypatch.setenv("EVOKE_W_REUSE", "0.1")
    config = EvokeScoringConfig.apply_env_overrides(EvokeScoringConfig())
    assert config.w_reuse == pytest.approx(0.1)


def test_request_tags_from_kv_transfer_params_reads_evoke_subkey():
    tags = EvokeRequestTags.from_kv_transfer_params(
        {
            "evoke": {
                "source_type": "system",
                "priority": 2.5,
                "evoke_session": "abc",
            },
            "max_offload_tokens": 100,
        }
    )
    assert tags.source_type == "system"
    assert tags.priority == pytest.approx(2.5)
    assert tags.evoke_session == "abc"


def test_request_tags_none_degrades_to_defaults():
    tags = EvokeRequestTags.from_kv_transfer_params(None)
    assert tags.source_type is None
    assert tags.priority == pytest.approx(1.0)
    assert tags.evoke_session is None


def test_request_tags_missing_evoke_subkey_degrades():
    tags = EvokeRequestTags.from_kv_transfer_params({"max_offload_tokens": 100})
    assert tags.source_type is None
    assert tags.priority == pytest.approx(1.0)
    assert tags.evoke_session is None
