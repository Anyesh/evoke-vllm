import pytest

pytest.importorskip("vllm")

from vllm.v1.kv_offload.base import OffloadingKVEventsConfig  # noqa: E402

from evoke_vllm.config import EVOKE_EXTRA_CONFIG_KEY  # noqa: E402
from evoke_vllm.manager import EvokeOffloadingManager  # noqa: E402
from evoke_vllm.spec import EvokeOffloadingSpec  # noqa: E402


def _make_spec(extra_config):
    # CPUOffloadingSpec.__init__ needs a full VllmConfig + KVCacheConfig to run;
    # get_manager only reads these four attributes, so set them directly to test
    # the manager-selection wiring without booting an engine.
    spec = object.__new__(EvokeOffloadingSpec)
    spec.num_blocks = 4
    spec.extra_config = extra_config
    spec.kv_events_config = OffloadingKVEventsConfig(
        enable_kv_cache_events=False, self_describing_kv_events=False
    )
    spec._manager = None
    return spec


def test_get_manager_returns_evoke_manager_with_parsed_config():
    spec = _make_spec(
        {"cpu_bytes_to_use": 1, EVOKE_EXTRA_CONFIG_KEY: {"w_recency": 0.7}}
    )
    manager = spec.get_manager()
    assert isinstance(manager, EvokeOffloadingManager)
    assert manager.scoring_config.w_recency == pytest.approx(0.7)


def test_get_manager_is_cached():
    spec = _make_spec({"cpu_bytes_to_use": 1})
    assert spec.get_manager() is spec.get_manager()


def test_get_manager_passes_store_threshold_and_tracker_size():
    spec = _make_spec(
        {"cpu_bytes_to_use": 1, "store_threshold": 3, "max_tracker_size": 128}
    )
    manager = spec.get_manager()
    assert manager.store_threshold == 3
    assert manager.max_tracker_size == 128
