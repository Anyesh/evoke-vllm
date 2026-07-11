import pytest

pytest.importorskip("vllm")

from offload_harness import make_req_context, to_key, to_keys  # noqa: E402

from evoke_vllm.config import EvokeScoringConfig  # noqa: E402
from evoke_vllm.manager import EvokeOffloadingManager  # noqa: E402
from evoke_vllm.policy import EvokeCachePolicy  # noqa: E402


def _store(manager, int_hashes, kv_transfer_params=None):
    keys = to_keys(int_hashes)
    ctx = make_req_context(kv_transfer_params=kv_transfer_params)
    out = manager.prepare_store(keys, ctx)
    assert out is not None
    manager.complete_store(keys, ctx, success=True)
    return keys


def test_manager_constructs_evoke_policy():
    manager = EvokeOffloadingManager(num_blocks=4)
    assert isinstance(manager._policy, EvokeCachePolicy)


def test_lookup_maintains_unconditional_reuse_counts():
    manager = EvokeOffloadingManager(num_blocks=4)
    ctx = make_req_context()
    key = to_key(1)
    # store_threshold default keeps the stock gated tracker off ...
    assert manager.counts is None
    # ... while the reuse tracker counts every lookup regardless.
    assert manager.lookup(key, ctx) is False
    assert manager.reuse_counts[key] == 1
    manager.lookup(key, ctx)
    manager.lookup(key, ctx)
    assert manager.reuse_counts[key] == 3


def test_reuse_counts_reach_policy_scoring():
    manager = EvokeOffloadingManager(num_blocks=4)
    ctx = make_req_context()
    _store(manager, [1, 2])
    for _ in range(3):
        manager.lookup(to_key(1), ctx)
    assert manager._policy._reuse(to_key(1)) > manager._policy._reuse(to_key(2))


def test_prepare_store_threads_tags_to_policy():
    manager = EvokeOffloadingManager(num_blocks=4)
    keys = _store(
        manager,
        [1, 2],
        kv_transfer_params={
            "evoke": {
                "source_type": "system",
                "priority": 2.0,
                "evoke_session": "s1",
            }
        },
    )
    for key in keys:
        meta = manager._policy.meta[key]
        assert meta.source_type == "system"
        assert meta.priority == pytest.approx(2.0)


def test_untagged_store_degrades_cleanly():
    manager = EvokeOffloadingManager(num_blocks=4)
    _store(manager, [1])
    meta = manager._policy.meta[to_key(1)]
    assert meta.source_type is None
    assert meta.priority == pytest.approx(1.0)


def test_tag_threading_drives_eviction_end_to_end():
    manager = EvokeOffloadingManager(
        num_blocks=3,
        scoring_config=EvokeScoringConfig(
            w_recency=1.0, w_reuse=0.0, recency_half_life=1
        ),
    )
    ctx = make_req_context()
    _store(manager, [1], {"evoke": {"source_type": "document"}})
    _store(manager, [2], {"evoke": {"source_type": "system"}})
    _store(manager, [3])  # untagged filler
    for _ in range(10):  # age the document and system blocks; keep filler fresh
        manager.touch(to_keys([3]), ctx)
    out = manager.prepare_store(to_keys([4]), ctx)
    assert out is not None
    assert out.evicted_keys == to_keys([1])  # document evicted, system survives
    manager.complete_store(to_keys([4]), ctx, success=True)
    assert manager._policy.get(to_key(2)) is not None


def test_reuse_frequency_drives_eviction_end_to_end():
    manager = EvokeOffloadingManager(
        num_blocks=2,
        scoring_config=EvokeScoringConfig(w_recency=0.0, w_reuse=1.0),
    )
    ctx = make_req_context()
    _store(manager, [1, 2])
    for _ in range(4):  # block 1 is reused, block 2 is not
        manager.lookup(to_key(1), ctx)
    out = manager.prepare_store(to_keys([3]), ctx)
    assert out is not None
    assert out.evicted_keys == to_keys([2])
