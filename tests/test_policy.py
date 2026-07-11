import pytest

pytest.importorskip("vllm")

from offload_harness import ready_block, to_key  # noqa: E402

from evoke_vllm.config import EvokeRequestTags, EvokeScoringConfig  # noqa: E402
from evoke_vllm.policy import EvokeCachePolicy  # noqa: E402


def test_recency_decays_by_half_life():
    policy = EvokeCachePolicy(
        cache_capacity=8,
        scoring_config=EvokeScoringConfig(
            w_recency=1.0, w_reuse=0.0, recency_half_life=4
        ),
    )
    a = to_key(1)
    policy.insert(a, ready_block(0))  # tick advances to 1, a.last_touch = 1
    for i in range(2, 6):  # four more inserts advance tick to 5
        policy.insert(to_key(i), ready_block(i))
    # age(a) = 5 - 1 = 4 == half_life -> factor 0.5
    assert policy._recency(a) == pytest.approx(0.5)


def test_touch_refreshes_recency():
    policy = EvokeCachePolicy(
        cache_capacity=8,
        scoring_config=EvokeScoringConfig(w_recency=1.0, w_reuse=0.0),
    )
    a, b = to_key(1), to_key(2)
    policy.insert(a, ready_block(0))
    policy.insert(b, ready_block(1))
    for _ in range(5):
        policy.touch([b])
    assert policy._recency(a) < policy._recency(b)
    assert policy._score(a) < policy._score(b)


def test_reuse_raises_score_and_saturates():
    counts: dict = {}
    policy = EvokeCachePolicy(
        cache_capacity=8,
        scoring_config=EvokeScoringConfig(w_recency=0.0, w_reuse=1.0),
        reuse_counts=counts,
    )
    a, b = to_key(1), to_key(2)
    policy.insert(a, ready_block(0))
    policy.insert(b, ready_block(1))
    counts[b] = 3
    assert policy._reuse(a) == pytest.approx(0.0)
    assert policy._reuse(b) == pytest.approx(1.0 - 0.5**3)
    assert policy._score(b) > policy._score(a)


def test_source_floor_lifts_decayed_score():
    policy = EvokeCachePolicy(
        cache_capacity=64,
        scoring_config=EvokeScoringConfig(
            w_recency=1.0, w_reuse=0.0, recency_half_life=1
        ),
    )
    a = to_key(1)
    policy.insert(a, ready_block(0))
    for i in range(2, 40):  # age a far past the half-life
        policy.insert(to_key(i), ready_block(i))
    assert policy._recency(a) < 0.6
    policy.apply_request_tags(a, EvokeRequestTags(source_type="system"))
    assert policy._score(a) == pytest.approx(0.6)


def test_priority_multiplies_after_floor():
    policy = EvokeCachePolicy(
        cache_capacity=64,
        scoring_config=EvokeScoringConfig(
            w_recency=1.0, w_reuse=0.0, recency_half_life=1
        ),
    )
    a = to_key(1)
    policy.insert(a, ready_block(0))
    for i in range(2, 40):
        policy.insert(to_key(i), ready_block(i))
    policy.apply_request_tags(a, EvokeRequestTags(source_type="system", priority=2.0))
    assert policy._score(a) == pytest.approx(1.2)  # floor 0.6 * priority 2.0


def test_system_floor_outlives_document_under_pressure():
    policy = EvokeCachePolicy(
        cache_capacity=8,
        scoring_config=EvokeScoringConfig(
            w_recency=1.0, w_reuse=0.0, recency_half_life=1
        ),
    )
    sys_key, doc_key, filler = to_key(1), to_key(2), to_key(3)
    policy.insert(sys_key, ready_block(0))
    policy.insert(doc_key, ready_block(1))
    policy.insert(filler, ready_block(2))
    policy.apply_request_tags(sys_key, EvokeRequestTags(source_type="system"))
    policy.apply_request_tags(doc_key, EvokeRequestTags(source_type="document"))
    for _ in range(20):  # age sys and doc; keep filler fresh
        policy.touch([filler])
    evicted = policy.evict(1, set())
    assert evicted is not None
    assert [key for key, _ in evicted] == [doc_key]


def test_floor_preserves_reuse_ranking_within_a_role():
    # A hard max() clamp scores every aged same-role block exactly at the
    # floor, so eviction ties everywhere and the stable sort degenerates to
    # insertion order, indistinguishable from FIFO/LRU (observed on the
    # 4070 Ti skew benchmark: A2 byte-identical to A1). The floor must
    # guarantee a minimum without erasing within-role ordering.
    counts: dict = {}
    policy = EvokeCachePolicy(
        cache_capacity=64,
        scoring_config=EvokeScoringConfig(recency_half_life=1),
        reuse_counts=counts,
    )
    hot, cold = to_key(1), to_key(2)
    policy.insert(hot, ready_block(0))
    policy.insert(cold, ready_block(1))
    policy.apply_request_tags(hot, EvokeRequestTags(source_type="user"))
    policy.apply_request_tags(cold, EvokeRequestTags(source_type="user"))
    counts[hot] = 6
    counts[cold] = 1
    filler = [to_key(i) for i in range(3, 40)]
    for idx, key in enumerate(filler):  # age hot and cold far past half-life
        policy.insert(key, ready_block(idx + 2))
    assert policy._score(hot) > policy._score(cold)
    evicted = policy.evict(1, set(filler))
    assert evicted is not None
    assert [key for key, _ in evicted] == [cold]


def test_evict_returns_exactly_n_lowest_score():
    policy = EvokeCachePolicy(
        cache_capacity=8,
        scoring_config=EvokeScoringConfig(w_recency=1.0, w_reuse=0.0),
    )
    keys = [to_key(i) for i in range(4)]
    for idx, key in enumerate(keys):
        policy.insert(key, ready_block(idx))
    # insertion order is oldest-first; the two oldest have the lowest recency
    evicted = policy.evict(2, set())
    assert evicted is not None
    assert len(evicted) == 2
    assert {key for key, _ in evicted} == {keys[0], keys[1]}
    assert policy.get(keys[2]) is not None
    assert policy.get(keys[3]) is not None


def test_evict_skips_protected():
    policy = EvokeCachePolicy(cache_capacity=8)
    a, b, c = to_key(1), to_key(2), to_key(3)
    for idx, key in enumerate((a, b, c)):
        policy.insert(key, ready_block(idx))
    evicted = policy.evict(2, {a})
    assert evicted is not None
    keys = {key for key, _ in evicted}
    assert a not in keys
    assert keys == {b, c}
    assert policy.get(a) is not None


def test_evict_skips_referenced_blocks():
    policy = EvokeCachePolicy(cache_capacity=8)
    a, b, c = to_key(1), to_key(2), to_key(3)
    policy.insert(a, ready_block(0, ref_cnt=0))
    policy.insert(b, ready_block(1, ref_cnt=1))  # in-flight read, not evictable
    policy.insert(c, ready_block(2, ref_cnt=0))
    evicted = policy.evict(2, set())
    assert evicted is not None
    assert {key for key, _ in evicted} == {a, c}
    assert policy.get(b) is not None


def test_evict_is_atomic_when_it_cannot_satisfy_n():
    policy = EvokeCachePolicy(cache_capacity=8)
    a, b = to_key(1), to_key(2)
    policy.insert(a, ready_block(0, ref_cnt=0))
    policy.insert(b, ready_block(1, ref_cnt=1))  # only a is evictable
    before = dict(policy.blocks)
    assert policy.evict(2, set()) is None
    assert policy.blocks == before  # no mutation on failure


def test_evict_zero_returns_empty_list():
    policy = EvokeCachePolicy(cache_capacity=8)
    policy.insert(to_key(1), ready_block(0))
    assert policy.evict(0, set()) == []


def test_mark_non_evictable_pins_and_mark_evictable_unpins():
    policy = EvokeCachePolicy(cache_capacity=8)
    a = to_key(1)
    policy.insert(a, ready_block(0, ref_cnt=0))
    policy.mark_non_evictable(a)
    assert policy.evict(1, set()) is None  # pinned despite ref_cnt == 0
    policy.mark_evictable(a)
    evicted = policy.evict(1, set())
    assert evicted is not None
    assert [key for key, _ in evicted] == [a]


def test_clear_removes_all_blocks():
    policy = EvokeCachePolicy(cache_capacity=8)
    policy.insert(to_key(1), ready_block(0))
    policy.insert(to_key(2), ready_block(1))
    policy.clear()
    assert policy.get(to_key(1)) is None
    assert policy.evict(1, set()) is None
