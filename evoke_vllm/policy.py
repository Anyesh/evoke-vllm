"""EVOKE eviction policy: recency, reuse, and client-supplied structure.

Scores each offloaded block from the signals honestly available at
``CachePolicy`` and ``CPUOffloadingManager`` scope on stock vLLM (design spec
01a sections 1-2): recency decay since last touch, a reuse-count proxy fed by
the manager, and a source-role floor plus priority multiplier sourced from
per-request tags. Attention mass and embedding coherence are not scored in
P1; there is no stock signal for either. ``EvokeScoringConfig`` carries
``w_attention`` and ``w_coherence`` at zero as forward-declared, RFC-track
knobs, but ``_score`` does not read them: the P1 score is a weighted sum of
recency and reuse, lifted to the source-role floor, then multiplied by
priority. ``evict`` returns exactly ``n`` blocks with ``ref_cnt == 0`` that
are neither ``protected`` nor pinned, atomically (lowest score first).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy

from evoke_vllm.config import EvokeRequestTags, EvokeScoringConfig


@dataclass
class EvokeBlockMeta:
    last_touch_tick: int
    source_type: str | None = None
    priority: float = 1.0
    # Pinned mirrors ref_cnt > 0 via the manager's mark_(non_)evictable hooks;
    # a pinned block is never an eviction candidate even at ref_cnt 0.
    pinned: bool = False


class EvokeCachePolicy(CachePolicy):
    def __init__(
        self,
        cache_capacity: int,
        scoring_config: EvokeScoringConfig | None = None,
        reuse_counts: dict[OffloadKey, int] | None = None,
    ) -> None:
        self.cache_capacity = cache_capacity
        self.config = scoring_config or EvokeScoringConfig()
        self.blocks: dict[OffloadKey, BlockStatus] = {}
        self.meta: dict[OffloadKey, EvokeBlockMeta] = {}
        # Reuse counts are owned by the manager and shared by reference so the
        # score can read a signal the CachePolicy ABC never receives; absent a
        # manager (unit tests) this is a private dict.
        self._reuse_counts: dict[OffloadKey, int] = (
            reuse_counts if reuse_counts is not None else {}
        )
        self._tick = 0

    def _recency(self, key: OffloadKey) -> float:
        age = self._tick - self.meta[key].last_touch_tick
        return 0.5 ** (age / max(1, self.config.recency_half_life))

    def _reuse(self, key: OffloadKey) -> float:
        # Saturating map of the unbounded hit count into [0, 1) so the reuse
        # term is commensurable with recency and the source floor.
        count = self._reuse_counts.get(key, 0)
        return 1.0 - 0.5**count

    def _score(self, key: OffloadKey) -> float:
        meta = self.meta[key]
        raw = self.config.w_recency * self._recency(
            key
        ) + self.config.w_reuse * self._reuse(key)
        if meta.source_type is not None:
            floor = self.config.source_floors.get(meta.source_type, 0.0)
            # The floor must stay a guaranteed minimum without flattening
            # ranking: a hard max() clamp scores every aged same-role block
            # exactly at the floor (the dynamic part of an aged block cannot
            # exceed w_reuse, which sits below the default floors), so
            # eviction ties everywhere and the stable sort degenerates to
            # insertion order, i.e. FIFO. Observed on the 4070 Ti skew
            # benchmark as A2 byte-identical to stock LRU. Rescaling keeps
            # the guarantee and stays strictly monotonic in raw.
            raw = floor + (1.0 - floor) * raw
        return raw * meta.priority

    def get(self, key: OffloadKey) -> BlockStatus | None:
        return self.blocks.get(key)

    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        self._tick += 1
        self.blocks[key] = block
        self.meta[key] = EvokeBlockMeta(last_touch_tick=self._tick)

    def remove(self, key: OffloadKey) -> None:
        self.blocks.pop(key, None)
        self.meta.pop(key, None)

    def touch(self, keys: Iterable[OffloadKey]) -> None:
        for key in keys:
            if key in self.meta:
                self._tick += 1
                self.meta[key].last_touch_tick = self._tick

    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        if n == 0:
            return []
        candidates: list[tuple[OffloadKey, BlockStatus, float]] = []
        for key, block in self.blocks.items():
            meta = self.meta[key]
            if block.ref_cnt == 0 and key not in protected and not meta.pinned:
                candidates.append((key, block, self._score(key)))
        if len(candidates) < n:
            return None
        candidates.sort(key=lambda triple: triple[2])
        chosen = candidates[:n]
        evicted: list[tuple[OffloadKey, BlockStatus]] = []
        for key, block, _ in chosen:
            del self.blocks[key]
            del self.meta[key]
            evicted.append((key, block))
        return evicted

    def clear(self) -> None:
        self.blocks.clear()
        self.meta.clear()

    def mark_evictable(self, key: OffloadKey) -> None:
        meta = self.meta.get(key)
        if meta is not None:
            meta.pinned = False

    def mark_non_evictable(self, key: OffloadKey) -> None:
        meta = self.meta.get(key)
        if meta is not None:
            meta.pinned = True

    def apply_request_tags(self, key: OffloadKey, tags: EvokeRequestTags) -> None:
        """Not part of the stock ``CachePolicy`` ABC.

        Called by ``EvokeOffloadingManager.prepare_store`` after ``insert``,
        so source-role and priority tags reach a block that the ABC itself
        never sees (the ABC only receives ``OffloadKey`` values, per design
        spec 01a section 1).
        """
        meta = self.meta.get(key)
        if meta is None:
            return
        meta.source_type = tags.source_type
        meta.priority = tags.priority
