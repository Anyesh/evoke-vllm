"""EVOKE offloading manager: threads ``req_context`` into the policy.

``CachePolicy`` only ever sees ``OffloadKey`` values (a content hash plus
group index); it has no request identity. ``CPUOffloadingManager``, in
contrast, receives ``ReqContext`` (``req_id`` and ``kv_transfer_params``) on
every primitive. That asymmetry means the package must own the manager,
not just the policy: the manager subclass below is where
``kv_transfer_params["evoke"]`` tags get attached to keys as they are stored.

The stock manager selects its policy through a ``Literal["lru", "arc"]``
constructor parameter backed by the private ``_CACHE_POLICIES`` dict, which
has no registration hook for a third name. ``EvokeOffloadingManager`` does
not use that parameter; it always constructs an ``EvokeCachePolicy``.

The subclass stays thin to limit drift against upstream releases: it
overrides only public primitives (``__init__``, ``lookup``,
``prepare_store``) and inherits all ref-counting, block-pool, and event logic
unchanged.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Collection

from vllm.v1.kv_offload.base import OffloadKey, PrepareStoreOutput, ReqContext
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

from evoke_vllm.config import EvokeRequestTags, EvokeScoringConfig
from evoke_vllm.policy import EvokeCachePolicy


class EvokeOffloadingManager(CPUOffloadingManager):
    def __init__(
        self,
        num_blocks: int,
        enable_events: bool = False,
        store_threshold: int = 1,
        max_tracker_size: int = 64_000,
        scoring_config: EvokeScoringConfig | None = None,
    ) -> None:
        # cache_policy="lru" only satisfies the base's _CACHE_POLICIES lookup;
        # the LRU instance it builds is discarded and replaced below, because
        # the dict has no registration hook for a third policy name.
        super().__init__(
            num_blocks=num_blocks,
            cache_policy="lru",
            enable_events=enable_events,
            store_threshold=store_threshold,
            max_tracker_size=max_tracker_size,
        )
        self.scoring_config = scoring_config or EvokeScoringConfig()
        # Reuse tracker maintained unconditionally, unlike the stock
        # self.counts which exists only when store_threshold >= 2. Shared by
        # reference with the policy so its score can read the reuse signal.
        self.reuse_counts: OrderedDict[OffloadKey, int] = OrderedDict()
        self._policy: EvokeCachePolicy = EvokeCachePolicy(
            cache_capacity=num_blocks,
            scoring_config=self.scoring_config,
            reuse_counts=self.reuse_counts,
        )

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        if key in self.reuse_counts:
            self.reuse_counts.move_to_end(key)
            self.reuse_counts[key] += 1
        else:
            if len(self.reuse_counts) >= self.max_tracker_size:
                self.reuse_counts.popitem(last=False)
            self.reuse_counts[key] = 1
        return super().lookup(key, req_context)

    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        output = super().prepare_store(keys, req_context)
        if output is None:
            return None
        tags = EvokeRequestTags.from_kv_transfer_params(req_context.kv_transfer_params)
        for key in output.keys_to_store:
            self._policy.apply_request_tags(key, tags)
        return output
