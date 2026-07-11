"""Fake-block harness for the vLLM-gated test lanes.

Modeled on stock vLLM's own offload tests
(``tests/v1/kv_offload/cpu/test_manager.py`` and
``tests/v1/kv_connector/unit/offloading_connector/utils.py``): drive the
manager and policy directly with content-hash keys and ``BlockStatus`` slots,
never a real GPU. Every symbol used here (``make_offload_key``, ``ReqContext``,
``BlockStatus``) is part of stock ``vllm==0.24.0``; this module is imported
only after ``pytest.importorskip("vllm")`` in the gated test files, so it never
loads on the pure-unit lane.
"""

from __future__ import annotations

from collections.abc import Iterable

from vllm.v1.kv_offload.base import OffloadKey, ReqContext, make_offload_key
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus


def to_key(int_hash: int, group_idx: int = 0) -> OffloadKey:
    return make_offload_key(str(int_hash).encode(), group_idx)


def to_keys(int_hashes: Iterable[int]) -> list[OffloadKey]:
    return [to_key(i) for i in int_hashes]


def make_req_context(
    req_id: str = "", kv_transfer_params: dict | None = None
) -> ReqContext:
    return ReqContext(req_id=req_id, kv_transfer_params=kv_transfer_params)


def ready_block(block_id: int, ref_cnt: int = 0) -> BlockStatus:
    block = BlockStatus(block_id)
    block.ref_cnt = ref_cnt
    return block
