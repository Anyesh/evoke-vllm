"""Story 1.4: prove the sanctioned dynamic-loading route end to end.

Exercises ``OffloadingSpecFactory.create_spec`` (``vllm/v1/kv_offload/factory.py``),
the exact call ``OffloadingConnector.__init__`` makes, with real ``VllmConfig`` /
``KVCacheConfig`` objects built the way stock vLLM's own offloading-connector
tests build them (``tests/v1/kv_connector/unit/offloading_connector/utils.py``
in the vLLM source tree). ``EvokeOffloadingSpec`` and ``EvokeOffloadingManager``
are never imported and constructed directly for the assertions below; the
factory resolves both through ``spec_name`` / ``spec_module_path`` in
``kv_connector_extra_config``, the same route vllm-ascend uses for
``NPUOffloadingSpec``.

Boundary: ``VllmConfig.model_config`` is left at its documented ``None``
default. The field's docstring in ``vllm.config.vllm.VllmConfig`` carries a
maintainer TODO ("use default_factory once default constructing ModelConfig
doesn't try to download a model") and every ``model_config`` read in
``VllmConfig.__post_init__`` is guarded by ``is not None``, so omitting it is
not a stand-in for the real thing, it is vLLM's own sanctioned way to skip a
model checkpoint. ``OffloadingSpec.__init__`` and ``CPUOffloadingSpec.__init__``
never read ``model_config`` either; they read ``kv_transfer_config``,
``kv_events_config``, ``parallel_config``, and ``cache_config`` block sizing
(via ``resolve_kv_cache_block_sizes``), all of which are real here. This is
the deepest layer reachable without booting a scheduler, request runner, or
GPU worker, none of which this package's injection point (spec + manager)
touches; that boundary is also why ``get_handlers`` (the GPU/XPU-gated half
of ``OffloadingSpec``) is out of scope for this test.
"""

import pytest

pytest.importorskip("vllm")

import torch  # noqa: E402
from offload_harness import make_req_context, to_key  # noqa: E402
from vllm.config import DeviceConfig, KVTransferConfig, VllmConfig  # noqa: E402
from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm.v1.kv_offload.factory import OffloadingSpecFactory  # noqa: E402

from evoke_vllm.manager import EvokeOffloadingManager  # noqa: E402
from evoke_vllm.policy import EvokeCachePolicy  # noqa: E402
from evoke_vllm.spec import EvokeOffloadingSpec  # noqa: E402

GPU_BLOCK_SIZE = 16
CPU_BLOCKS = 4
CPU_BYTES_TO_USE = CPU_BLOCKS * 1024


def _extra_config() -> dict:
    return {
        "spec_name": "EvokeOffloadingSpec",
        "spec_module_path": "evoke_vllm.spec",
        "cpu_bytes_to_use": CPU_BYTES_TO_USE,
        "block_size": GPU_BLOCK_SIZE,
        "store_threshold": 2,
        "offload_prompt_only": True,
        "evoke": {"w_recency": 0.9},
    }


def _build_vllm_config() -> VllmConfig:
    kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config=_extra_config(),
    )
    return VllmConfig(
        kv_transfer_config=kv_transfer_config, device_config=DeviceConfig("cpu")
    )


def _build_kv_cache_config() -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=CPU_BLOCKS,
        kv_cache_tensors=[KVCacheTensor(size=CPU_BYTES_TO_USE, shared_by=["layer"])],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=GPU_BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )


def test_factory_resolves_evoke_spec_manager_and_policy():
    spec = OffloadingSpecFactory.create_spec(
        _build_vllm_config(), _build_kv_cache_config()
    )

    assert type(spec) is EvokeOffloadingSpec
    manager = spec.get_manager()
    assert type(manager) is EvokeOffloadingManager
    assert type(manager._policy) is EvokeCachePolicy
    # non-default extra_config["evoke"]["w_recency"] must reach the policy's
    # scoring config through the manager, not just stop at the spec.
    assert manager.scoring_config.w_recency == pytest.approx(0.9)
    assert manager._policy.config.w_recency == pytest.approx(0.9)


def test_factory_created_manager_store_lookup_round_trip_threads_tags():
    manager = OffloadingSpecFactory.create_spec(
        _build_vllm_config(), _build_kv_cache_config()
    ).get_manager()

    key = to_key(1)
    ctx = make_req_context(
        kv_transfer_params={"evoke": {"source_type": "system", "priority": 2.0}}
    )
    # store_threshold=2 gates the stock reuse tracker; two lookups make the
    # key store-eligible, mirroring a real repeated prefix-cache hit.
    assert manager.lookup(key, ctx) is False
    assert manager.lookup(key, ctx) is False

    out = manager.prepare_store([key], ctx)
    assert out is not None
    assert out.keys_to_store == [key]
    manager.complete_store([key], ctx, success=True)

    meta = manager._policy.meta[key]
    assert meta.source_type == "system"
    assert meta.priority == pytest.approx(2.0)
    assert manager.lookup(key, ctx) is True
