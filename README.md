# evoke-vllm

evoke-vllm is a relevance-driven eviction policy for stock vLLM's CPU KV-cache
offload tier, ported from [EVOKE](https://doi.org/10.5281/zenodo.21285585)
(reversible KV eviction and recovery, published and demonstrated on llama.cpp).
Recency, reuse frequency, and client-supplied request structure decide which
offloaded blocks get dropped under memory pressure, instead of plain LRU. It
plugs in through vLLM's documented `OffloadingSpec` / `spec_module_path`
extension point, so it installs alongside a stock `pip install vllm==0.24.0`
with no fork and no patched vLLM required.

This P1 release evicts on recency, reuse, and client-supplied structure
only. It does not use attention mass or embedding-based coherence scoring;
stock vLLM exposes neither signal at the policy or manager scope this
package can reach, so those weights are wired into the config as inert
zeros rather than simulated from proxies. It also does not implement
smart-recovery bring-back (top-K restore keyed on a query embedding) or the
RoPE re-anchoring that would go with landing a block at a different
position than it was written at: stock vLLM's restore path always lands a
block back at the position it was hashed from, so there is no trigger for
either feature yet. Both are reserved for the RFC track once the relevant
GPU-side hooks exist upstream. The scoring and eviction logic (recency,
reuse, source floors, priority, atomic evict) is covered by the offline
test suite and validated on GPU; see the results section below.

## Install

```bash
uv add evoke-vllm
# or
pip install evoke-vllm
```

This pulls in `vllm==0.24.0` as a pinned dependency.

## Config sketch

Point vLLM's `OffloadingConnector` at this package through
`kv_connector_extra_config`:

```python
kv_transfer_config = KVTransferConfig(
    kv_connector="OffloadingConnector",
    kv_role="kv_both",
    kv_connector_extra_config={
        "spec_name": "EvokeOffloadingSpec",
        "spec_module_path": "evoke_vllm.spec",
        "cpu_bytes_to_use": 32 * 1024**3,
        "block_size": 256,
        "store_threshold": 2,
        "offload_prompt_only": True,
        "evoke": {
            "w_recency": 0.5,
            "w_reuse": 0.5,
            "recency_half_life": 64,
            "source_floors": {"system": 0.6, "user": 0.6, "assistant": 0.5},
        },
    },
)
```

`spec_name` and `spec_module_path` select this package's spec through
vLLM's dynamic-import route; `cpu_bytes_to_use`, `block_size`, and
`store_threshold` are stock offload knobs. The `evoke` sub-key carries
this package's own tuning and env-var overrides for it.

This route is proven end to end, not just asserted: `tests/test_factory_route.py`
builds real `VllmConfig` / `KVCacheConfig` objects carrying exactly this shape of
`kv_connector_extra_config`, calls stock vLLM's own
`OffloadingSpecFactory.create_spec` (the same call `OffloadingConnector.__init__`
makes) without importing `evoke_vllm.spec` directly, and confirms the factory
resolves `EvokeOffloadingSpec`, its manager is `EvokeOffloadingManager`, the
manager's policy is `EvokeCachePolicy`, and a non-default `evoke` weight reaches
the policy's scoring config. It also drives one `prepare_store` -> `complete_store`
-> `lookup` round trip through the factory-created manager and confirms an
`evoke` tag from `kv_transfer_params` lands on the stored block. That test stops
at the spec/manager layer (scheduler-side); it does not boot an engine or touch
the GPU-side `get_handlers` path, which needs a real model and CUDA/XPU device.

Per-request tags travel inside `kv_transfer_params`, alongside stock keys
such as `max_offload_tokens`:

```python
sampling_params.extra_args = {
    "kv_transfer_params": {
        "evoke": {
            "source_type": "user",
            "priority": 1.5,
            "evoke_session": "conversation-42",
        }
    }
}
```

Tags are read exactly as stock vLLM reads `max_offload_tokens` today: they
are optional, and untagged traffic falls back to recency plus reuse. They
drive scoring and metrics grouping only; eviction and restore stay
content-addressed regardless of tagging.

`offload_prompt_only` defaults to `true`, so only prompt and prefill blocks
are offloaded and eligible for restore; decode-generated blocks are skipped
unless an operator sets it to `false`.

## Results

Two kinds of GPU validation back this package, both runnable from this repo.

The fidelity gate (`scripts/README_GATES.md`) checks that restored blocks
decode like never-evicted ones. On Qwen2.5-1.5B (RTX 2060 6GB) it passes
70 of 70 probes with real offload traffic. On Qwen2.5-7B FP8 (RTX 4070 Ti
SUPER 16GB), continuations through restored blocks diverged from a pinned
greedy baseline on 9 of 80 probes while stock recompute diverged on 16 of
80 of its own, with 10 of 10 passkey retrievals through restored content:
restoring saved bytes is more deterministic than recomputing them.

The benchmark matrix (`bench/README.md`, same 7B setup) compares stock
vLLM, the stock LRU offload policy, this policy, and composition with
LMCache across four CPU budgets. The regime map from those runs: with
re-access skewed toward a hot set and the CPU budget above that hot set,
scored eviction beats stock LRU 58% to 32% on restore hit rate and 0.42s
to 1.03s on mean hot-request TTFT, at equal task quality. With uniform
re-access, recency is already the right ranking: it ties LRU at three of
four budgets and loses one cell (21% vs 33% restore hits at the 3 GiB
budget, single run).
With the budget below the hot set both policies collapse together and the
useful knob is `store_threshold`, not scoring. If your workload re-accesses
its context uniformly, this package will not beat the stock LRU policy,
and you should know that before installing it.

## License

Apache-2.0.
