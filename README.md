# evoke-vllm

evoke-vllm is a relevance-driven eviction policy for stock vLLM's CPU KV-cache
offload tier, ported from [EVOKE](https://doi.org/10.5281/zenodo.21285585)
(reversible KV eviction and recovery, published and demonstrated on llama.cpp).
Recency, reuse frequency, and client-supplied request structure decide which
offloaded blocks get dropped under memory pressure, instead of plain LRU. It
plugs in through vLLM's documented `OffloadingSpec` / `spec_module_path`
extension point, so it installs alongside a stock `pip install vllm==0.24.0`
with no fork and no patched vLLM required.

On an agent-style workload, where a hot set of contexts gets revisited every
loop while cold scans churn past it, scored eviction lifts restore hit rate
from 32% to 58% over stock LRU and cuts mean hot-request TTFT from 1.03s to
0.42s at matching task quality, 0.48 vs 0.50 (Qwen2.5-7B FP8, RTX 4070 Ti
SUPER 16GB). Every number here reads off the W1S table in
`bench/REPORT.md`, which is rendered from the result JSONs in this repo.

## Install

```bash
uv add evoke-vllm    # inside a uv project; run `uv init` first if you have none
# or
pip install evoke-vllm
```

This pulls in `vllm==0.24.0` as a pinned dependency, and vLLM brings torch
with it: budget roughly 9 GB of disk for a fresh environment.

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

`tests/test_factory_route.py` exercises this route end to end: it drives
stock vLLM's own `OffloadingSpecFactory.create_spec` over exactly this
config shape, confirms the factory resolves `EvokeOffloadingSpec` with
`EvokeOffloadingManager` and `EvokeCachePolicy` underneath, and runs a
store -> lookup round trip with an `evoke` tag through the factory-created
manager. The GPU-side `get_handlers` path needs a real model and device;
the GPU gates in the results section cover that.

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

## Running the tests

```bash
git clone https://github.com/Anyesh/evoke-vllm
cd evoke-vllm
uv sync
uv run pytest
```

`uv sync` installs the dev group (pytest, ruff, blake3, datasets) that the
offline suite needs; installing pytest alone is not enough, since the bench
workloads import `datasets` at module level. The suite runs CPU-only; one
test marked `network` is skipped unless you enable real HF Hub access. The
GPU gates and the benchmark matrix have their own instructions in
`scripts/README_GATES.md` and `bench/README.md`.

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
LMCache across four CPU budgets. Three regimes fall out. With re-access
skewed toward a hot set and the CPU budget above that hot set (the W1S
rows in `bench/REPORT.md`, 3 GiB budget), scored eviction beats stock LRU
58% to 32% on restore hit rate and 0.42s to 1.03s on mean hot-request
TTFT, at matching task quality (0.48 vs 0.50). With uniform re-access,
recency is already the right ranking: it ties LRU at three of four
budgets and loses one cell (W1 at the 3 GiB budget: 21% vs 33% restore
hits at identical quality, single run). With the budget below the hot set, both policies
collapse together and the useful knob is `store_threshold`, not scoring.
Scored eviction earns its keep when the workload has a hot set; under
uniform re-access, stock LRU is already optimal.

## Scope of this release

P1 scores on recency, reuse frequency, and client-supplied tags.
Attention-mass and embedding-coherence scoring, smart-recovery bring-back
(top-K restore keyed on a query embedding), and the RoPE re-anchoring that
goes with landing a block at a new position all need GPU-side signals and
restore triggers that stock vLLM does not expose at this extension point
yet. They are the subject of the companion upstream RFC, and their config
weights ship as inert zeros rather than proxy simulations, so the config
surface is already shaped for them.

## License

Apache-2.0.
