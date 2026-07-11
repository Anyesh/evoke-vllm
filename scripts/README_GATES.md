# GPU correctness gates

Human-run, GPU-required verification that `evoke_vllm` restores CPU-offloaded
KV blocks correctly: token streams and top-logprobs from a run that evicts
and restores must match a run that never evicted, and the restore has to
have actually happened (not a vacuous pass).

The `local-2060` profile passed this gate on the real RTX 2060 on
2026-07-11: 70/70 requests matched the baseline exactly, passkey recall
10/10 on both arms, 1,232 external prefix cache hits, 40.4 MB offloaded and
35.3 MB restored. The workload flags and profile values below are the
validated ones from that run. The `wsl2-4070ti` profile remains dry-run
validated only; treat its values as starting points.

## What the gate checks

`scripts/fidelity_gate.py` drives a temperature-0, multi-session workload
(`scripts/gate_lib.py`) against a running vLLM OpenAI-compatible server:
each of several sessions grows its own prefix turn by turn (append-only, so
vLLM's content-addressed block hashing treats each turn as a cache hit
against the previous one), and the sessions are interleaved round-robin so
that, under a memory-constrained GPU profile, one session's blocks get
evicted from the GPU prefix cache while other sessions run. A final replay
pass reuses each session's full grown prefix. Two full recordings are
compared:

1. **Fidelity**: every request's generated tokens and top-logprobs from the
   evoke-connector run must match the baseline (no-connector) run within
   tolerance (exact token match, `--logprob-atol` on logprob values, default
   `0.05`).
2. **Non-vacuous**: the evoke run must show `external_prefix_cache_hits` and
   `cpu_to_gpu` offload bytes actually increasing over the run. If nothing
   was ever evicted from the GPU (profile too generous) or nothing came back
   from the CPU tier, the gate fails with an explicit "gate is vacuous"
   message rather than passing by doing nothing.

Both checks land in one JSON result file from `fidelity_gate.py compare`,
and the process exit code reflects the verdict (`0` pass, `1` fail), so a
CI-style caller can gate on exit status alone.

A single GPU generally cannot hold both a baseline server and an
evoke-connector server at once, so the workflow is two sequential
recordings against the same port, not two servers running side by side.

## Prerequisites (both profiles)

```bash
cd evoke-vllm
uv sync
```

`uv sync` installs `evoke_vllm` itself into the project's `.venv` (editable),
which is what lets `spec_module_path: "evoke_vllm.spec"` resolve inside the
`vllm serve` process later. Confirm before doing anything GPU-side:

```bash
uv run python -c "import evoke_vllm.spec"   # only works once vllm + evoke_vllm are both importable
uv run pytest -q                             # both existing test lanes should be green
```

## Dry-run first, on any machine, no GPU required

```bash
scripts/serve.sh --profile local-2060 --dry-run
scripts/serve.sh --profile local-2060 --baseline --dry-run
scripts/serve.sh --profile wsl2-4070ti --dry-run
python scripts/fidelity_gate.py record --dry-run --model qwen2.5-1.5b-instruct \
    --run-label evoke --out /tmp/plan.json --sessions 3 --growth-turns 3
```

`serve.sh --dry-run` prints the resolved `kv-transfer-config` JSON and the
exact `uv run vllm serve ...` command without executing it.
`fidelity_gate.py record --dry-run` prints every HTTP request (method, URL,
JSON body) the workload would send, without a server. Both are what
`tests/test_serve_script.py` and `tests/test_gate_lib.py` assert against
offline.

## Profile: local-2060 (RTX 2060 6GB, Turing)

Qwen2.5-1.5B-Instruct, fp16, `gpu-memory-utilization=0.75`, validated on
the real card 2026-07-11 (GATE PASS, non-vacuous). See
`profiles/local-2060.env` for the full knob list and the observed numbers
behind each value.

The default 12-request workload (3 sessions x 3 growth turns) is **not
enough to make this gate meaningful** on this profile: its roughly 4k-token
session footprint never fills the 7,840-token GPU pool, nothing evicts, and
the gate fails as vacuous. The flags below are the validated workload
(10 sessions x 6 growth turns + 10 replays = 70 requests, about 18k tokens
of session footprint, 2.3x pool overcommit). Environment variables override
profile values, so a busy port or a different memory target does not
require editing the profile (the validated run used `EVOKE_PORT=8151` to
avoid a busy port 8000).

```bash
mkdir -p scripts/results

# 1. Baseline: stock vLLM, no connector, no eviction possible.
EVOKE_PORT=8151 scripts/serve.sh --profile local-2060 --baseline
# in a second terminal, once the server is ready:
uv run python scripts/fidelity_gate.py record \
    --base-url http://localhost:8151 --model qwen2.5-1.5b-instruct \
    --sessions 10 --growth-turns 6 --filler-words 220 \
    --run-label baseline --out scripts/results/local-baseline.json
# Ctrl-C the server.

# 2. Evoke: EVOKE connector, constrained GPU pool. Identical workload flags.
EVOKE_PORT=8151 scripts/serve.sh --profile local-2060
uv run python scripts/fidelity_gate.py record \
    --base-url http://localhost:8151 --model qwen2.5-1.5b-instruct \
    --sessions 10 --growth-turns 6 --filler-words 220 \
    --run-label evoke --out scripts/results/local-evoke.json
# Ctrl-C the server.

# 3. Compare, no server needed.
uv run python scripts/fidelity_gate.py compare \
    --baseline scripts/results/local-baseline.json \
    --evoke scripts/results/local-evoke.json \
    --out scripts/results/local-fidelity-result.json
echo "exit code: $?"
```

Observed runtime on the 2060: about 70-90s from `serve.sh` to a ready
`/health` (engine init is ~37s of that, model cached), 2-3 minutes per
70-request recording, roughly 15 minutes for the whole three-step sequence
including both server starts.

If step 3 reports `vacuous`: nothing was evicted from the GPU tier during
the evoke recording. Grow the workload first
(`--sessions`/`--growth-turns`/`--filler-words` on **both** `record`
commands, identically) so more KV competes for the same GPU pool; lower
`EVOKE_GPU_MEMORY_UTILIZATION` only if the workload cannot reasonably grow
further. The sizing rule that worked: read the `GPU KV cache size: N
tokens` line the server prints at startup, estimate the workload footprint
as `sessions x growth_turns x (filler_words x 1.35)` tokens, and make the
footprint at least 2x N. The `record` run also warns live, request by
request, while its offload counters remain zero.

## Profile: wsl2-4070ti (RTX 4070 Ti SUPER 16GB, WSL2 on Windows)

Qwen2.5-7B-Instruct, FP8 weights
(`RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic`, apache-2.0, verified on
huggingface.co) with fp16 KV cache, per spec `02a-workloads.md` section 5.
See `profiles/wsl2-4070ti.env` for the full knob list.

### One-time WSL2 + CUDA setup (on the Windows box)

1. Confirm the Windows NVIDIA driver already supports WSL2 GPU passthrough:
   open PowerShell and run `nvidia-smi`. If that fails, update the Windows
   NVIDIA driver first (GPU driver installs on the Windows side only; do
   **not** install a Linux NVIDIA driver inside WSL2, it will conflict).
2. `wsl --install` (or `wsl --update` if WSL is already present) from an
   elevated PowerShell, then reboot if prompted. This needs a WSL2 kernel
   recent enough to carry GPU passthrough.
3. Inside the WSL2 Ubuntu shell, install the CUDA toolkit via NVIDIA's
   WSL-Ubuntu network repo (not the generic Ubuntu repo, and specifically
   **not** the `nvidia-driver` package, which is Windows' job):
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt-get update
   sudo apt-get -y install cuda-toolkit
   ```
4. Verify from inside WSL2: `nvidia-smi` should list the 4070 Ti SUPER.
5. Raise the WSL2 VM memory cap if needed. WSL2 defaults to 50% of Windows
   RAM; `EVOKE_CPU_BYTES_TO_USE` in `profiles/wsl2-4070ti.env` (8GiB
   default) has to fit inside that cap, not just "spare RAM" on the host.
   Edit `%UserProfile%\.wslconfig` on Windows:
   ```ini
   [wsl2]
   memory=24GB
   ```
   then `wsl --shutdown` from PowerShell and reopen the WSL2 shell.

### Project setup (inside WSL2 Ubuntu)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# clone or copy this repo into the WSL2 filesystem, then:
cd /path/to/evoke-vllm
uv sync
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The FP8 checkpoint (about 7.6GB) downloads automatically on first
`vllm serve` via the Hugging Face Hub; if it 401s, `huggingface-cli login`
first (no gating was observed on `RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic`
as of verification, but org-level HF settings can change).

### Run the gate (validated on the real card 2026-07-11)

This profile needs an **asymmetric** two-arm setup, unlike local-2060.
Three findings from the real run drive it:

1. **The baseline reference must never evict.** When the baseline pool is
   smaller than the workload footprint, the baseline recomputes evicted
   prefixes in different chunk shapes than it originally computed them,
   which shifts logits enough to flip near-tie tokens on the FP8 path. A
   first attempt with both arms at the same utilization failed fidelity
   12/108, every failure at the same turn, purely from this reference
   contamination. Baseline runs at 0.85 (77,168-token pool).
2. **The evoke arm needs `EVOKE_STORE_THRESHOLD=1`.** With the product
   default of 2, a block is stored only on its second computation, so
   under mild pressure stores lag eviction: one run stored 69.7MB and
   restored zero bytes (vacuous). Threshold 1 stores on first offer, which
   is what a restore-fidelity gate wants.
3. **Size the workload with the real tokenizer, not a words-to-tokens
   guess.** A 1.35 tokens/word estimate overshot; the actual factor for
   the filler vocabulary is about 1.28, and a 19.7k-token workload against
   a 17.3k pool (1.14x) produced zero eviction. The flags below measure at
   39,736 tokens (2.3x the evoke arm's 17,280-token pool), the same
   overcommit that worked on the 2060.

```bash
mkdir -p scripts/results

# 1. Baseline: stock vLLM, big pool, never evicts.
EVOKE_GPU_MEMORY_UTILIZATION=0.85 scripts/serve.sh --profile wsl2-4070ti --baseline
uv run python scripts/fidelity_gate.py record \
    --base-url http://localhost:8000 --model qwen2.5-7b-instruct-fp8 \
    --sessions 10 --growth-turns 7 --filler-words 450 \
    --run-label baseline --out scripts/results/wsl2-baseline.json
# Ctrl-C, wait for nvidia-smi memory.used to drop below ~1.2GB.

# 2. Evoke: constrained pool, store-on-first-offer.
EVOKE_GPU_MEMORY_UTILIZATION=0.65 EVOKE_STORE_THRESHOLD=1 \
    scripts/serve.sh --profile wsl2-4070ti
uv run python scripts/fidelity_gate.py record \
    --base-url http://localhost:8000 --model qwen2.5-7b-instruct-fp8 \
    --sessions 10 --growth-turns 7 --filler-words 450 \
    --run-label evoke --out scripts/results/wsl2-evoke.json
# Ctrl-C

uv run python scripts/fidelity_gate.py compare \
    --baseline scripts/results/wsl2-baseline.json \
    --evoke scripts/results/wsl2-evoke.json \
    --out scripts/results/wsl2-fidelity-result.json
```

Observed runtime: about 60-90s per server start (checkpoint cached), 5-8
minutes per 80-request recording, roughly 30 minutes for the sequence.

### Interpreting an FP8 fidelity failure: run the stock-recompute control

On this FP8/7B stack the exact-token-match bar is not achievable for any
run whose cache state diverges from the reference, connector or no
connector: recomputing an evicted prefix in different chunk shapes shifts
logits by ~0.1 and flips near-tie tokens deep into long contexts. The
observed gate result was 9 of 80 diverged (all in the replay phase,
passkey recall 10/10 on both arms, 7.8GB restored, 136k external hit
tokens). Whether that is restore corruption or upstream numerics is
answerable with a control arm: record **stock vLLM with the same small
pool** (evicts and recomputes, no connector installed) and compare it
against the same never-evicting baseline:

```bash
EVOKE_GPU_MEMORY_UTILIZATION=0.65 scripts/serve.sh --profile wsl2-4070ti --baseline
uv run python scripts/fidelity_gate.py record \
    --base-url http://localhost:8000 --model qwen2.5-7b-instruct-fp8 \
    --sessions 10 --growth-turns 7 --filler-words 450 \
    --run-label evoke --out scripts/results/wsl2-control.json
uv run python scripts/fidelity_gate.py compare \
    --baseline scripts/results/wsl2-baseline.json \
    --evoke scripts/results/wsl2-control.json \
    --out scripts/results/wsl2-control-result.json
# The control's "vacuous" verdict is expected (no connector, no restores);
# only its fidelity failure count matters.
```

Verdict rule: the connector passes if its diverged-request count is at or
below the stock control's and passkey recall is intact on both arms. On
the real card the control diverged **more** than the connector arm (16/80
across growth and replay, vs 9/80 replay-only): bit-exact restores from
CPU are numerically closer to the never-evicted reference than stock's
own recompute path. That is the expected signature of a faithful restore.
Garbled output, failed passkeys, or divergence counts well above the
control would instead indicate real restore corruption.

Note on the 2060: its gate passed 70/70 exact even though its baseline
also evicted (fp16/1.5B recompute happened to be bit-stable there). The
never-evicting-baseline rule above is the principled recipe regardless of
whether a given stack gets lucky.

## What this gate deliberately does not check

`EVOKE_CPU_BYTES_TO_USE` in both profiles is set generously relative to the
workload on purpose, so nothing gets evicted from the CPU tier itself
during a run: this isolates "GPU evicts, CPU restores" (what the fidelity
gate proves) from EVOKE's own eviction *ordering* under CPU pressure
(recency/reuse/source-floor scoring), which is already covered by the
offline unit suite in `tests/test_policy.py` and does not need a GPU.

## Two metric-naming discrepancies worth knowing about

**The `_total` suffix (bit us on the real 2060 run).** vLLM's source
declares counters with bare names (`vllm:kv_offload_total_bytes`,
`vllm:external_prefix_cache_hits`), but prometheus_client's text exposition
appends `_total` to every counter, so the live `/metrics` endpoint serves
`vllm:kv_offload_total_bytes_total` and
`vllm:external_prefix_cache_hits_total`. Verifying names against vLLM's
source alone is what let the first real gate run read zeros from a server
that had in fact offloaded 40 MB: the run looked vacuous while the
connector was working. Both parsers (`scripts/gate_lib.py`,
`bench/metrics.py`) now resolve each counter under both spellings, exact
match only, so the `_created` timestamp gauges that accompany every counter
never leak into the sums. The fixtures
`tests/fixtures/metrics_live_vllm_0_24_0.prom` and
`tests/bench_fixtures/metrics_live_before.prom` / `metrics_live_after.prom`
are verbatim scrapes from the real 2060 server and pin the served names.

**Label casing.** Spec `02a-workloads.md` refers to
`kv_offload_total_bytes{transfer_type="cpu_to_gpu"}`. The vLLM 0.24.0
actually installed in this project's `.venv`
(`vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py`)
emits that label as `"CPU_to_GPU"` / `"GPU_to_CPU"` (mixed case), not
lowercase. `scripts/gate_lib.summarize_offload_metrics` matches
`transfer_type` case-insensitively for exactly this reason; see the fixture
`tests/fixtures/metrics_healthy.prom`, which uses the real mixed-case
labels, and `tests/fixtures/metrics_flat_fallback.prom`, which exercises
the fallback to the newer flat `vllm:kv_offload_load_bytes` /
`vllm:kv_offload_store_bytes` counters for whenever the deprecated,
labeled metric is eventually removed upstream.
