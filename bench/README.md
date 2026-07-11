# bench

Benchmark harness for the run matrix behind the numbers in the top-level
README: three workloads (W1 AR subset, W2 LRU subset, W3 verdant-session
replay) x four arms (A0 stock, A1 LRU-offload control, A2 EVOKE-offload,
A3 MultiConnector composition) x four CPU-offload budgets, 33 core runs.
The published results were collected on an RTX 4070 Ti SUPER 16GB under
WSL2 with the `wsl2-4070ti` profile; the harness itself is
hardware-agnostic and driven entirely by `matrix.toml` and `profiles/*.env`.

## Layout

- `workloads/memory_agent_bench.py`: consumes `ai-hyz/MemoryAgentBench`
  directly from the HF Hub (pinned revision), reusing only its two scorers
  (`substring_exact_match`, `exact_match`), not its bash harness.
- `workloads/verdant_replay.py` + `workloads/cas.py`: parses a recorded
  verdant agent-session trace (`trace_path` in `matrix.toml`; the trace used
  for the published results is private and not shipped), resolves content
  hashes through
  verdant's content-addressed blob store, and falls back to deterministic
  length-preserving filler when a blob is missing.
- `metrics.py`: Prometheus `/metrics` text parsing plus before/after delta
  and histogram-quantile math (TTFT, hit rate, offload/restore bytes).
- `arms.py`: builds the `kv-transfer-config` JSON and full `vllm serve`
  command for each arm, sourcing `profiles/*.env` the same way
  `scripts/serve.sh` does.
- `matrix.py` + `matrix.toml`: resolves the run matrix, groups cells by
  server config to minimize restarts, and renders the dry-run plan.
- `runner.py`: executes one (arm, workload, budget) cell against a running
  server and writes one JSON file per cell to `results/`.
- `cli.py` (`python -m bench ...`): `matrix --dry-run`, `run-cell`,
  `prefetch`.

## Commands

Print the resolved matrix without touching a server (works with no GPU, no
running server, no network):

```bash
uv run python -m bench matrix --dry-run --profile local-2060
```

Warm the HF dataset cache for W1/W2 on a connected machine before a GPU run:

```bash
uv run python -m bench prefetch
```

Run one cell against an already-running server (the GPU box, after starting
it with the printed `vllm serve` command):

```bash
uv run python -m bench run-cell \
  --arm A2 --workload W1 --budget B2 \
  --server-url http://127.0.0.1:8000 \
  --profile local-2060 \
  --out bench/results/A2_W1_B2.json
```

`matrix` without `--dry-run` prints an error and points back at `--dry-run`
plus the printed commands: this harness does not orchestrate `vllm serve`
subprocess lifecycle itself (starting, health-polling, stopping), since that
step is inherently GPU-box-only and untestable in this repo's CPU lane. The
dry-run output is the copy-paste script for that box.

## The A3 smoke test

`matrix.toml`'s `[smoke_test]` table names a single short cell (A3 on W1 at
budget B2) that must pass before the six A3 composition rows run, per
spec 02a-workloads.md section 3. If it fails (cross-layer breakage, LMCache
layout or install friction), edit the `A3` `[[cells]]` entry in
`matrix.toml` to `arm = "A4"` (LMCache alone) for the documented
side-by-side fallback, and rerun `--dry-run` to get the updated plan.

## LMCache

A3 and A4 need the `lmcache` PyPI package installed on the GPU box; it is
**not** a dependency of this project (`pyproject.toml` does not list it),
matching how `evoke_vllm`'s `LMCacheConnectorV1` import is itself lazy
(`vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py` only
imports `lmcache` inside a method, not at module load), so nothing here
requires it until a server actually serves the A3/A4 config. Install it
separately on the GPU box (`uv add lmcache` or `pip install lmcache`) before
running those cells.

## Offline tests

`uv run pytest` runs every bench test with no GPU, no server, and no
network: workload loaders are tested against small fixtures (a fake CAS
directory, a tiny trace, canned HF-dataset-shaped rows), the metrics parser
against literal Prometheus text fixtures, and matrix resolution against
`matrix.toml` itself. Tests that need a real HF fetch are marked
`@pytest.mark.network` and skipped by default.
