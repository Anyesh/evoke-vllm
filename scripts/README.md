# scripts

GPU correctness gates from design spec 01a section 5: a human-run
serve-and-measure pass that checks offload-then-restore fidelity, adapted
from EVOKE's `verify_kv_restore.py` and
`verify_kv_fidelity*.py` to vLLM's OpenAI-compatible HTTP server instead of
an in-process llama.cpp engine. See `README_GATES.md` for exact usage.

- `serve.sh` launches `vllm serve` for a hardware profile
  (`profiles/*.env`), with the EVOKE offload connector wired in by default
  or stripped out under `--baseline`.
- `fidelity_gate.py` drives a multi-session growing-prefix workload against
  a running server, and diffs a baseline recording against an
  evoke-connector recording for token/logprob fidelity plus a non-vacuous
  restore check.
- `gate_lib.py` is the pure, stdlib-only core (workload construction,
  request building, Prometheus metrics parsing) both of the above and
  `tests/test_gate_lib.py` import.

Both scripts have a `--dry-run` mode that needs no server and no GPU; see
`tests/test_gate_lib.py` and `tests/test_serve_script.py` for the offline
unit-test lane covering them.
