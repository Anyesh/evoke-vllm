# Benchmark report

Every table below is rendered from the per-cell result JSONs next to this
file by `python -m bench report`; see `bench/README.md` for the harness,
workload definitions (`matrix.toml`, `matrix-skew.toml`), and how to rerun
any cell. Arms: A0 stock vLLM (no offload), A1 stock CPU offload with the
LRU policy, A2 CPU offload with the EVOKE policy from this package, A3
EVOKE composed with LMCache through MultiConnector.

W3 replays a recorded agent session for latency-overhead measurement; its
quality score is not meaningful and its content resolves to deterministic
filler. A `-` cell means the metric does not exist for that arm (stock
vLLM has no restore path), which is different from measuring zero.


## W1

| arm | budget | wall s | quality | hit rate | hit tokens | ttft p50 s | prefill avoided |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | NA | 186.6 | 0.47 | - | 0 | 1.750 | 61584 |
| A1 | B0 | 63.4 | 0.46 | 92.0% | 1021840 | 0.182 | 1083424 |
| A1 | B1 | 146.6 | 0.47 | 32.5% | 360688 | 1.320 | 422272 |
| A1 | B2 | 181.7 | 0.47 | 7.6% | 84320 | 1.643 | 145904 |
| A1 | B3 | 193.8 | 0.47 | 0.0% | 0 | 1.750 | 61584 |
| A2 | B0 | 61.5 | 0.46 | 92.0% | 1021840 | 0.180 | 1083424 |
| A2 | B1 | 162.9 | 0.47 | 20.6% | 228512 | 1.500 | 290096 |
| A2 | B2 | 182.0 | 0.47 | 7.6% | 84320 | 1.643 | 145904 |
| A2 | B3 | 195.2 | 0.47 | 0.0% | 0 | 1.750 | 61584 |
| A3 | B2 | 104.6 | 0.46 | 92.0% | 1021392 | 0.638 | 1082976 |
| A3 | B3 | 115.0 | 0.46 | 89.5% | 993616 | 0.658 | 1055200 |

## W1S

| arm | budget | wall s | quality | hit rate | hit tokens | ttft p50 s | prefill avoided |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | B1 | 96.6 | 0.50 | 32.4% | 226304 | 1.346 | 257360 |
| A1 | B2 | 122.5 | 0.50 | 6.0% | 42176 | 1.667 | 73232 |
| A2 | B1 | 76.6 | 0.48 | 58.0% | 405248 | 0.225 | 436304 |
| A2 | B2 | 119.4 | 0.50 | 8.1% | 56640 | 1.667 | 87696 |

## W2

| arm | budget | wall s | quality | hit rate | hit tokens | ttft p50 s | prefill avoided |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | NA | 153.1 | 0.29 | - | 0 | 1.750 | 16464 |
| A1 | B0 | 79.6 | 0.32 | 85.5% | 607200 | 0.187 | 623664 |
| A1 | B1 | 148.2 | 0.29 | 10.1% | 71552 | 1.649 | 88016 |
| A1 | B2 | 154.5 | 0.29 | 3.1% | 21984 | 1.710 | 38448 |
| A1 | B3 | 157.9 | 0.29 | 0.0% | 0 | 1.750 | 16464 |
| A2 | B0 | 78.8 | 0.32 | 85.5% | 607200 | 0.187 | 623664 |
| A2 | B1 | 148.3 | 0.29 | 10.0% | 70752 | 1.649 | 87216 |
| A2 | B2 | 155.6 | 0.29 | 3.1% | 21984 | 1.710 | 38448 |
| A2 | B3 | 158.1 | 0.29 | 0.0% | 0 | 1.750 | 16464 |
| A3 | B2 | 187.5 | 0.29 | 16.6% | 117984 | 1.686 | 134448 |
| A3 | B3 | 174.2 | 0.29 | 16.5% | 117408 | 1.578 | 133872 |

## W3

| arm | budget | wall s | quality | hit rate | hit tokens | ttft p50 s | prefill avoided |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | NA | 178.9 | - | - | 0 | 0.098 | 19312 |
| A1 | B0 | 183.8 | - | 0.1% | 352 | 0.123 | 19664 |
| A1 | B1 | 184.2 | - | 0.1% | 352 | 0.123 | 19664 |
| A1 | B2 | 189.3 | - | 0.1% | 176 | 0.124 | 19488 |
| A1 | B3 | 183.9 | - | 0.0% | 0 | 0.123 | 19312 |
| A2 | B0 | 189.2 | - | 0.1% | 352 | 0.123 | 19664 |
| A2 | B1 | 183.8 | - | 0.1% | 176 | 0.123 | 19488 |
| A2 | B2 | 188.8 | - | 0.0% | 0 | 0.127 | 19312 |
| A2 | B3 | 190.2 | - | 0.0% | 0 | 0.127 | 19312 |
| A3 | B2 | 203.6 | - | 0.0% | 0 | 0.173 | 19312 |
| A3 | B3 | 203.2 | - | 0.0% | 0 | 0.175 | 19312 |
