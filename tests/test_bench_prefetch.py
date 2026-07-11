from pathlib import Path

import pytest

from bench.matrix import load_matrix
from bench.workloads.factory import mab_config_from_spec
from bench.workloads.memory_agent_bench import fetch_rows

REAL_MATRIX_PATH = Path(__file__).resolve().parents[1] / "bench" / "matrix.toml"


@pytest.mark.network
def test_fetch_rows_hits_real_hf_dataset_for_w1_and_w2():
    config = load_matrix(REAL_MATRIX_PATH)
    for workload_id in ("W1", "W2"):
        mab_config = mab_config_from_spec(config.workloads[workload_id])
        rows = fetch_rows(mab_config)
        assert len(rows) > 0
        for row in rows:
            assert row["metadata"]["source"] in mab_config.sources
            assert "context" in row
            assert "questions" in row
