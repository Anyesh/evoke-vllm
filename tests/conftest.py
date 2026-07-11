import os

import pytest


@pytest.fixture(autouse=True)
def _clean_evoke_env(monkeypatch):
    # Config parsing reads EVOKE_* env overrides; a stray one in the ambient
    # shell would make from_extra_config tests non-deterministic.
    for name in list(os.environ):
        if name.startswith("EVOKE_"):
            monkeypatch.delenv(name, raising=False)


def pytest_addoption(parser):
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="run tests marked @pytest.mark.network (real HF Hub fetch)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-network"):
        return
    skip_network = pytest.mark.skip(reason="needs --run-network (real HF Hub fetch)")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)
