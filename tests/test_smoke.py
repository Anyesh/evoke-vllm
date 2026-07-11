import pytest

pytest.importorskip("vllm")

from vllm.v1.kv_offload.base import OffloadingSpec  # noqa: E402
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager  # noqa: E402
from vllm.v1.kv_offload.cpu.policies.base import CachePolicy  # noqa: E402
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec  # noqa: E402

from evoke_vllm.manager import EvokeOffloadingManager  # noqa: E402
from evoke_vllm.policy import EvokeCachePolicy  # noqa: E402
from evoke_vllm.spec import EvokeOffloadingSpec  # noqa: E402


def test_spec_subclasses_stock_cpu_spec():
    assert issubclass(EvokeOffloadingSpec, CPUOffloadingSpec)
    assert issubclass(EvokeOffloadingSpec, OffloadingSpec)


def test_manager_subclasses_stock_cpu_manager():
    assert issubclass(EvokeOffloadingManager, CPUOffloadingManager)


def test_policy_subclasses_stock_cache_policy():
    assert issubclass(EvokeCachePolicy, CachePolicy)


def test_spec_overrides_get_manager():
    assert "get_manager" in EvokeOffloadingSpec.__dict__


def test_manager_overrides_prepare_store():
    assert "prepare_store" in EvokeOffloadingManager.__dict__
