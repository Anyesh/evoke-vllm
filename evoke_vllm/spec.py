"""EVOKE offloading spec: the injection point vLLM's factory loads.

``OffloadingSpecFactory.create_spec`` (``vllm/v1/kv_offload/factory.py``)
reads ``spec_name`` and ``spec_module_path`` from ``kv_connector_extra_config``
and dynamically imports the module. This is the same route vllm-ascend uses
for ``NPUOffloadingSpec`` and LMCache uses via the sibling
``kv_connector_module_path``; no custom top-level ``KVConnector`` is needed
because the eviction policy lives in the manager, not the connector.
A deployment selects this spec with
``spec_name="EvokeOffloadingSpec"`` and
``spec_module_path="evoke_vllm.spec"`` in ``kv_connector_extra_config``.

``CPUOffloadingSpec.__init__`` already does everything P1 needs (parses
``cpu_bytes_to_use``, computes ``num_blocks``, validates block-size
alignment), so it is inherited unchanged. Only ``get_manager`` differs: it
returns an ``EvokeOffloadingManager`` and parses the EVOKE score recipe from
``kv_connector_extra_config`` instead of going through the stock
``cache_policy`` selector.
"""

from __future__ import annotations

from vllm.v1.kv_offload.base import OffloadingManager
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

from evoke_vllm.config import EvokeScoringConfig
from evoke_vllm.manager import EvokeOffloadingManager


class EvokeOffloadingSpec(CPUOffloadingSpec):
    def get_manager(self) -> OffloadingManager:
        if self._manager is None:
            store_threshold = int(self.extra_config.get("store_threshold", 0))
            max_tracker_size = int(self.extra_config.get("max_tracker_size", 64_000))
            self._manager = EvokeOffloadingManager(
                num_blocks=self.num_blocks,
                enable_events=self.kv_events_config.enable_kv_cache_events,
                store_threshold=store_threshold,
                max_tracker_size=max_tracker_size,
                scoring_config=EvokeScoringConfig.from_extra_config(self.extra_config),
            )
        return self._manager
