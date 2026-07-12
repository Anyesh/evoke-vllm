"""Config surface for evoke_vllm.

Two channels carry EVOKE tuning:

- ``EvokeScoringConfig`` is operator-level tuning read from the connector's
  ``kv_connector_extra_config`` under an ``evoke`` sub-key, with env-var
  overrides for ops.
- ``EvokeRequestTags`` is per-request, client-supplied structure read from
  ``kv_transfer_params["evoke"]``, coexisting with stock keys such as
  ``max_offload_tokens``.

Env-var overrides (prefix ``EVOKE_``, per-role floors under ``EVOKE_FLOOR_``)
take precedence over the ``extra_config`` values so ops can retune a running
deployment without editing the connector config.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

EVOKE_EXTRA_CONFIG_KEY = "evoke"
EVOKE_KV_TRANSFER_PARAMS_KEY = "evoke"
EVOKE_ENV_PREFIX = "EVOKE_"
EVOKE_FLOOR_ENV_PREFIX = "EVOKE_FLOOR_"

SOURCE_SYSTEM = "system"
SOURCE_USER = "user"
SOURCE_ASSISTANT = "assistant"
SOURCE_DOCUMENT = "document"

DEFAULT_SOURCE_FLOORS: dict[str, float] = {
    SOURCE_SYSTEM: 0.6,
    SOURCE_USER: 0.6,
    SOURCE_ASSISTANT: 0.5,
}


@dataclass
class EvokeScoringConfig:
    """P1 scoring recipe: a weighted sum of recency and reuse, lifted to a
    source-role floor and multiplied by a per-request priority.

    ``w_attention`` and ``w_coherence`` are carried at zero. They are
    RFC-track and have no stock vLLM signal to populate them
    with in P1; the fields exist so enabling them later is a configuration
    change, not a rewrite.
    """

    w_recency: float = 0.5
    w_reuse: float = 0.5
    w_attention: float = 0.0
    w_coherence: float = 0.0
    recency_half_life: int = 64
    source_floors: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_SOURCE_FLOORS)
    )

    @classmethod
    def from_extra_config(cls, extra_config: dict[str, Any]) -> EvokeScoringConfig:
        raw = extra_config.get(EVOKE_EXTRA_CONFIG_KEY) if extra_config else None
        evoke: dict[str, Any] = raw if isinstance(raw, dict) else {}
        config = cls()
        if "w_recency" in evoke:
            config.w_recency = float(evoke["w_recency"])
        if "w_reuse" in evoke:
            config.w_reuse = float(evoke["w_reuse"])
        if "w_attention" in evoke:
            config.w_attention = float(evoke["w_attention"])
        if "w_coherence" in evoke:
            config.w_coherence = float(evoke["w_coherence"])
        if "recency_half_life" in evoke:
            config.recency_half_life = int(evoke["recency_half_life"])
        floors = evoke.get("source_floors")
        if isinstance(floors, dict):
            config.source_floors = {str(k): float(v) for k, v in floors.items()}
        return cls.apply_env_overrides(config)

    @classmethod
    def apply_env_overrides(cls, base: EvokeScoringConfig) -> EvokeScoringConfig:
        env = os.environ

        def _float_override(name: str, current: float) -> float:
            value = env.get(name)
            return float(value) if value is not None else current

        base.w_recency = _float_override(EVOKE_ENV_PREFIX + "W_RECENCY", base.w_recency)
        base.w_reuse = _float_override(EVOKE_ENV_PREFIX + "W_REUSE", base.w_reuse)
        base.w_attention = _float_override(
            EVOKE_ENV_PREFIX + "W_ATTENTION", base.w_attention
        )
        base.w_coherence = _float_override(
            EVOKE_ENV_PREFIX + "W_COHERENCE", base.w_coherence
        )
        half_life = env.get(EVOKE_ENV_PREFIX + "RECENCY_HALF_LIFE")
        if half_life is not None:
            base.recency_half_life = int(half_life)
        for name, value in env.items():
            if name.startswith(EVOKE_FLOOR_ENV_PREFIX):
                role = name[len(EVOKE_FLOOR_ENV_PREFIX) :].lower()
                if role:
                    base.source_floors[role] = float(value)
        return base


@dataclass
class EvokeRequestTags:
    """Per-request tags read from ``kv_transfer_params["evoke"]``.

    Client-cooperative: absent tags degrade cleanly to recency plus reuse,
    which is the tested default path, not a failure mode.
    """

    source_type: str | None = None
    priority: float = 1.0
    evoke_session: str | None = None

    @classmethod
    def from_kv_transfer_params(
        cls, kv_transfer_params: dict[str, Any] | None
    ) -> EvokeRequestTags:
        if not kv_transfer_params:
            return cls()
        raw = kv_transfer_params.get(EVOKE_KV_TRANSFER_PARAMS_KEY)
        if not isinstance(raw, dict):
            return cls()
        source_type = raw.get("source_type")
        evoke_session = raw.get("evoke_session")
        return cls(
            source_type=None if source_type is None else str(source_type),
            priority=float(raw.get("priority", 1.0)),
            evoke_session=None if evoke_session is None else str(evoke_session),
        )
