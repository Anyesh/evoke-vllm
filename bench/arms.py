"""Per-arm server config: kv-transfer-config JSON and the exact serve command.

Mirrors ``scripts/serve.sh``'s conventions (same ``EVOKE_*`` profile keys,
same ``block_size``/``store_threshold`` defaults, same offload-block-size
divisibility check) so the two tools describe one consistent server, not two
independently-drifting ones. ``serve.sh`` only ever builds baseline (A0) or
single-connector EVOKE (A2); this module adds the stock-LRU control (A1),
the MultiConnector composition (A3), and the LMCache-alone fallback (A4)
that the run matrix needs, and parameterizes ``cpu_bytes_to_use`` per budget
instead of taking one fixed value from the profile.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ARM_STOCK = "A0"
ARM_OFFLOAD_LRU = "A1"
ARM_OFFLOAD_EVOKE = "A2"
ARM_MULTI_EVOKE_LMCACHE = "A3"
ARM_LMCACHE_ALONE = "A4"

ARMS_REQUIRING_BUDGET = {ARM_OFFLOAD_LRU, ARM_OFFLOAD_EVOKE, ARM_MULTI_EVOKE_LMCACHE}
ARMS_REQUIRING_LMCACHE = {ARM_MULTI_EVOKE_LMCACHE, ARM_LMCACHE_ALONE}
ARMS_REQUIRING_SMOKE_TEST = {ARM_MULTI_EVOKE_LMCACHE}

OFFLOAD_BLOCK_SIZE = 64
STORE_THRESHOLD = 2

DEFAULT_EVOKE_TUNING = {
    "w_recency": 0.5,
    "w_reuse": 0.5,
    "recency_half_life": 64,
    "source_floors": {"system": 0.6, "user": 0.6, "assistant": 0.5},
}


class ProfileError(ValueError):
    pass


@dataclass(frozen=True)
class ServeProfile:
    model: str
    served_model_name: str
    host: str
    port: int
    dtype: str
    max_model_len: int
    block_size: int
    gpu_memory_utilization: float
    quantization: str = ""
    offload_block_size: int = OFFLOAD_BLOCK_SIZE
    store_threshold: int = STORE_THRESHOLD
    w_recency: float = 0.5
    w_reuse: float = 0.5
    recency_half_life: int = 64

    def __post_init__(self) -> None:
        if self.offload_block_size % self.block_size != 0:
            raise ProfileError(
                f"offload_block_size ({self.offload_block_size}) must be a "
                f"multiple of block_size ({self.block_size})"
            )

    @classmethod
    def from_env(cls, env: dict[str, str]) -> ServeProfile:
        def required(key: str) -> str:
            value = env.get(key)
            if not value:
                raise ProfileError(f"missing required profile key {key}")
            return value

        model = required("EVOKE_MODEL")
        return cls(
            model=model,
            served_model_name=env.get("EVOKE_SERVED_MODEL_NAME") or model,
            host=required("EVOKE_HOST"),
            port=int(required("EVOKE_PORT")),
            dtype=required("EVOKE_DTYPE"),
            max_model_len=int(required("EVOKE_MAX_MODEL_LEN")),
            block_size=int(required("EVOKE_BLOCK_SIZE")),
            gpu_memory_utilization=float(required("EVOKE_GPU_MEMORY_UTILIZATION")),
            quantization=env.get("EVOKE_QUANTIZATION", ""),
            offload_block_size=int(
                env.get("EVOKE_OFFLOAD_BLOCK_SIZE", OFFLOAD_BLOCK_SIZE)
            ),
            store_threshold=int(env.get("EVOKE_STORE_THRESHOLD", STORE_THRESHOLD)),
            w_recency=float(env.get("EVOKE_W_RECENCY", 0.5)),
            w_reuse=float(env.get("EVOKE_W_REUSE", 0.5)),
            recency_half_life=int(env.get("EVOKE_RECENCY_HALF_LIFE", 64)),
        )


def parse_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        env[key] = value
    return env


def load_profile(path: Path) -> ServeProfile:
    return ServeProfile.from_env(parse_env_file(path))


def _offloading_extra_config(
    cpu_bytes_to_use: int,
    profile: ServeProfile,
    *,
    spec_name: str | None = None,
    spec_module_path: str | None = None,
    evoke_tuning: dict | None = None,
) -> dict:
    config: dict = {
        "cpu_bytes_to_use": cpu_bytes_to_use,
        "block_size": profile.offload_block_size,
        "store_threshold": profile.store_threshold,
        "offload_prompt_only": True,
    }
    if spec_name is not None:
        config["spec_name"] = spec_name
        config["spec_module_path"] = spec_module_path
    if evoke_tuning is not None:
        config["evoke"] = evoke_tuning
    return config


def _evoke_tuning_from_profile(profile: ServeProfile) -> dict:
    return {
        "w_recency": profile.w_recency,
        "w_reuse": profile.w_reuse,
        "recency_half_life": profile.recency_half_life,
        "source_floors": DEFAULT_EVOKE_TUNING["source_floors"],
    }


def kv_transfer_config_for_arm(
    arm_id: str, profile: ServeProfile, cpu_bytes_to_use: int | None
) -> dict | None:
    if arm_id == ARM_STOCK:
        return None

    if arm_id in ARMS_REQUIRING_BUDGET and cpu_bytes_to_use is None:
        raise ValueError(f"arm {arm_id} requires cpu_bytes_to_use")

    if arm_id == ARM_OFFLOAD_LRU:
        return {
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": _offloading_extra_config(
                cpu_bytes_to_use, profile
            ),
        }

    if arm_id == ARM_OFFLOAD_EVOKE:
        return {
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": _offloading_extra_config(
                cpu_bytes_to_use,
                profile,
                spec_name="EvokeOffloadingSpec",
                spec_module_path="evoke_vllm.spec",
                evoke_tuning=_evoke_tuning_from_profile(profile),
            ),
        }

    if arm_id == ARM_MULTI_EVOKE_LMCACHE:
        evoke_child = {
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": _offloading_extra_config(
                cpu_bytes_to_use,
                profile,
                spec_name="EvokeOffloadingSpec",
                spec_module_path="evoke_vllm.spec",
                evoke_tuning=_evoke_tuning_from_profile(profile),
            ),
        }
        lmcache_child = {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}
        return {
            "kv_connector": "MultiConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {"connectors": [evoke_child, lmcache_child]},
        }

    if arm_id == ARM_LMCACHE_ALONE:
        return {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}

    raise ValueError(f"unknown arm {arm_id!r}")


def serve_command(
    profile: ServeProfile,
    arm_id: str,
    cpu_bytes_to_use: int | None,
    *,
    repo_root: str = ".",
    extra_args: tuple[str, ...] = (),
) -> list[str]:
    cmd = [
        "uv",
        "run",
        "--project",
        repo_root,
        "vllm",
        "serve",
        profile.model,
        "--host",
        profile.host,
        "--port",
        str(profile.port),
        "--served-model-name",
        profile.served_model_name,
        "--dtype",
        profile.dtype,
        "--max-model-len",
        str(profile.max_model_len),
        "--block-size",
        str(profile.block_size),
        "--gpu-memory-utilization",
        str(profile.gpu_memory_utilization),
    ]
    if profile.quantization:
        cmd += ["--quantization", profile.quantization]

    kv_transfer_config = kv_transfer_config_for_arm(arm_id, profile, cpu_bytes_to_use)
    if kv_transfer_config is not None:
        cmd += ["--kv-transfer-config", json.dumps(kv_transfer_config)]

    cmd += list(extra_args)
    return cmd


def base_url(profile: ServeProfile) -> str:
    host = "127.0.0.1" if profile.host == "0.0.0.0" else profile.host
    return f"http://{host}:{profile.port}"
