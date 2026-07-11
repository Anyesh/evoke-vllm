#!/usr/bin/env bash
# Launches vllm serve for a hardware profile, with the EVOKE offload
# connector wired in by default or stripped out entirely under --baseline.
#
# Usage:
#   scripts/serve.sh --profile local-2060
#   scripts/serve.sh --profile wsl2-4070ti
#   scripts/serve.sh --profile local-2060 --baseline
#   scripts/serve.sh --profile local-2060 --dry-run
#   scripts/serve.sh --profile local-2060 --extra-arg --enforce-eager
#
# Without --profile, all EVOKE_* variables listed in profiles/*.env must
# already be exported in the shell; --profile just sources the matching
# profiles/<name>.env file first. --dry-run prints the resolved
# kv-transfer-config JSON and the exact "uv run vllm serve ..." command
# without executing it or requiring vllm/a GPU to be available, so it can be
# validated CPU-side and copy-pasted onto the real hardware later.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROFILES_DIR="${REPO_ROOT}/profiles"

usage() {
    cat <<'EOF'
Usage: serve.sh [--profile NAME] [--baseline] [--dry-run]
                 [--extra-arg ARG]... [-h|--help]

  --profile NAME   Source profiles/NAME.env before building the command.
  --baseline       Omit --kv-transfer-config entirely (stock vLLM, no
                    offload connector, no CPU tier).
  --dry-run        Print the resolved config and command; do not exec.
  --extra-arg ARG  Append ARG to the vllm serve invocation. Repeatable.
EOF
}

PROFILE=""
BASELINE=0
DRY_RUN=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)
            PROFILE="$2"
            shift 2
            ;;
        --baseline)
            BASELINE=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --extra-arg)
            EXTRA_ARGS+=("$2")
            shift 2
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -n "${PROFILE}" ]]; then
    PROFILE_FILE="${PROFILES_DIR}/${PROFILE}.env"
    if [[ ! -f "${PROFILE_FILE}" ]]; then
        echo "profile not found: ${PROFILE_FILE}" >&2
        echo "available profiles:" >&2
        ls "${PROFILES_DIR}"/*.env 2>/dev/null | xargs -n1 basename >&2 || true
        exit 1
    fi
    # Environment wins over profile values, matching the package's
    # env-over-extra_config precedence, so callers can override single
    # knobs (EVOKE_PORT=8151 scripts/serve.sh ...) without editing profiles.
    PRE_SET_VARS=()
    while IFS= read -r var; do
        PRE_SET_VARS+=("${var}=${!var}")
    done < <(compgen -v EVOKE_ || true)
    set -a
    # shellcheck source=/dev/null
    source "${PROFILE_FILE}"
    set +a
    for pair in "${PRE_SET_VARS[@]}"; do
        export "${pair?}"
    done
fi

REQUIRED_VARS=(
    EVOKE_MODEL
    EVOKE_HOST
    EVOKE_PORT
    EVOKE_DTYPE
    EVOKE_MAX_MODEL_LEN
    EVOKE_BLOCK_SIZE
    EVOKE_GPU_MEMORY_UTILIZATION
    EVOKE_CPU_BYTES_TO_USE
    EVOKE_OFFLOAD_BLOCK_SIZE
    EVOKE_STORE_THRESHOLD
)
MISSING=()
for var in "${REQUIRED_VARS[@]}"; do
    if [[ -z "${!var:-}" ]]; then
        MISSING+=("${var}")
    fi
done
if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "missing required config (set via --profile or export directly): ${MISSING[*]}" >&2
    exit 1
fi

# offloaded_block_size must be a multiple of the engine's KV block size;
# vllm/v1/kv_offload/base.py asserts this inside OffloadingSpec.__init__,
# which only runs after the model has already loaded. Catching it here fails
# in milliseconds instead of after a multi-minute model load.
if ((EVOKE_OFFLOAD_BLOCK_SIZE % EVOKE_BLOCK_SIZE != 0)); then
    echo "EVOKE_OFFLOAD_BLOCK_SIZE (${EVOKE_OFFLOAD_BLOCK_SIZE}) must be a multiple of EVOKE_BLOCK_SIZE (${EVOKE_BLOCK_SIZE})" >&2
    exit 1
fi

EVOKE_SERVED_MODEL_NAME="${EVOKE_SERVED_MODEL_NAME:-${EVOKE_MODEL}}"
EVOKE_QUANTIZATION="${EVOKE_QUANTIZATION:-}"
EVOKE_W_RECENCY="${EVOKE_W_RECENCY:-0.5}"
EVOKE_W_REUSE="${EVOKE_W_REUSE:-0.5}"
EVOKE_RECENCY_HALF_LIFE="${EVOKE_RECENCY_HALF_LIFE:-64}"

VLLM_CMD=(
    uv run --project "${REPO_ROOT}" vllm serve "${EVOKE_MODEL}"
    --host "${EVOKE_HOST}"
    --port "${EVOKE_PORT}"
    --served-model-name "${EVOKE_SERVED_MODEL_NAME}"
    --dtype "${EVOKE_DTYPE}"
    --max-model-len "${EVOKE_MAX_MODEL_LEN}"
    --block-size "${EVOKE_BLOCK_SIZE}"
    --gpu-memory-utilization "${EVOKE_GPU_MEMORY_UTILIZATION}"
)

if [[ -n "${EVOKE_QUANTIZATION}" ]]; then
    VLLM_CMD+=(--quantization "${EVOKE_QUANTIZATION}")
fi

KV_TRANSFER_CONFIG=""
if [[ "${BASELINE}" -eq 0 ]]; then
    # spec_name/spec_module_path select evoke_vllm.spec.EvokeOffloadingSpec
    # through vLLM's dynamic-import route (OffloadingSpecFactory.create_spec)
    # instead of the hardcoded stock cache-policy names; see
    # evoke_vllm/spec.py and design spec 01a section 5.
    KV_TRANSFER_CONFIG=$(
        cat <<JSON
{"kv_connector": "OffloadingConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"spec_name": "EvokeOffloadingSpec", "spec_module_path": "evoke_vllm.spec", "cpu_bytes_to_use": ${EVOKE_CPU_BYTES_TO_USE}, "block_size": ${EVOKE_OFFLOAD_BLOCK_SIZE}, "store_threshold": ${EVOKE_STORE_THRESHOLD}, "offload_prompt_only": true, "evoke": {"w_recency": ${EVOKE_W_RECENCY}, "w_reuse": ${EVOKE_W_REUSE}, "recency_half_life": ${EVOKE_RECENCY_HALF_LIFE}}}}
JSON
    )
    VLLM_CMD+=(--kv-transfer-config "${KV_TRANSFER_CONFIG}")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    VLLM_CMD+=("${EXTRA_ARGS[@]}")
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "# profile: ${PROFILE:-<none, using exported env>}"
    if [[ "${BASELINE}" -eq 1 ]]; then
        echo "# mode: baseline (no connector)"
    else
        echo "# mode: evoke"
        echo "# kv-transfer-config JSON:"
        echo "${KV_TRANSFER_CONFIG}"
    fi
    echo "# command:"
    printf '%q ' "${VLLM_CMD[@]}"
    echo
    exit 0
fi

exec "${VLLM_CMD[@]}"
