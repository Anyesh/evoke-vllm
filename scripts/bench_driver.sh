#!/bin/bash
# Drives the copy-paste plan bench-plan.txt renders: starts each group's
# server, health-polls it, runs the group's cells, then kills the server
# and waits for VRAM to actually release before the next group (observed
# on the 4070 Ti: launching into residual VRAM shrinks the KV pool by up
# to ~9k tokens, changing the cell's effective config). Exits 2 if the
# smoke test (first cell in the plan) fails, per spec 02a section 3, so
# the operator can flip A3 to A4 in matrix.toml and regenerate the plan.
set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLAN="${1:-$REPO_ROOT/bench-plan.txt}"
LOG="$REPO_ROOT/bench/results/driver.log"
cd "$REPO_ROOT"
mkdir -p bench/results
failed=0
ran=0

wait_health() {
  for _ in $(seq 1 90); do
    code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health 2>/dev/null)
    [ "$code" = "200" ] && return 0
    pgrep -f 'vllm serve' >/dev/null || { echo "SERVER DIED during boot" >>"$LOG"; return 1; }
    sleep 10
  done
  return 1
}

settle() {
  pkill -f 'vllm serve' 2>/dev/null
  for _ in $(seq 1 40); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$used" -lt 1200 ] && return 0
    sleep 5
  done
  echo "WARN: VRAM did not settle below 1200MiB" >>"$LOG"
}

while IFS= read -r line; do
  case "$line" in
    "uv run --project"*"vllm serve"*)
      settle
      echo "[$(date +%H:%M:%S)] SERVER: ${line:0:120}..." >>"$LOG"
      nohup bash -c "$line" >>"$REPO_ROOT/bench/results/server-$(date +%s).log" 2>&1 &
      if ! wait_health; then
        echo "[$(date +%H:%M:%S)] HEALTH FAIL, group cells will fail" >>"$LOG"
        failed=$((failed+1))
      fi
      ;;
    "uv run python -m bench run-cell"*)
      echo "[$(date +%H:%M:%S)] CELL: $line" >>"$LOG"
      if bash -c "$line" >>"$LOG" 2>&1; then
        ran=$((ran+1))
      else
        failed=$((failed+1))
        echo "[$(date +%H:%M:%S)] CELL FAILED" >>"$LOG"
        if [ "$ran" -eq 0 ]; then
          settle
          echo "SMOKE-FAILED" | tee -a "$LOG"
          exit 2
        fi
      fi
      ;;
  esac
done < "$PLAN"
settle
echo "DONE ran=$ran failed=$failed" | tee -a "$LOG"
