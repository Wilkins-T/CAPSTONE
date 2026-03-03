#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$ROOT_DIR/pipeline_runs/$RUN_ID"
LOG_FILE="$LOG_DIR/pipeline.log"
STATE_FILE="$LOG_DIR/completed_steps.txt"
mkdir -p "$LOG_DIR"
touch "$STATE_FILE"

OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Optional extra args (space-separated strings)
S1_EXTRA_ARGS="${S1_EXTRA_ARGS:-}"
S2_EXTRA_ARGS="${S2_EXTRA_ARGS:-}"
S3_EXTRA_ARGS="${S3_EXTRA_ARGS:-}"
MAIN_EXTRA_ARGS="${MAIN_EXTRA_ARGS:-}"
ASSET_EXTRA_ARGS="${ASSET_EXTRA_ARGS:-}"

# Retry/resume controls
MAX_STEP_RETRIES="${MAX_STEP_RETRIES:-3}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-60}"
RETRY_BACKOFF_MULTIPLIER="${RETRY_BACKOFF_MULTIPLIER:-2}"
SKIP_COMPLETED_ON_RERUN="${SKIP_COMPLETED_ON_RERUN:-1}"

# Extra args used only on retry attempts
S1_RETRY_ARGS="${S1_RETRY_ARGS:---resume}"
S2_RETRY_ARGS="${S2_RETRY_ARGS:-}"
S3_RETRY_ARGS="${S3_RETRY_ARGS:---resume}"
MAIN_RETRY_ARGS="${MAIN_RETRY_ARGS:-}"
ASSET_RETRY_ARGS="${ASSET_RETRY_ARGS:-}"

exec > >(tee -a "$LOG_FILE") 2>&1

is_completed() {
  local step_id="$1"
  grep -Fxq "$step_id" "$STATE_FILE"
}

mark_completed() {
  local step_id="$1"
  if ! is_completed "$step_id"; then
    echo "$step_id" >> "$STATE_FILE"
  fi
}

retry_args_for_step() {
  local step_id="$1"
  case "$step_id" in
    stage1) echo "$S1_RETRY_ARGS" ;;
    stage2) echo "$S2_RETRY_ARGS" ;;
    stage3) echo "$S3_RETRY_ARGS" ;;
    train) echo "$MAIN_RETRY_ARGS" ;;
    assets) echo "$ASSET_RETRY_ARGS" ;;
    *) echo "" ;;
  esac
}

run_step() {
  local step_id="$1"
  local step_name="$2"
  shift 2
  local cmd=("$@")

  if [[ "$SKIP_COMPLETED_ON_RERUN" == "1" ]] && is_completed "$step_id"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP: $step_name (already completed in RUN_ID=$RUN_ID)"
    return 0
  fi

  local attempt=1
  local sleep_s="$RETRY_SLEEP_SECONDS"

  while (( attempt <= MAX_STEP_RETRIES )); do
    local run_cmd=("${cmd[@]}")

    if (( attempt > 1 )); then
      local retry_extra
      retry_extra="$(retry_args_for_step "$step_id")"
      if [[ -n "$retry_extra" ]]; then
        local retry_arr=()
        # shellcheck disable=SC2206
        retry_arr=( $retry_extra )
        run_cmd+=("${retry_arr[@]}")
      fi
    fi

    echo
    echo "=================================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] START: $step_name (attempt $attempt/$MAX_STEP_RETRIES)"
    echo "CMD: ${run_cmd[*]}"
    echo "=================================================================="

    if "${run_cmd[@]}"; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE: $step_name"
      mark_completed "$step_id"
      return 0
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAIL: $step_name (attempt $attempt/$MAX_STEP_RETRIES)"
    if (( attempt == MAX_STEP_RETRIES )); then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] ABORT: retries exhausted for $step_name"
      return 1
    fi

    echo "Retrying in ${sleep_s}s..."
    sleep "$sleep_s"
    sleep_s=$(( sleep_s * RETRY_BACKOFF_MULTIPLIER ))
    attempt=$(( attempt + 1 ))
  done
}

echo "Pipeline root: $ROOT_DIR"
echo "Run ID: $RUN_ID"
echo "Log file: $LOG_FILE"
echo "State file: $STATE_FILE"
echo "Python: $PYTHON_BIN"
echo "OLLAMA_URL: $OLLAMA_URL"
echo "Retry policy: max=$MAX_STEP_RETRIES, initial_sleep=${RETRY_SLEEP_SECONDS}s, backoff=x$RETRY_BACKOFF_MULTIPLIER"
echo "Skip completed on rerun: $SKIP_COMPLETED_ON_RERUN"
echo "To resume this run later: RUN_ID=$RUN_ID bash run_full_pipeline.sh"
echo

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: Python binary '$PYTHON_BIN' not found in PATH"
  exit 1
fi

if ! curl -sSf "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
  echo "ERROR: Ollama not reachable at $OLLAMA_URL"
  echo "Start Ollama first (and ensure model is pulled), then rerun."
  exit 1
fi

echo "Preflight checks passed."

# shellcheck disable=SC2206
S1_ARGS=( $S1_EXTRA_ARGS )
# shellcheck disable=SC2206
S2_ARGS=( $S2_EXTRA_ARGS )
# shellcheck disable=SC2206
S3_ARGS=( $S3_EXTRA_ARGS )
# shellcheck disable=SC2206
M_ARGS=( $MAIN_EXTRA_ARGS )
# shellcheck disable=SC2206
A_ARGS=( $ASSET_EXTRA_ARGS )

run_step "stage1" "Stage 1: stories from arrays" "$PYTHON_BIN" generate_custom_data_to_sentence.py "${S1_ARGS[@]}"
run_step "stage2" "Stage 2: future stories" "$PYTHON_BIN" generate_future_stories.py "${S2_ARGS[@]}"
run_step "stage3" "Stage 3: stories to arrays" "$PYTHON_BIN" generate_stories_to_array.py "${S3_ARGS[@]}"
run_step "train" "Training + evaluation" "$PYTHON_BIN" main.py "${M_ARGS[@]}"
run_step "assets" "Presentation artifacts" "$PYTHON_BIN" generate_presentation_assets.py "${A_ARGS[@]}"

echo
echo "Pipeline completed successfully."
echo "Artifacts and logs: $LOG_DIR"
