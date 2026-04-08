#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUN_ID_WAS_PROVIDED=0
if [[ -n "${RUN_ID:-}" ]]; then
  RUN_ID_WAS_PROVIDED=1
fi
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$ROOT_DIR/pipeline_runs_v2/$RUN_ID"
LOG_FILE="$LOG_DIR/pipeline_v2.log"
STATE_FILE="$LOG_DIR/completed_steps.txt"
mkdir -p "$LOG_DIR"
touch "$STATE_FILE"

PYTHON_BIN="${PYTHON_BIN:-python}"
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
OLLAMA_AUTOSTART="${OLLAMA_AUTOSTART:-1}"
OLLAMA_START_TIMEOUT_SECONDS="${OLLAMA_START_TIMEOUT_SECONDS:-20}"
OLLAMA_START_LOG="${OLLAMA_START_LOG:-/tmp/ollama_serve.log}"

MAX_STEP_RETRIES="${MAX_STEP_RETRIES:-3}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-60}"
RETRY_BACKOFF_MULTIPLIER="${RETRY_BACKOFF_MULTIPLIER:-2}"
SKIP_COMPLETED_ON_RERUN="${SKIP_COMPLETED_ON_RERUN:-1}"

S1_EXTRA_ARGS="${S1_EXTRA_ARGS:-}"
S2_EXTRA_ARGS="${S2_EXTRA_ARGS:---concurrency 4}"
S3_EXTRA_ARGS="${S3_EXTRA_ARGS:-}"
MAIN_EXTRA_ARGS="${MAIN_EXTRA_ARGS:-}"
ASSET_EXTRA_ARGS="${ASSET_EXTRA_ARGS:-}"

S1_RETRY_ARGS="${S1_RETRY_ARGS:---resume}"
S2_RETRY_ARGS="${S2_RETRY_ARGS:---resume --concurrency 4}"
S3_RETRY_ARGS="${S3_RETRY_ARGS:---resume}"
MAIN_RETRY_ARGS="${MAIN_RETRY_ARGS:-}"
ASSET_RETRY_ARGS="${ASSET_RETRY_ARGS:-}"

AUTO_RESUME_ON_EXISTING_RUN="${AUTO_RESUME_ON_EXISTING_RUN:-1}"
AUTO_RESUME_ACTIVE=0
if [[ "$AUTO_RESUME_ON_EXISTING_RUN" == "1" && "$RUN_ID_WAS_PROVIDED" == "1" ]]; then
  if [[ -s "$LOG_FILE" ]] || [[ -s "$STATE_FILE" ]]; then
    AUTO_RESUME_ACTIVE=1
  fi
fi

STORIES_V2_DIR="$ROOT_DIR/drive-download-20260219T230708Z-1-001/data_stories_v2"
FUTURE_STORIES_V2_DIR="$ROOT_DIR/drive-download-20260219T230708Z-1-001/data_stories_future_v2"
EMBEDDINGS_V2_DIR="$ROOT_DIR/drive-download-20260219T230708Z-1-001/data_story_embeddings_v2"

exec > >(tee -a "$LOG_FILE") 2>&1

progress_bar() {
  local n="$1"
  local total="$2"
  local width=30
  local filled=$(( n * width / total ))
  local empty=$(( width - filled ))
  local bar
  bar="$(printf '%*s' "$filled" '' | tr ' ' '#')$(printf '%*s' "$empty" '' | tr ' ' '-')"
  echo "PIPELINE [$bar] ${n}/${total}"
}

ollama_ready() {
  curl -sSf "$OLLAMA_URL/api/tags" >/dev/null 2>&1
}

ensure_ollama() {
  if ollama_ready; then
    return 0
  fi

  if [[ "$OLLAMA_AUTOSTART" != "1" ]]; then
    echo "ERROR: Ollama not reachable at $OLLAMA_URL"
    echo "Hint: start it manually with: ollama serve"
    return 1
  fi

  if ! command -v ollama >/dev/null 2>&1; then
    echo "ERROR: Ollama CLI not found in PATH; cannot auto-start."
    return 1
  fi

  echo "Ollama not reachable at $OLLAMA_URL; attempting auto-start..."
  nohup ollama serve >"$OLLAMA_START_LOG" 2>&1 &

  local waited=0
  while (( waited < OLLAMA_START_TIMEOUT_SECONDS )); do
    if ollama_ready; then
      echo "Ollama is ready."
      return 0
    fi
    sleep 1
    waited=$(( waited + 1 ))
  done

  echo "ERROR: Ollama still not reachable at $OLLAMA_URL after ${OLLAMA_START_TIMEOUT_SECONDS}s."
  echo "Ollama start log: $OLLAMA_START_LOG"
  if [[ -f "$OLLAMA_START_LOG" ]]; then
    echo "Last 20 lines of Ollama log:"
    tail -n 20 "$OLLAMA_START_LOG"
  fi
  return 1
}

is_completed() {
  grep -Fxq "$1" "$STATE_FILE"
}

mark_completed() {
  if ! is_completed "$1"; then
    echo "$1" >> "$STATE_FILE"
  fi
}

retry_args_for_step() {
  case "$1" in
    stage1) echo "$S1_RETRY_ARGS" ;;
    stage2) echo "$S2_RETRY_ARGS" ;;
    stage3) echo "$S3_RETRY_ARGS" ;;
    train) echo "$MAIN_RETRY_ARGS" ;;
    assets) echo "$ASSET_RETRY_ARGS" ;;
    *) echo "" ;;
  esac
}

initial_args_for_step() {
  case "$1" in
    stage1)
      if [[ "$AUTO_RESUME_ACTIVE" == "1" ]] || compgen -G "$STORIES_V2_DIR/stories_*.jsonl" >/dev/null; then
        echo "$S1_RETRY_ARGS"
      else
        echo ""
      fi
      ;;
    stage2)
      if [[ "$AUTO_RESUME_ACTIVE" == "1" ]] || [[ -f "$FUTURE_STORIES_V2_DIR/future_stories.jsonl" ]]; then
        echo "$S2_RETRY_ARGS"
      else
        echo ""
      fi
      ;;
    stage3)
      if [[ "$AUTO_RESUME_ACTIVE" == "1" ]] || compgen -G "$EMBEDDINGS_V2_DIR/data_*/checkpoint.json" >/dev/null; then
        echo "$S3_RETRY_ARGS"
      else
        echo ""
      fi
      ;;
    *) echo "" ;;
  esac
}

run_step() {
  local step_id="$1"
  local step_name="$2"
  shift 2
  local cmd=("$@")

  if [[ "$SKIP_COMPLETED_ON_RERUN" == "1" ]] && is_completed "$step_id"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP: $step_name"
    return 0
  fi

  local attempt=1
  local sleep_s="$RETRY_SLEEP_SECONDS"
  while (( attempt <= MAX_STEP_RETRIES )); do
    local run_cmd=("${cmd[@]}")
    local extra_args=""
    if (( attempt == 1 )); then
      extra_args="$(initial_args_for_step "$step_id")"
    else
      local retry_extra
      retry_extra="$(retry_args_for_step "$step_id")"
      extra_args="$retry_extra"
    fi
    if [[ -n "$extra_args" ]]; then
        local retry_arr=()
        # shellcheck disable=SC2206
        retry_arr=( $extra_args )
        run_cmd+=("${retry_arr[@]}")
    fi

    echo
    echo "=================================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] START: $step_name (attempt $attempt/$MAX_STEP_RETRIES)"
    echo "CMD: ${run_cmd[*]}"
    echo "=================================================================="

    if "${run_cmd[@]}"; then
      mark_completed "$step_id"
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE: $step_name"
      return 0
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAIL: $step_name (attempt $attempt/$MAX_STEP_RETRIES)"
    if (( attempt == MAX_STEP_RETRIES )); then
      echo "Retries exhausted for $step_name"
      return 1
    fi
    echo "Retrying in ${sleep_s}s..."
    sleep "$sleep_s"
    sleep_s=$(( sleep_s * RETRY_BACKOFF_MULTIPLIER ))
    attempt=$(( attempt + 1 ))
  done
}

echo "Run ID: $RUN_ID"
echo "Log file: $LOG_FILE"
echo "State file: $STATE_FILE"
echo "To resume later: RUN_ID=$RUN_ID bash run_full_pipeline_v2.sh"
if [[ "$AUTO_RESUME_ACTIVE" == "1" ]]; then
  echo "Existing RUN_ID detected; auto-resume is enabled for stages 1-3 on first attempt."
fi
if compgen -G "$STORIES_V2_DIR/stories_*.jsonl" >/dev/null; then
  echo "Existing stage 1 outputs detected; stage 1 will auto-resume on first attempt."
fi
if [[ -f "$FUTURE_STORIES_V2_DIR/future_stories.jsonl" ]]; then
  echo "Existing stage 2 outputs detected; stage 2 will auto-resume on first attempt."
fi
if compgen -G "$EMBEDDINGS_V2_DIR/data_*/checkpoint.json" >/dev/null; then
  echo "Existing stage 3 checkpoints detected; stage 3 will auto-resume on first attempt."
fi
TOTAL_STEPS=5
DONE_STEPS=0
progress_bar "$DONE_STEPS" "$TOTAL_STEPS"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: python not found: $PYTHON_BIN"
  exit 1
fi
if ! ensure_ollama; then
  exit 1
fi

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

run_step "stage1" "V2 Stage 1: year stories" "$PYTHON_BIN" generate_year_stories_v2.py "${S1_ARGS[@]}"
DONE_STEPS=$((DONE_STEPS + 1))
progress_bar "$DONE_STEPS" "$TOTAL_STEPS"
run_step "stage2" "V2 Stage 2: future stories" "$PYTHON_BIN" generate_future_stories_v2.py "${S2_ARGS[@]}"
DONE_STEPS=$((DONE_STEPS + 1))
progress_bar "$DONE_STEPS" "$TOTAL_STEPS"
run_step "stage3" "V2 Stage 3: story embeddings" "$PYTHON_BIN" stories_to_embeddings_v2.py "${S3_ARGS[@]}"
DONE_STEPS=$((DONE_STEPS + 1))
progress_bar "$DONE_STEPS" "$TOTAL_STEPS"
run_step "train" "V2 training + eval" "$PYTHON_BIN" main_v2.py "${M_ARGS[@]}"
DONE_STEPS=$((DONE_STEPS + 1))
progress_bar "$DONE_STEPS" "$TOTAL_STEPS"
run_step "assets" "V2 presentation assets" "$PYTHON_BIN" generate_presentation_assets_v2.py "${A_ARGS[@]}"
DONE_STEPS=$((DONE_STEPS + 1))
progress_bar "$DONE_STEPS" "$TOTAL_STEPS"

echo "V2 pipeline completed successfully."
