#!/usr/bin/env bash
set -euo pipefail

# Run OpenAI Batch shard ablations sequentially:
#   AITA-YTA -> AITA-NTA-OG -> AITA-NTA-FLIP
#
# Usage:
#   ./run_shard_ablation_batches.sh
#
# Optional environment overrides:
#   MODEL=gpt-5.4-mini
#   OUT_DIR=decomposition/artifacts
#   TARGET_TURNS=4,6,8
#   POLL_SECONDS=900
#   MAX_COLLECT_ATTEMPTS=120
#   RESUBMIT=1   # submit again even if a batch_state file already exists

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set. Export it or put it in .env before running." >&2
  exit 1
fi

MODEL="${MODEL:-gpt-5.4-mini}"
OUT_DIR="${OUT_DIR:-decomposition/artifacts}"
TARGET_TURNS="${TARGET_TURNS:-4,6,8}"
POLL_SECONDS="${POLL_SECONDS:-900}"
MAX_COLLECT_ATTEMPTS="${MAX_COLLECT_ATTEMPTS:-120}"
RESUBMIT="${RESUBMIT:-0}"

mkdir -p "$OUT_DIR"

submit_dataset() {
  local dataset_name="$1"
  local dataset_path="$2"
  local source_field="$3"
  local run_id="$4"
  local seed_path="${5:-}"

  local state_path="${OUT_DIR}/shard_ablation.${dataset_name}.openai.batch_state.json"
  local input_path="${OUT_DIR}/shard_ablation.${dataset_name}.openai.batch_input.jsonl"

  if [[ -f "$state_path" && "$RESUBMIT" != "1" ]]; then
    echo "[$dataset_name] Existing state found; skipping submit: $state_path"
    return 0
  fi

  if [[ -n "$seed_path" && ! -f "$seed_path" ]]; then
    echo "[$dataset_name] Seed shard artifact is missing: $seed_path" >&2
    exit 1
  fi

  echo "[$dataset_name] Submitting OpenAI Batch..."
  local submit_cmd=(
    uv run --project decomposition python -m decomposition.cli shard-ablation-batch submit
    --dataset "$dataset_path"
    --dataset-name "$dataset_name"
    --source-field "$source_field"
    --run-id "$run_id"
    --provider openai
    --model "$MODEL"
    --target-turns "$TARGET_TURNS"
    --out-dir "$OUT_DIR"
    --batch-state "$state_path"
    --batch-input "$input_path"
  )
  if [[ -n "$seed_path" ]]; then
    submit_cmd+=(--seed-shards "$seed_path")
  fi
  "${submit_cmd[@]}"
}

collect_until_ready() {
  local dataset_name="$1"
  local state_path="${OUT_DIR}/shard_ablation.${dataset_name}.openai.batch_state.json"
  local log_path="${OUT_DIR}/shard_ablation.${dataset_name}.openai.collect.log"
  local attempt=1

  while (( attempt <= MAX_COLLECT_ATTEMPTS )); do
    echo "[$dataset_name] Collect attempt ${attempt}/${MAX_COLLECT_ATTEMPTS}..."
    set +e
    uv run --project decomposition python -m decomposition.cli shard-ablation-batch collect \
      --batch-state "$state_path" 2>&1 | tee "$log_path"
    local status="${PIPESTATUS[0]}"
    set -e

    if (( status != 0 )); then
      echo "[$dataset_name] Collect failed; see $log_path" >&2
      exit "$status"
    fi

    if ! grep -q "results=not_ready" "$log_path"; then
      echo "[$dataset_name] Collect completed."
      return 0
    fi

    echo "[$dataset_name] Batch not ready; sleeping ${POLL_SECONDS}s."
    sleep "$POLL_SECONDS"
    attempt=$((attempt + 1))
  done

  echo "[$dataset_name] Timed out waiting for batch completion after ${MAX_COLLECT_ATTEMPTS} attempts." >&2
  exit 1
}

run_dataset() {
  local dataset_name="$1"
  local dataset_path="$2"
  local source_field="$3"
  local run_id="$4"
  local seed_path="${5:-}"

  submit_dataset "$dataset_name" "$dataset_path" "$source_field" "$run_id" "$seed_path"
  collect_until_ready "$dataset_name"
}

run_dataset \
  "AITA-YTA" \
  "datasets/AITA-YTA.csv" \
  "prompt" \
  "aita-yta-openai-ablation-v1" \
  "decomposition/artifacts/shards.AITA-YTA.openai.smoke.jsonl"

run_dataset \
  "AITA-NTA-OG" \
  "datasets/AITA-NTA-OG.csv" \
  "original_post" \
  "aita-nta-og-openai-ablation-v1" \
  "decomposition/artifacts/shards.AITA-NTA-OG.anthropic.jsonl"

run_dataset \
  "AITA-NTA-FLIP" \
  "datasets/AITA-NTA-FLIP.csv" \
  "flipped_story" \
  "aita-nta-flip-openai-ablation-v1"

echo "All shard ablation batches completed."
