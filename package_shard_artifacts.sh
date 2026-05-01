#!/usr/bin/env bash
set -euo pipefail

# Build a local, ignored share bundle for the shard ablation artifacts.
#
# Usage:
#   ./package_shard_artifacts.sh
#
# Optional environment overrides:
#   ARTIFACTS_DIR=decomposition/artifacts
#   SHARE_DIR=shared_shard_artifacts
#   ZIP_PATH=shared_shard_artifacts.zip
#   INCLUDE_ERRORS=1
#   SKIP_VALIDATE=0

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

ARTIFACTS_DIR="${ARTIFACTS_DIR:-decomposition/artifacts}"
SHARE_DIR="${SHARE_DIR:-shared_shard_artifacts}"
ZIP_PATH="${ZIP_PATH:-${SHARE_DIR}.zip}"
INCLUDE_ERRORS="${INCLUDE_ERRORS:-1}"
SKIP_VALIDATE="${SKIP_VALIDATE:-0}"

guard_generated_path() {
  local path="$1"
  if [[ -z "$path" || "$path" == "/" || "$path" == "." || "$path" == ".." ]]; then
    echo "Refusing unsafe generated path: '$path'" >&2
    exit 1
  fi
}

copy_required_pattern() {
  local pattern="$1"
  local description="$2"
  local matches=("$ARTIFACTS_DIR"/$pattern)
  if (( ${#matches[@]} == 0 )); then
    echo "Missing required $description artifacts matching: $ARTIFACTS_DIR/$pattern" >&2
    exit 1
  fi
  cp "${matches[@]}" "$SHARE_DIR/"
}

copy_optional_pattern() {
  local pattern="$1"
  local matches=("$ARTIFACTS_DIR"/$pattern)
  if (( ${#matches[@]} == 0 )); then
    return
  fi
  cp "${matches[@]}" "$SHARE_DIR/"
}

checksum_files() {
  if command -v shasum >/dev/null 2>&1; then
    for file in *.jsonl *.json README.md; do
      [[ -e "$file" ]] && shasum -a 256 "$file"
    done > SHA256SUMS.txt
  elif command -v sha256sum >/dev/null 2>&1; then
    for file in *.jsonl *.json README.md; do
      [[ -e "$file" ]] && sha256sum "$file"
    done > SHA256SUMS.txt
  else
    echo "Neither shasum nor sha256sum is available." >&2
    exit 1
  fi
}

if [[ ! -d "$ARTIFACTS_DIR" ]]; then
  echo "Artifact directory does not exist: $ARTIFACTS_DIR" >&2
  exit 1
fi

guard_generated_path "$SHARE_DIR"
guard_generated_path "$ZIP_PATH"

shopt -s nullglob

shard_files=("$ARTIFACTS_DIR"/shards.AITA-*.openai.k*.jsonl)
if (( ${#shard_files[@]} == 0 )); then
  echo "No shard artifacts found under $ARTIFACTS_DIR" >&2
  exit 1
fi

if [[ "$SKIP_VALIDATE" != "1" ]]; then
  for shard in "${shard_files[@]}"; do
    echo "Validating $shard"
    uv run --project decomposition python -m decomposition.cli validate --input "$shard"
  done
fi

rm -rf "$SHARE_DIR"
rm -f "$ZIP_PATH"
mkdir -p "$SHARE_DIR"

copy_required_pattern 'shards.AITA-*.openai.k*.jsonl' 'shard'
copy_required_pattern 'atomic_units.AITA-*.openai.jsonl' 'atomic unit'
copy_required_pattern 'shard_ablation.AITA-*.openai.summary.json' 'summary'
copy_optional_pattern 'shards.AITA-*.openai.k*.eligible_all*.jsonl'
copy_optional_pattern 'cohort.AITA-*.openai.eligible_all*.summary.json'
copy_optional_pattern 'cohorts/shards.AITA-*.openai.k*.eligible_all*.jsonl'
copy_optional_pattern 'cohorts/cohort.AITA-*.openai.eligible_all*.summary.json'
if [[ "$INCLUDE_ERRORS" == "1" ]]; then
  copy_required_pattern 'run_errors.AITA-*.openai.atomic.jsonl' 'atomic error'
fi

generated_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
cat > "$SHARE_DIR/README.md" <<EOF
# AITA Shard Ablation Artifacts

Generated at: $generated_at

This bundle contains OpenAI atomic-unit extraction artifacts and deterministic
shard-count ablations for the class project datasets.

Datasets:
- AITA-YTA
- AITA-NTA-OG
- AITA-NTA-FLIP

Shard counts:
- k4
- k6
- k8

Files:
- \`shards.<dataset>.openai.k<4|6|8>.jsonl\`: final validated shard artifacts
- \`shards.<dataset>.openai.k<4|6|8>.eligible_all_*.jsonl\`: optional matched-cohort shard artifacts
- \`atomic_units.<dataset>.openai.jsonl\`: reusable atomic-unit caches
- \`shard_ablation.<dataset>.openai.summary.json\`: run summaries
- \`cohort.<dataset>.openai.eligible_all_*.summary.json\`: optional matched-cohort summaries
- \`run_errors.<dataset>.openai.atomic.jsonl\`: rows that failed atomic extraction
- \`SHA256SUMS.txt\`: SHA-256 checksums for integrity checks
- \`LINE_COUNTS.txt\`: line counts for quick completeness checks

Short examples remain in shard files as \`ineligible_target_shards\` records.

To validate a shard file from the repo root:

\`\`\`bash
uv run --project decomposition python -m decomposition.cli validate --input <path>
\`\`\`
EOF

(
  cd "$SHARE_DIR"
  checksum_files
  wc -l *.jsonl > LINE_COUNTS.txt
)

zip -r "$ZIP_PATH" "$SHARE_DIR" >/dev/null

echo "Wrote $SHARE_DIR"
echo "Wrote $ZIP_PATH"
