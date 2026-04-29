# Decomposition Module

This directory contains decomposition-specific notes and research material.

For the project-level architecture, sharding contract, and cross-module handoff rules, start with the framework design:

- [Multi-Turn ELEPHANT Sharding Framework and Handoff Contract](../docs/multi-turn-elephant-framework-design.md)

## Module Ownership

The decomposition module owns the first stage of the benchmark pipeline:

```text
single AITA post
    |
    v
decomposition module
    |
    v
shards.jsonl
```

The module is responsible for:

- ingesting raw AITA post text;
- extracting traceable atomic units;
- validating raw-text source spans;
- merging units into neutral chronological shards;
- writing `shards.jsonl`.

The module does not own:

- conversational rephrasing;
- user simulator prompting;
- target model prompting;
- judge prompting or scoring.

## Local References

- [Deep research report](./deep-research-report.md)

## Implementation

The v1 implementation is contract-first: every emitted `shards.jsonl` record is
validated against the project-level schema before downstream modules consume it.
Run CLI commands from the repository root so imports resolve against the checked-out
package while `uv` uses the decomposition-owned project config.

Show the CLI command map:

```bash
uv run --project decomposition python -m decomposition.cli --help
```

Run the deterministic baseline:

```bash
uv run --project decomposition python -m decomposition.cli deterministic \
  --dataset datasets/AITA-NTA-OG.csv \
  --dataset-name AITA-NTA-OG \
  --source-field original_post \
  --run-id mt-elephant-pilot-v1 \
  --out decomposition/artifacts/shards.deterministic.jsonl \
  --errors decomposition/artifacts/run_errors.deterministic.jsonl
```

Run native LLM segmentation. OpenAI is the default provider:

```bash
export OPENAI_API_KEY="..."

uv run --project decomposition python -m decomposition.cli llm \
  --dataset datasets/AITA-YTA.csv \
  --dataset-name AITA-YTA \
  --source-field prompt \
  --run-id mt-elephant-pilot-v1 \
  --provider openai \
  --model gpt-5.4-mini \
  --limit 20 \
  --resume \
  --concurrency 2 \
  --llm-retries 1 \
  --provider-max-retries 2 \
  --raw-responses-mode append \
  --progress auto \
  --out decomposition/artifacts/shards.AITA-YTA.openai.jsonl \
  --errors decomposition/artifacts/run_errors.AITA-YTA.openai.jsonl \
  --raw-responses decomposition/artifacts/seg_v1_openai_responses.AITA-YTA.jsonl
```

The `llm` command is the normal human path for LLM-assisted segmentation. It
uses OpenAI Responses with structured outputs by default, then still runs local
source-span alignment before writing `shards.jsonl`; model offsets are never
trusted. `gpt-5.4-mini` is the cost-conscious default for remaining datasets.
Use `--model gpt-5.5` for a higher-quality smoke run or final strict pass.

Progress defaults to `auto`, which shows a `tqdm` bar in an interactive
terminal and line-by-line progress in redirected logs. Use `--resume` for long
runs: validated rows in `--out` with matching request fingerprints are reused,
and `--out.tmp` is also loaded by default so interrupted runs can keep paid-for
validated rows. Existing rows without fingerprints are not trusted as cache
hits. Failed or stale rows are queried again. `--llm-retries` controls extra
attempts for model-output problems such as malformed JSON or source alignment
failures; provider transport retries are delegated to the provider SDK through
`--provider-max-retries`, `--openai-max-retries`, or `--anthropic-max-retries`.

For live runs, start conservatively with `--concurrency 2`. Final
`shards.jsonl` output still preserves dataset order even when provider calls
finish out of order. Raw response sidecars default to append mode when
`--resume` is enabled, or pass `--raw-responses-mode overwrite` to intentionally
start a fresh sidecar.

Run shard-count ablations from one atomic cache:

```bash
uv run --project decomposition python -m decomposition.cli shard-ablation \
  --dataset datasets/AITA-YTA.csv \
  --dataset-name AITA-YTA \
  --source-field prompt \
  --run-id aita-yta-openai-ablation-v1 \
  --provider openai \
  --model gpt-5.4-mini \
  --target-turns 4,6,8 \
  --out-dir decomposition/artifacts \
  --seed-shards decomposition/artifacts/shards.AITA-YTA.openai.smoke.jsonl \
  --resume \
  --resume-include-temp \
  --concurrency 2 \
  --llm-retries 1 \
  --provider-max-retries 2 \
  --progress auto
```

`shard-ablation` asks OpenAI only for atomic units, caches those rows in
`atomic_units.<dataset>.openai.jsonl`, then deterministically writes
`shards.<dataset>.openai.k4.jsonl`, `k6`, and `k8` with `natural_dp_v1`.
Short rows remain in every shard file as `ineligible_target_shards` records with
their atomic units preserved. The expensive recovery order is: final atomic
cache, `atomic_units...jsonl.tmp`, valid raw response sidecar rows, explicit
`--seed-shards` files, then new OpenAI calls for true misses. If a run crashes
during calls, rerun the same command; if it crashes during shard generation,
rerun and it should make zero OpenAI calls once the atomic cache is complete.

OpenAI Batch is available for the same ablation path:

```bash
uv run --project decomposition python -m decomposition.cli shard-ablation-batch submit \
  --dataset datasets/AITA-YTA.csv \
  --dataset-name AITA-YTA \
  --source-field prompt \
  --run-id aita-yta-openai-ablation-v1 \
  --target-turns 4,6,8 \
  --out-dir decomposition/artifacts \
  --seed-shards decomposition/artifacts/shards.AITA-YTA.openai.smoke.jsonl \
  --batch-state decomposition/artifacts/shard_ablation_openai_batch_state.json

uv run --project decomposition python -m decomposition.cli shard-ablation-batch collect \
  --batch-state decomposition/artifacts/shard_ablation_openai_batch_state.json
```

Batch submit writes the prepared input JSONL, uploaded file id, and batch id into
the state file incrementally. Batch collect is rerunnable and handles unordered
OpenAI output rows by `custom_id` before writing the same atomic cache and shard
artifacts as realtime mode.

Live smoke run:

```bash
uv run --project decomposition python -m decomposition.cli shard-ablation \
  --dataset datasets/AITA-YTA.csv \
  --dataset-name AITA-YTA \
  --source-field prompt \
  --run-id aita-yta-openai-ablation-smoke-v1 \
  --limit 20 \
  --target-turns 4,6,8 \
  --out-dir decomposition/artifacts \
  --seed-shards decomposition/artifacts/shards.AITA-YTA.openai.smoke.jsonl \
  --resume \
  --concurrency 2
```

Anthropic remains available as a compatibility provider:

```bash
export ANTHROPIC_API_KEY="..."

uv run --project decomposition python -m decomposition.cli anthropic \
  --dataset datasets/AITA-NTA-OG.csv \
  --dataset-name AITA-NTA-OG \
  --source-field original_post \
  --run-id mt-elephant-pilot-v1 \
  --model claude-sonnet-4-5 \
  --resume \
  --out decomposition/artifacts/shards.AITA-NTA-OG.anthropic.jsonl \
  --errors decomposition/artifacts/run_errors.AITA-NTA-OG.anthropic.jsonl \
  --raw-responses decomposition/artifacts/seg_v1_anthropic_responses.AITA-NTA-OG.jsonl
```

Run several datasets from a JSONL manifest:

```jsonl
{"dataset_name":"AITA-YTA","dataset":"datasets/AITA-YTA.csv","out":"decomposition/artifacts/shards.AITA-YTA.openai.jsonl","errors":"decomposition/artifacts/run_errors.AITA-YTA.openai.jsonl","raw_responses":"decomposition/artifacts/seg_v1_openai_responses.AITA-YTA.jsonl"}
{"dataset_name":"AITA-NTA-FLIP","dataset":"datasets/AITA-NTA-FLIP.csv","out":"decomposition/artifacts/shards.AITA-NTA-FLIP.openai.jsonl","errors":"decomposition/artifacts/run_errors.AITA-NTA-FLIP.openai.jsonl","raw_responses":"decomposition/artifacts/seg_v1_openai_responses.AITA-NTA-FLIP.jsonl"}
```

```bash
uv run --project decomposition python -m decomposition.cli llm-bulk \
  --manifest decomposition/artifacts/openai_manifest.jsonl \
  --provider openai \
  --run-id mt-elephant-pilot-v1 \
  --resume \
  --concurrency 2
```

Manifest rows may omit `source_field` for known datasets. Defaults are:
`AITA-NTA-OG=original_post`, `AITA-NTA-FLIP=flipped_story`,
`AITA-YTA=prompt`, `OEQ=prompt`, and `SS=sentence`. The bulk runner also keeps
a manifest-wide content cache, so byte-identical source text with matching
prompt/schema/provider/model/max-token/temperature settings can be reused across
datasets after local revalidation.

Use OpenAI Batch for asynchronous full-dataset processing:

```bash
uv run --project decomposition python -m decomposition.cli openai-batch submit \
  --manifest decomposition/artifacts/openai_manifest.jsonl \
  --run-id mt-elephant-pilot-v1 \
  --batch-state decomposition/artifacts/openai_batch_state.json

uv run --project decomposition python -m decomposition.cli openai-batch collect \
  --batch-state decomposition/artifacts/openai_batch_state.json
```

OpenAI batch submit streams a JSONL input file to disk, uploads it, and sends
only cache misses to `/v1/responses`. Batch collect can be rerun safely: it
loads existing final and temp shard artifacts, validates every successful result
through the same local alignment path, and writes deterministic dataset-ordered
outputs. Anthropic batch remains available through `anthropic-batch`.

Generate provider-neutral LLM requests for replay, batch runs, or debugging:

```bash
uv run --project decomposition python -m decomposition.cli generate-requests \
  --dataset datasets/AITA-NTA-OG.csv \
  --dataset-name AITA-NTA-OG \
  --source-field original_post \
  --run-id mt-elephant-pilot-v1 \
  --out decomposition/artifacts/seg_v1_requests.jsonl
```

Ingest provider-neutral LLM responses:

```bash
uv run --project decomposition python -m decomposition.cli ingest-responses \
  --requests decomposition/artifacts/seg_v1_requests.jsonl \
  --responses decomposition/artifacts/seg_v1_responses.jsonl \
  --out decomposition/artifacts/shards.llm.jsonl \
  --errors decomposition/artifacts/run_errors.llm.jsonl
```

Validate any shard artifact:

```bash
uv run --project decomposition python -m decomposition.cli validate \
  --input decomposition/artifacts/shards.llm.jsonl
```

Run tests:

```bash
cd decomposition
uv run pytest
```

## Common Errors

`source field 'missing' not found`

The CSV header does not contain the value passed to `--source-field`. Re-run with
one of the printed field names, for example `--source-field original_post`.

`unit N text not found after offset M`

An LLM response added, removed, reordered, or reworded source text. Regenerate that
response so every `atomic_units[].text` value is an exact verbatim span from the
raw post. Local code computes offsets after ingest; model offsets are ignored.

`ok records must contain exactly target_turns shards`

The response or deterministic planner did not satisfy the requested target count.
For legacy LLM-assisted runs, make sure the response has exactly four shard
objects. For ablations, inspect `target_turns`, the atomic-unit count, and the
summary JSON.

## Developer Notes

Keep decomposition-specific project files under this directory when possible. The
root workflow file remains at `.github/workflows/tests.yml` because GitHub Actions
requires workflow files there, but it is scoped to `decomposition/**`.
