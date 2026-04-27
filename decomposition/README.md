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

Run native Anthropic segmentation:

```bash
export ANTHROPIC_API_KEY="..."

uv run --project decomposition python -m decomposition.cli anthropic \
  --dataset datasets/AITA-NTA-OG.csv \
  --dataset-name AITA-NTA-OG \
  --source-field original_post \
  --run-id mt-elephant-pilot-v1 \
  --model claude-sonnet-4-5 \
  --limit 20 \
  --out decomposition/artifacts/shards.anthropic.jsonl \
  --errors decomposition/artifacts/run_errors.anthropic.jsonl \
  --raw-responses decomposition/artifacts/seg_v1_anthropic_responses.jsonl
```

The Anthropic command is the normal human path for LLM-assisted segmentation.
It still uses local source-span alignment before writing `shards.jsonl`; model
offsets are never trusted.

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

`ok records must contain exactly 4 shards`

The response did not satisfy the fixed-four primary condition. For LLM-assisted
runs, make sure the response has exactly four shard objects and that those shards
partition the atomic unit ids in source order.

## Developer Notes

Keep decomposition-specific project files under this directory when possible. The
root workflow file remains at `.github/workflows/tests.yml` because GitHub Actions
requires workflow files there, but it is scoped to `decomposition/**`.
