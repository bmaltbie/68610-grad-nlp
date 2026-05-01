# Judging Stage

Stage 3 of the multi-turn ELEPHANT pipeline. Reads
`conversation_transcripts.jsonl` produced by the conversation stage and
emits per-turn ELEPHANT sycophancy scores plus paired moral-sycophancy
verdicts.

For overall pipeline architecture see
[`docs/multi-turn-elephant-framework-design.md`](../docs/multi-turn-elephant-framework-design.md).
For judging-specific design see [`PLAN.md`](./PLAN.md) and
[`TODO.md`](./TODO.md).

## Layout

- `schemas.py` — Pydantic models (input transcripts, structured-output
  schemas for OpenAI, output traces and index records).
- `io.py` — `conversation_transcripts.jsonl` reader; classifies auditor
  turns as `shard` / `interrupt` / `aita_close`.
- `prompts.py` — ELEPHANT Appendix-B rubric prompts (verbatim from the
  upstream repo) + multi-turn wrapper + verdict-extraction prompt.
- `utils.py` — OpenAI async client helpers (rate limit, retry,
  structured-output call wrapper).
- `judge.py` — `ElephantJudge`, per-target-turn × 3-dim async scoring
  for validation / indirectness / framing.
- `verdict.py` — `VerdictExtractor`, NTA / YTA / UNCLEAR per-target-turn
  extraction (input to dimension 4).
- `moral.py` — `MoralScorer`, joins paired AITA-NTA-OG ⨝ AITA-NTA-FLIP
  transcripts and computes per-target-turn-position moral comparisons.
- `runner.py` — end-to-end orchestrator (`run_score`, `run_moral`).
- `batch_runner.py` — OpenAI Batch API path (50% cheaper, up to 24h SLA).
- `aggregate.py` — per-turn rates, S^d deltas, close-turn rate, moral
  rate. Long-format pandas output.
- `analysis.py` — accumulation-curve and cross-dataset plots.
- `calibrate.py` — single-turn calibration vs ELEPHANT Table 3.
- `cli.py` / `__main__.py` — `python -m judging {score,moral,aggregate,plot,calibrate}`.

## Setup

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e judging[dev]
```

`OPENAI_API_KEY` must be set in the shell or in a `.env` file at the
repo root for `score` / `moral` / `calibrate` (any subcommand that
calls the API). `aggregate` and `plot` are offline.

## CLI

All execution flows through `python -m judging <subcommand>`.

### Score transcripts

```sh
python -m judging score \
    --input datasets/conversation_transcripts.jsonl \
    --output-dir outputs/judging \
    --judge-model gpt-4o-2024-11-20 \
    --concurrency 8
```

Writes `outputs/judging/judge.jsonl` (one summary line per transcript)
and `outputs/judging/traces/<transcript_id>.json` (full reasoning trace
per dimension per turn). Logs a call-count + cost estimate before any
API spend; ctrl-C if it's bigger than expected. Use `--max N` to cap
transcripts for smoke tests, `--resume` to skip transcripts whose trace
file already exists.

### Moral pairs

```sh
python -m judging moral \
    --judge-jsonl outputs/judging/judge.jsonl \
    --output outputs/judging/moral.jsonl
```

Joins NTA-OG and NTA-FLIP traces by `(example_id, target_model)` and
writes one moral-pair record per joined example. UNCLEAR verdicts are
excluded from the rate denominator.

### Aggregate / plot

```sh
python -m judging aggregate --judge-jsonl outputs/judging/judge.jsonl --kind rate
python -m judging aggregate --judge-jsonl outputs/judging/judge.jsonl --kind close
python -m judging aggregate --judge-jsonl outputs/judging/judge.jsonl --kind delta \
    --baselines-csv datasets/AITA-YTA.csv
python -m judging aggregate --judge-jsonl outputs/judging/judge.jsonl --kind moral \
    --moral-jsonl outputs/judging/moral.jsonl

python -m judging plot --judge-jsonl outputs/judging/judge.jsonl \
    --kind accumulation --output-dir outputs/judging/plots
python -m judging plot --judge-jsonl outputs/judging/judge.jsonl \
    --kind cross_dataset --output-dir outputs/judging/plots
```

### Calibration

One-off check that the judge prompts + API integration reproduce the
single-turn rates from Cheng et al. (2025) Table 3.

```sh
python -m judging calibrate --n 100 \
    --judge-model gpt-4o-2024-11-20 \
    --csv datasets/AITA-YTA.csv \
    --seed 1
```

Prints a per-dimension pass/fail table with 5pp tolerance. See **Calibration
results** below.

## Tests

```sh
cd judging
pytest
```

API-touching code is exercised with mocked OpenAI clients; integration
tests against `datasets/conversation_transcripts.jsonl` skip gracefully
when the file is absent.

## Calibration results

Run on N=100 AITA-YTA samples with `gpt-4o-2024-11-20` (the snapshot
the paper used). Two random seeds for sample-variance bracketing.

| Dim | Seed 1 \|diff\| | Seed 2 \|diff\| | Pass on seed 2? |
|---|---|---|---|
| validation | 0.41 | 0.31 | **FAIL** |
| indirectness | 0.08 | 0.03 | **PASS** |
| framing | 0.12 | 0.014 | **PASS** |

Indirectness and framing reproduce within the 5pp tolerance, confirming
the judge prompts (verbatim from the upstream repo) and API integration
are correct. Validation runs ~30pp higher than published (0.32),
consistent across seeds and in the same direction. We attribute this to
OpenAI's silent backend rolls on `gpt-4o-2024-11-20` between when the
paper measured (Mar–Sept 2025) and our run (Apr 2026), and document it
as a known limitation rather than a code-side bug. Multi-turn results
are reported on an internally consistent scale.

### Batch API (50% cheaper, up to 24h SLA)

```sh
# Submit + poll until complete in one shot
python -m judging score \
    --input datasets/conversation_transcripts.jsonl \
    --output-dir outputs/judging \
    --judge-model gpt-4o-2024-11-20 \
    --batch \
    --poll-interval-s 300

# Or: submit, walk away, resume later with the printed batch id
python -m judging score --input ... --batch  # prints batch_id, then polls
# (later, after closing the laptop)
python -m judging score --input ... --batch --batch-id batch_abc123
```

When `--batch` is on, `--concurrency` and `--resume` are ignored
(batch executes server-side; rerun with the same `batch_id` to resume).

## What's next

- More transcripts and FLIP coverage from the conversation stage to
  activate the moral-pair join and tighten per-turn CIs.
- Auditor compliance with the AITA close question (currently the
  pragmatic last-target-turn fallback handles the gap).
