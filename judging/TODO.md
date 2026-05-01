# TODO — Judging Stage

Datasets in scope: **AITA-YTA**, **AITA-NTA-OG**, **AITA-NTA-FLIP**.
Goal: faithful ELEPHANT reproduction (4 metrics) extended single-turn → multi-turn.

**5 commits total.** Each step ends at a natural commit boundary. Tests use the real
`datasets/conversation_transcripts.jsonl` — no mock fixtures. All execution is
driven through `python -m judging` subcommands.

---

## Step 1 — Foundation: schemas + I/O + CLI skeleton

- [x] `judging/schemas.py` + `judging/tests/test_schemas.py` (22 tests, done)
- [ ] `judging/io.py`
  - `load_transcripts(path) -> Iterator[Transcript]` reads `conversation_transcripts.jsonl`
  - `classify_auditor_turn(text, seed_identifier, seeds)` → `shard | interrupt | aita_close`
    using the live `conversation/seed.json` catalogue
  - Drop `succeeded: False` at ingest; log dropped count + dominant `reason` values
  - Validate alternating roles + sequential turn numbers
- [ ] `judging/cli.py` — argparse with subcommands: `score`, `moral`, `aggregate`,
  `plot`, `calibrate`. Stubs at this step (raise `NotImplementedError`); we're
  committing the wiring
- [ ] `judging/__main__.py` — `python -m judging` → `cli.main()`
- [ ] `judging/tests/test_io.py` — reads the real `datasets/conversation_transcripts.jsonl`
- [ ] `judging/tests/test_cli.py` — `-h` works, missing-arg errors are clean

**Commit:** `judging: add Transcript I/O, auditor-turn classifier, and CLI skeleton`

---

## Step 2 — Judge + Verdict Extractor (dims 1–4 prep)

- [ ] `judging/prompts.py` — verbatim ELEPHANT Appendix-B prompts
  (`VALIDATION_PROMPT`, `INDIRECTNESS_PROMPT`, `FRAMING_PROMPT`),
  `SYSTEM_PROMPT_JUDGE`, `VERDICT_EXTRACTION_PROMPT`,
  `build_multiturn_judge_prompt(rubric, transcript, target_turn)`
- [ ] `judging/utils.py` — `create_async_client()`, `call_structured()` using
  `client.beta.chat.completions.parse()`, retry with backoff, `RateLimiter`
- [ ] `judging/judge.py` — `ElephantJudge`: per-target-turn × 3-dim async scoring;
  concurrent dimension calls via `asyncio.gather`
- [ ] `judging/verdict.py` — `VerdictExtractor`: per-target-turn `NTA/YTA/UNCLEAR`
- [ ] Tests in `test_prompts.py`, `test_judge.py`, `test_verdict.py` with mocked
  OpenAI client

**Commit:** `judging: ElephantJudge (dims 1–3) + VerdictExtractor (dim 4 prep)`

---

## Step 3 — Moral scorer + runner (orchestration)

- [ ] `judging/moral.py` — `MoralScorer`: `join_pairs()` groups OG/FLIP by
  `(example_id, target_model)`; `score_pair()` aligns by ordinal target-turn
  position, computes `MoralPair`; UNCLEAR pairs excluded from rate
- [ ] `judging/runner.py` — read input JSONL → run `ElephantJudge` (always) +
  `VerdictExtractor` (NTA-OG / NTA-FLIP) → write `outputs/judging/judge.jsonl`
  index + per-trace JSON; second pass joins traces and writes `moral.jsonl`
- [ ] Wire `judging.cli score` and `judging.cli moral` to the runner; flags:
  `--input`, `--output-dir`, `--resume`, `--dry-run`, `--batch`, `--judge-model`,
  `--concurrency`
- [ ] `judging/tests/test_moral.py` — synthesised verdicts in-test (no fixture
  files); rate computation skips UNCLEAR; unmatched pairs warn
- [ ] `judging/tests/test_runner.py` — patch judge + verdict, run on the first
  1–2 records of `datasets/conversation_transcripts.jsonl`, verify output shapes

**Commit:** `judging: moral scorer + end-to-end runner with resume/dry-run/batch`

---

## Step 4 — Aggregation + analysis

Headline comparison: **per-turn rate trajectory** (multi-turn) vs **single-turn baseline** at the close turn — the apples-to-apples vs original ELEPHANT.

- [ ] `judging/aggregate.py` — load index + traces into DataFrames;
  `compute_rate(df, by)`, `compute_delta(df, baselines_csv)` (AITA-YTA only),
  `compute_moral_rate(moral_jsonl)`,
  `compute_close_turn_rate(df)` *(rate restricted to target turns whose
  preceding auditor turn is `aita_close` — apples-to-apples vs single-turn)*
- [ ] `judging/analysis.py` — `plot_accumulation_curves()` (per-turn rate over
  `t = 1..T`), `plot_cross_dataset()` (compare YTA / NTA-OG / NTA-FLIP
  trajectories). Each takes a `judge.jsonl` path and an output dir
- [ ] Wire `judging.cli aggregate --kind {rate,delta,moral,close}` and
  `judging.cli plot --kind {accumulation,cross_dataset}`
- [ ] Tests for aggregate functions on small DataFrames; smoke-test plots write
  a non-empty PNG

**Commit:** `judging: aggregate + plot subcommands (rates, deltas, accumulation curves)`

---

## Step 5 — Calibration

- [ ] Add `--single-turn` mode to `judging.judge`: bypass multi-turn formatter,
  use original Appendix-B prompt verbatim (`Question: {prompt}\nAdvice: {response}`,
  `max_tokens=2`, no reasoning)
- [ ] Wire `judging.cli calibrate --n 30 --dataset AITA-YTA`: samples N records
  from `datasets/AITA-YTA.csv`, runs single-turn judge, prints per-dimension
  mean rate vs published ELEPHANT Table 3, pass/fail diff per dim (~5 pp tolerance)
- [ ] Unit test for the `--single-turn` prompt mode using a mocked client; the
  real run is a one-off ~$0.50 spend (deferred until step 7)

**Commit:** `judging: calibration subcommand vs ELEPHANT Table 3 (--single-turn mode)`

---

## After step 5 — tiered rollout (real API spend, no new code)

- [ ] **A. Smoke test** (~$1): `python -m judging score --input ... --max 10` on
  10 transcripts. Eyeball 5 reasoning traces; assert 100% structured-output
  parse success.
- [ ] **B. Calibration** (~$0.50): `python -m judging calibrate --n 30`. Stop if
  it doesn't pass ~5 pp tolerance.
- [ ] **C. Pilot** (~$30–60, ~$15–30 batch): ~500 transcripts spanning 3 datasets
  × 2–3 seeds. Generate plots. Decision point.
- [ ] **D. Full study** (only if needed, ~$100–250 batch): scale up cells
  flagged in C.

---

## Open questions

- [ ] Holistic 0–10 rubric score — defer; not in original ELEPHANT
- [ ] Which target models to evaluate (auditor and target are currently both
  `openai/gpt-5.4-nano` — flag as confound in the paper)
- [ ] Backfill human baselines for NTA-OG / NTA-FLIP — would need top-comment
  text; punt unless reviewers ask
