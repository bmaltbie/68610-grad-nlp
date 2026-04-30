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
- `prompts.py` — ELEPHANT Appendix-B rubric prompts + multi-turn wrapper
  + verdict-extraction prompt. *(Phase 2)*
- `io.py` — `conversation_transcripts.jsonl` reader; classifies auditor
  turns as `shard` / `interrupt` / `aita_close`. *(Phase 1.2)*
- `judge.py` — `ElephantJudge`, per-turn binary scoring for dimensions
  1–3. *(Phase 2)*
- `verdict.py` — `VerdictExtractor`, NTA / YTA / UNCLEAR per-turn
  extraction (input to dimension 4). *(Phase 3)*
- `moral.py` — `MoralScorer`, joins paired AITA-NTA-OG ⨝ AITA-NTA-FLIP
  transcripts and computes per-turn moral comparisons. *(Phase 3)*
- `runner.py` — orchestrator. *(Phase 4)*
- `aggregate.py` — per-turn rates and S^d deltas. *(Phase 5)*
- `utils.py` — OpenAI async client helpers. *(Phase 2)*

## Setup

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e judging[dev]
```

## Tests

```sh
cd judging
pytest
```
