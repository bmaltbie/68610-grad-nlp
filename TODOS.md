# TODOS

## Testing

### Golden End-to-End Contract Fixtures

**What:** Add a golden-fixture cross-stage contract test suite for `shards.jsonl -> transcript.jsonl -> judge.jsonl`.

**Why:** Schema-valid files can still be incompatible in practice, and this catches interface drift before three modules all finish independently.

**Context:** The current design requires decomposition-stage validation and defers downstream contracts to the conversation and judging owners. This is the next hardening step once initial versions of all three artifacts exist.

**Effort:** M
**Priority:** P2
**Depends on:** Initial versions of `shards.jsonl`, `transcript.jsonl`, and `judge.jsonl`

### Judge-Stage Drift Eval Suite

**What:** Add a locked judge-stage eval suite to detect scoring drift in the judging prompt or model.

**Why:** Judge outputs can remain schema-valid while rubric behavior silently changes, which is a science problem, not just a tooling problem.

**Context:** The current design requires locked evals for segmentation but not for judging. This is the clean follow-up once transcript and judge contracts are stable.

**Effort:** M
**Priority:** P2
**Depends on:** Stable transcript contract, stable judge output contract, and a small representative transcript set

### Decomposer Quality Eval Suite

**What:** Add a locked decomposer quality eval suite comparing OpenAI and Anthropic decomposition outputs on exact-span preservation, shard naturalness, and fixed-four coverage.

**Why:** Fake-client tests prove provider wiring and artifact validation, but they do not prove the default OpenAI decomposer produces research-quality shard boundaries.

**Context:** The OpenAI-default migration should include a small live smoke run first. Once real OpenAI outputs exist, this eval suite should lock a representative corpus across AITA-NTA-OG, AITA-NTA-FLIP, AITA-YTA, OEQ, and SS so future prompt or model changes cannot silently degrade decomposition quality.

**Effort:** M
**Priority:** P2
**Depends on:** OpenAI provider path merged, representative rows from each supported dataset, and an agreed human or judge rubric
