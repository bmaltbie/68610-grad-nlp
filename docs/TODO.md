# TODO — Evaluation Layer

## Phase 1: Foundation
- [ ] Create `pyproject.toml` with dependencies (openai, pydantic, pandas, matplotlib, seaborn, pytest, python-dotenv, tqdm)
- [ ] Create `.env.example` and `.gitignore`
- [ ] Implement `src/elephant_eval/schemas.py` — Pydantic models for Transcript, TurnMessage, TurnScore, TranscriptScore
- [ ] Write `scripts/generate_mock_data.py` — generate synthetic sharded transcripts for development
- [ ] Generate `tests/fixtures/mock_transcripts.jsonl`
- [ ] Write `tests/test_schemas.py`

## Phase 2: Core Judge
- [ ] Implement `src/elephant_eval/prompts.py` — port 3 ELEPHANT judge prompts verbatim from Appendix B, create multi-turn wrapper
- [ ] Implement `src/elephant_eval/utils.py` — OpenAI async client, retry with backoff, rate limiter, binary response parsing
- [ ] Implement `src/elephant_eval/judge.py` — `ElephantJudge` class with per-turn binary scoring
- [ ] Write `tests/test_prompts.py` — verify prompt construction
- [ ] Write `tests/test_judge.py` — mock API responses, test binary parsing edge cases

## Phase 3: Orchestration
- [ ] Implement `src/elephant_eval/runner.py` — read transcript.jsonl, run judge, write scores.jsonl; support `--resume` and `--dry-run`
- [ ] Implement `src/elephant_eval/moral.py` — moral sycophancy scorer (NTA/YTA pair comparison, per-turn)
- [ ] Write `scripts/run_eval.py` — CLI wrapper

## Phase 4: Analysis
- [ ] Implement `src/elephant_eval/aggregate.py` — compute S^d_{m,P} deltas per-turn and aggregate
- [ ] Write `analysis/accumulation_curves.py` — turn-by-turn sycophancy rate plots (does sycophancy decrease as shards reveal more?)
- [ ] Write `analysis/cross_model.py` — compare which Target models course-correct fastest
- [ ] Create `analysis/notebook.ipynb` — interactive exploration

## Phase 5: Calibration
- [ ] Write `scripts/run_calibration.py` — full-post single-turn scores vs ELEPHANT baselines (target >90% agreement)
- [ ] Cost estimation: `--dry-run` flag counts transcripts and estimates API cost

## Open Questions
- [ ] How to shard posts (by paragraph? by sentence? by narrative arc?) — upstream concern
- [ ] Number of shards per post (2? 3? variable?)
- [ ] Whether to add a holistic rubric score (0-10) on top of per-turn binary scores later
- [ ] Which Target Agent models to evaluate
- [ ] How to extract NTA/YTA verdict from Target Agent's free-form multi-turn responses (regex? LLM classifier? constrained output?)
