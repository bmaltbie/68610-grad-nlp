"""End-to-end orchestrator for the judging stage.

Two pipelines:
- ``run_score`` reads ``conversation_transcripts.jsonl``, runs
  ``ElephantJudge`` over every transcript, runs ``VerdictExtractor``
  on AITA-NTA-{OG,FLIP} only, writes a per-trace JSON file under
  ``traces/`` and one summary line to ``judge.jsonl``.
- ``run_moral`` reads ``judge.jsonl`` + ``traces/`` and writes
  ``moral.jsonl`` (joined OG ⨝ FLIP pairs).

Flags supported via ``judging.cli score``:
- ``--resume`` skips transcripts whose trace file already exists and
  preserves prior ``judge.jsonl`` lines.
- ``--max`` caps the number of transcripts processed (smoke testing).
- ``--batch`` is reserved for the OpenAI Batch API path; currently
  raises ``NotImplementedError`` — the async-with-retry path here is
  fast enough for the smoke / pilot phases.

The runner always logs a call-count + cost estimate before any API
spend, so you can ctrl-C if the number is bigger than expected.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from judging.io import load_transcripts
from judging.judge import DIMENSIONS, ElephantJudge
from judging.moral import MoralScorer, load_traces
from judging.schemas import (
    JudgeIndexRecord,
    TargetTurnScore,
    TargetTurnVerdict,
    Transcript,
    TranscriptScore,
    TurnSummary,
)
from judging.utils import create_async_client
from judging.verdict import VerdictExtractor

log = logging.getLogger(__name__)

RUBRIC_VERSION = "elephant-appendix-b-2025"
# Rough per-call cost. GPT-4o is ~$0.0025/1k input + $0.01/1k output
# tokens; a typical structured-output judge call is ~700 input + 100
# output ≈ $0.003. Order-of-magnitude estimate; tweak after a smoke run.
COST_PER_CALL_USD = 0.003


# ---------------------------------------------------------------------------
# Output layout helpers
# ---------------------------------------------------------------------------


def _trace_path(output_dir: Path, transcript_id: str) -> Path:
    return output_dir / "traces" / f"{transcript_id}.json"


def _judge_jsonl_path(output_dir: Path) -> Path:
    return output_dir / "judge.jsonl"


def _moral_jsonl_path(output_dir: Path) -> Path:
    return output_dir / "moral.jsonl"


def _ensure_output_dirs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "traces").mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Resume bookkeeping
# ---------------------------------------------------------------------------


def _existing_trace_ids(output_dir: Path) -> set[str]:
    return {p.stem for p in (output_dir / "traces").glob("*.json")}


def _read_index_ids(output_dir: Path) -> set[str]:
    """Return the set of transcript_ids currently in judge.jsonl."""
    path = _judge_jsonl_path(output_dir)
    if not path.exists():
        return set()
    out: set[str] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.add(json.loads(line)["transcript_id"])
    return out


# ---------------------------------------------------------------------------
# Trace + index writers
# ---------------------------------------------------------------------------


def _build_transcript_score(
    transcript: Transcript,
    *,
    judge_model: str,
    per_turn_scores: list[TargetTurnScore],
    per_turn_verdicts: Optional[list[TargetTurnVerdict]],
) -> TranscriptScore:
    return TranscriptScore(
        transcript_id=transcript.transcript_id,
        example_id=transcript.example_id,
        dataset_name=transcript.dataset_name,
        target_model=transcript.target_model,
        judge_model=judge_model,
        rubric_version=RUBRIC_VERSION,
        judged_at=datetime.now(timezone.utc),
        per_target_turn_scores=per_turn_scores,
        per_target_turn_moral_verdict=per_turn_verdicts,
    )


def _build_index_record(
    transcript: Transcript,
    score: TranscriptScore,
    trace_path: Path,
) -> JudgeIndexRecord:
    summary = [
        TurnSummary(
            turn=s.turn,
            preceded_by=s.preceded_by,
            validation=s.validation.score,
            indirectness=s.indirectness.score,
            framing=s.framing.score,
        )
        for s in score.per_target_turn_scores
    ]
    return JudgeIndexRecord(
        transcript_id=transcript.transcript_id,
        example_id=transcript.example_id,
        dataset_name=transcript.dataset_name,
        target_model=transcript.target_model,
        judge_model=score.judge_model,
        rubric_version=score.rubric_version,
        trace_path=str(trace_path),
        per_turn_summary=summary,
    )


def _write_trace(score: TranscriptScore, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(score.model_dump_json(indent=2))


def _append_index(record: JudgeIndexRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(record.model_dump_json() + "\n")


# ---------------------------------------------------------------------------
# Cost estimator
# ---------------------------------------------------------------------------


def estimate_calls(transcripts: Iterable[Transcript]) -> dict[str, int]:
    """Count API calls without spending. Used by ``--dry-run``."""
    n_judge = n_verdict = 0
    n_transcripts = 0
    for tr in transcripts:
        n_transcripts += 1
        n_target = len(tr.target_turns())
        n_judge += n_target * len(DIMENSIONS)
        if tr.dataset_name in {"AITA-NTA-OG", "AITA-NTA-FLIP"}:
            n_verdict += n_target
    return {
        "transcripts": n_transcripts,
        "judge_calls": n_judge,
        "verdict_calls": n_verdict,
        "total_calls": n_judge + n_verdict,
        "estimated_usd": round((n_judge + n_verdict) * COST_PER_CALL_USD, 2),
    }


# ---------------------------------------------------------------------------
# Score runner
# ---------------------------------------------------------------------------


async def _score_one(
    transcript: Transcript,
    judge: ElephantJudge,
    verdict: Optional[VerdictExtractor],
    *,
    output_dir: Path,
    judge_model: str,
) -> JudgeIndexRecord:
    needs_verdict = transcript.dataset_name in {"AITA-NTA-OG", "AITA-NTA-FLIP"}

    if needs_verdict and verdict is not None:
        scores, verdicts = await asyncio.gather(
            judge.score_transcript(transcript),
            verdict.extract_transcript(transcript),
        )
    else:
        scores = await judge.score_transcript(transcript)
        verdicts = None

    score = _build_transcript_score(
        transcript,
        judge_model=judge_model,
        per_turn_scores=list(scores),
        per_turn_verdicts=list(verdicts) if verdicts is not None else None,
    )
    trace_path = _trace_path(output_dir, transcript.transcript_id)
    _write_trace(score, trace_path)

    index_record = _build_index_record(transcript, score, trace_path)
    _append_index(index_record, _judge_jsonl_path(output_dir))
    return index_record


async def run_score(
    *,
    input_path: Path | str,
    output_dir: Path | str,
    judge_model: str = "gpt-4o",
    concurrency: int = 8,
    resume: bool = False,
    max_count: Optional[int] = None,
    batch: bool = False,
    batch_id: Optional[str] = None,
    poll_interval_s: int = 60,
) -> dict:
    """Score every transcript in ``input_path`` and write outputs.

    Returns a stats dict (transcripts processed, calls made).

    When ``batch=True``, dispatch to the OpenAI Batch API path
    (``judging.batch_runner.run_batch``). ``--concurrency`` and
    ``--resume`` are ignored under batch (the batch endpoint runs
    server-side; rerun with the same ``batch_id`` to resume).
    """
    if batch:
        from judging.batch_runner import run_batch

        return await run_batch(
            input_path=input_path,
            output_dir=output_dir,
            judge_model=judge_model,
            batch_id=batch_id,
            poll_interval_s=poll_interval_s,
            max_count=max_count,
        )

    output_dir = Path(output_dir)
    _ensure_output_dirs(output_dir)

    transcripts = list(load_transcripts(input_path))
    if max_count is not None:
        transcripts = transcripts[:max_count]

    skip_ids: set[str] = set()
    if resume:
        skip_ids = _existing_trace_ids(output_dir) & _read_index_ids(output_dir)
        log.info("resume: skipping %d already-scored transcripts", len(skip_ids))
    else:
        # Truncate index so we don't append duplicates on re-run.
        _judge_jsonl_path(output_dir).write_text("")

    todo = [t for t in transcripts if t.transcript_id not in skip_ids]
    if not todo:
        log.info("nothing to do (all transcripts already scored)")
        return {"transcripts": 0, "scored": 0}

    # Always log the cost estimate before spending. ctrl-C if too big.
    stats = estimate_calls(todo)
    log.info(
        "estimate: %d transcripts → %d judge + %d verdict = %d calls, ~$%.2f",
        stats["transcripts"],
        stats["judge_calls"],
        stats["verdict_calls"],
        stats["total_calls"],
        stats["estimated_usd"],
    )

    client = create_async_client()
    judge = ElephantJudge(client, model=judge_model, concurrency=concurrency)
    verdict = VerdictExtractor(client, model=judge_model, concurrency=concurrency)

    n_scored = 0
    for tr in todo:
        try:
            await _score_one(
                tr, judge, verdict, output_dir=output_dir, judge_model=judge_model
            )
        except Exception as e:
            log.exception("failed to score transcript %s: %s", tr.transcript_id, e)
            continue
        n_scored += 1
        if n_scored % 10 == 0:
            log.info("scored %d/%d", n_scored, len(todo))

    log.info("done: scored %d/%d transcripts", n_scored, len(todo))
    return {"transcripts": len(todo), "scored": n_scored}


# ---------------------------------------------------------------------------
# Moral runner (second pass)
# ---------------------------------------------------------------------------


def run_moral(
    *,
    judge_jsonl: Path | str,
    output_path: Path | str,
) -> int:
    """Read traces alongside ``judge_jsonl`` and write ``moral.jsonl``.

    Returns the number of pairs written.
    """
    judge_jsonl = Path(judge_jsonl)
    output_dir = judge_jsonl.parent
    traces_dir = output_dir / "traces"
    if not traces_dir.exists():
        raise FileNotFoundError(
            f"traces/ directory not found at {traces_dir}; run `score` first"
        )
    traces = list(load_traces(traces_dir))
    n = MoralScorer.write_moral_jsonl(traces, output_path)
    return n


__all__ = ["run_score", "run_moral", "estimate_calls"]
