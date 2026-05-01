"""OpenAI Batch API path for the judging stage.

Same outputs as the live async runner — same trace JSON files and the
same ``judge.jsonl`` index — but routed through OpenAI's offline batch
endpoint (50% cheaper, up to 24h SLA).

Flow:
1. Build one batch request per (transcript, target_turn, dimension) plus
   one per (transcript, target_turn) for verdict extraction on
   AITA-NTA-{OG,FLIP}.
2. Upload as a JSONL file with ``purpose='batch'``.
3. Create the batch on ``/v1/chat/completions``.
4. Poll until status is ``completed`` (or terminal-failed).
5. Download the output file, parse responses by ``custom_id``, assemble
   ``TranscriptScore`` records, and write traces + index identical to
   the async runner.

Submission is non-blocking from the caller's view (the batch executes
on OpenAI's side); the runner here polls until completion. Pass
``--batch-id`` to resume an already-submitted batch (skip submission,
just download + assemble).
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from openai import AsyncOpenAI
from pydantic import BaseModel

from judging.io import load_transcripts
from judging.judge import DIMENSIONS
from judging.prompts import build_judge_messages, build_verdict_messages
from judging.schemas import (
    JudgeIndexRecord,
    JudgeOutput,
    TargetTurnScore,
    TargetTurnVerdict,
    Transcript,
    TranscriptScore,
    TurnDimensionScore,
    TurnSummary,
    VerdictOutput,
)
from judging.utils import _with_retry, create_async_client

log = logging.getLogger(__name__)

RUBRIC_VERSION = "elephant-appendix-b-2025"
DEFAULT_POLL_INTERVAL_S = 60
TERMINAL_OK = {"completed"}
TERMINAL_BAD = {"failed", "expired", "cancelled", "canceled"}


# ---------------------------------------------------------------------------
# Custom ID encoding (turns + dimensions ↔ batch line identity)
# ---------------------------------------------------------------------------


def _judge_custom_id(transcript_id: str, turn: int, dim: str) -> str:
    return f"judge|{transcript_id}|{turn}|{dim}"


def _verdict_custom_id(transcript_id: str, turn: int) -> str:
    return f"verdict|{transcript_id}|{turn}"


def _parse_custom_id(custom_id: str) -> tuple[str, str, int, Optional[str]]:
    """Returns (kind, transcript_id, turn, dim_or_none)."""
    parts = custom_id.split("|")
    kind = parts[0]
    if kind == "judge":
        if len(parts) != 4:
            raise ValueError(f"malformed judge custom_id: {custom_id!r}")
        return kind, parts[1], int(parts[2]), parts[3]
    if kind == "verdict":
        if len(parts) != 3:
            raise ValueError(f"malformed verdict custom_id: {custom_id!r}")
        return kind, parts[1], int(parts[2]), None
    raise ValueError(f"unknown custom_id kind: {custom_id!r}")


# ---------------------------------------------------------------------------
# Pydantic model → OpenAI strict JSON schema response_format block.
# ---------------------------------------------------------------------------


def _strict_response_format(model: type[BaseModel]) -> dict:
    """Wrap a Pydantic model's JSON schema for OpenAI's strict mode."""
    schema = model.model_json_schema()
    # Strict mode requires additionalProperties=false on every object
    # and all properties listed under "required". Pydantic gives us
    # required fields; we just need to disallow extras.
    _disallow_additional_props(schema)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model.__name__,
            "strict": True,
            "schema": schema,
        },
    }


def _disallow_additional_props(node) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object":
            node.setdefault("additionalProperties", False)
        for v in node.values():
            _disallow_additional_props(v)
    elif isinstance(node, list):
        for item in node:
            _disallow_additional_props(item)


# ---------------------------------------------------------------------------
# Request builders
# ---------------------------------------------------------------------------


def _judge_request(
    *, transcript: Transcript, turn: int, dim: str, model: str
) -> dict:
    messages = build_judge_messages(
        dim, transcript=transcript, target_turn=turn  # type: ignore[arg-type]
    )
    return {
        "custom_id": _judge_custom_id(transcript.transcript_id, turn, dim),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": messages,
            "response_format": _strict_response_format(JudgeOutput),
            "temperature": 0,
            "seed": 42,
        },
    }


def _verdict_request(
    *, transcript: Transcript, turn: int, model: str
) -> dict:
    messages = build_verdict_messages(transcript, turn)
    return {
        "custom_id": _verdict_custom_id(transcript.transcript_id, turn),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": messages,
            "response_format": _strict_response_format(VerdictOutput),
            "temperature": 0,
            "seed": 42,
        },
    }


def build_batch_requests(
    transcripts: Iterable[Transcript], judge_model: str
) -> list[dict]:
    """All judge + verdict requests for the input transcripts."""
    requests: list[dict] = []
    for t in transcripts:
        target_turns = [tt.turn for tt in t.target_turns()]
        for turn in target_turns:
            for dim in DIMENSIONS:
                requests.append(_judge_request(transcript=t, turn=turn, dim=dim, model=judge_model))
            if t.dataset_name in {"AITA-NTA-OG", "AITA-NTA-FLIP"}:
                requests.append(_verdict_request(transcript=t, turn=turn, model=judge_model))
    return requests


# ---------------------------------------------------------------------------
# Submit / poll / download
# ---------------------------------------------------------------------------


@dataclass
class BatchSubmission:
    batch_id: str
    input_file_id: str
    n_requests: int
    submitted_at: datetime


async def submit_batch(
    client: AsyncOpenAI, requests: list[dict], *, completion_window: str = "24h"
) -> BatchSubmission:
    """Upload requests + create batch. Returns BatchSubmission."""
    payload = "\n".join(json.dumps(r) for r in requests).encode()

    async def _upload():
        return await client.files.create(
            file=("batch_input.jsonl", payload),
            purpose="batch",
        )

    file_obj = await _with_retry(_upload)

    async def _create():
        return await client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window,
        )

    batch = await _with_retry(_create)
    log.info(
        "submitted batch %s with %d requests (input_file=%s)",
        batch.id,
        len(requests),
        file_obj.id,
    )
    return BatchSubmission(
        batch_id=batch.id,
        input_file_id=file_obj.id,
        n_requests=len(requests),
        submitted_at=datetime.now(timezone.utc),
    )


async def wait_for_batch(
    client: AsyncOpenAI,
    batch_id: str,
    *,
    poll_interval_s: int = DEFAULT_POLL_INTERVAL_S,
) -> str:
    """Poll a batch until terminal status. Returns output_file_id on success."""

    while True:
        async def _retrieve():
            return await client.batches.retrieve(batch_id)

        batch = await _with_retry(_retrieve)
        status = batch.status
        counts = getattr(batch, "request_counts", None)
        if counts is not None:
            log.info(
                "batch %s status=%s (completed=%d/%d, failed=%d)",
                batch_id,
                status,
                getattr(counts, "completed", 0),
                getattr(counts, "total", 0),
                getattr(counts, "failed", 0),
            )
        else:
            log.info("batch %s status=%s", batch_id, status)

        if status in TERMINAL_OK:
            if not batch.output_file_id:
                raise RuntimeError(
                    f"batch {batch_id} completed but has no output_file_id"
                )
            return batch.output_file_id
        if status in TERMINAL_BAD:
            errors = getattr(batch, "errors", None)
            raise RuntimeError(
                f"batch {batch_id} ended with status={status}; errors={errors}"
            )
        await asyncio.sleep(poll_interval_s)


async def download_results(
    client: AsyncOpenAI, output_file_id: str
) -> dict[str, dict]:
    """Returns ``{custom_id: response_body_dict}`` for completed requests."""

    async def _content():
        return await client.files.content(output_file_id)

    resp = await _with_retry(_content)
    text = resp.text if hasattr(resp, "text") else resp.read().decode()
    out: dict[str, dict] = {}
    n_errors = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        cid = rec["custom_id"]
        if rec.get("error"):
            log.warning("batch line %s errored: %s", cid, rec["error"])
            n_errors += 1
            continue
        body = rec["response"]["body"]
        out[cid] = body
    if n_errors:
        log.warning("download_results: %d/%d errored lines skipped", n_errors, n_errors + len(out))
    return out


# ---------------------------------------------------------------------------
# Response parsing + output assembly
# ---------------------------------------------------------------------------


def _parsed_judge_output(body: dict) -> JudgeOutput:
    content = body["choices"][0]["message"]["content"]
    return JudgeOutput.model_validate_json(content)


def _parsed_verdict_output(body: dict) -> VerdictOutput:
    content = body["choices"][0]["message"]["content"]
    return VerdictOutput.model_validate_json(content)


def assemble_outputs(
    transcripts: list[Transcript],
    results: dict[str, dict],
    *,
    judge_model: str,
    output_dir: Path,
) -> int:
    """Match results to transcripts; write traces + judge.jsonl. Returns count."""
    output_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = output_dir / "traces"
    traces_dir.mkdir(exist_ok=True)
    index_path = output_dir / "judge.jsonl"
    index_path.write_text("")  # truncate (full re-write per batch)

    # Bucket results: {transcript_id: {(kind, turn, dim_or_None): body}}
    by_transcript: dict[str, dict[tuple[str, int, Optional[str]], dict]] = {}
    for cid, body in results.items():
        kind, tid, turn, dim = _parse_custom_id(cid)
        by_transcript.setdefault(tid, {})[(kind, turn, dim)] = body

    n_written = 0
    for t in transcripts:
        bucket = by_transcript.get(t.transcript_id)
        if not bucket:
            log.warning("no batch results for transcript %s; skipping", t.transcript_id)
            continue

        per_turn_scores: list[TargetTurnScore] = []
        per_turn_verdicts: list[TargetTurnVerdict] = []
        for tt in t.target_turns():
            try:
                dim_outputs = {
                    dim: _parsed_judge_output(bucket[("judge", tt.turn, dim)])
                    for dim in DIMENSIONS
                }
            except KeyError as e:
                log.warning(
                    "transcript %s turn %d missing judge result for %s",
                    t.transcript_id, tt.turn, e,
                )
                continue
            per_turn_scores.append(
                TargetTurnScore(
                    turn=tt.turn,
                    preceded_by=t.preceding_auditor_turn_type(tt.turn),
                    validation=TurnDimensionScore.from_judge_output(dim_outputs["validation"]),
                    indirectness=TurnDimensionScore.from_judge_output(dim_outputs["indirectness"]),
                    framing=TurnDimensionScore.from_judge_output(dim_outputs["framing"]),
                )
            )
            verdict_body = bucket.get(("verdict", tt.turn, None))
            if verdict_body is not None:
                vo = _parsed_verdict_output(verdict_body)
                per_turn_verdicts.append(
                    TargetTurnVerdict.from_verdict_output(
                        turn=tt.turn,
                        preceded_by=t.preceding_auditor_turn_type(tt.turn),
                        vo=vo,
                    )
                )

        if not per_turn_scores:
            log.warning("transcript %s has no scored turns; skipping", t.transcript_id)
            continue

        score = TranscriptScore(
            transcript_id=t.transcript_id,
            example_id=t.example_id,
            dataset_name=t.dataset_name,
            target_model=t.target_model,
            judge_model=judge_model,
            rubric_version=RUBRIC_VERSION,
            judged_at=datetime.now(timezone.utc),
            per_target_turn_scores=per_turn_scores,
            per_target_turn_moral_verdict=per_turn_verdicts or None,
        )
        trace_path = traces_dir / f"{t.transcript_id}.json"
        trace_path.write_text(score.model_dump_json(indent=2))

        index_record = JudgeIndexRecord(
            transcript_id=t.transcript_id,
            example_id=t.example_id,
            dataset_name=t.dataset_name,
            target_model=t.target_model,
            judge_model=judge_model,
            rubric_version=RUBRIC_VERSION,
            trace_path=str(trace_path),
            per_turn_summary=[
                TurnSummary(
                    turn=s.turn,
                    preceded_by=s.preceded_by,
                    validation=s.validation.score,
                    indirectness=s.indirectness.score,
                    framing=s.framing.score,
                )
                for s in per_turn_scores
            ],
        )
        with open(index_path, "a") as f:
            f.write(index_record.model_dump_json() + "\n")
        n_written += 1
    return n_written


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------


async def run_batch(
    *,
    input_path: Path | str,
    output_dir: Path | str,
    judge_model: str = "gpt-4o",
    batch_id: Optional[str] = None,
    poll_interval_s: int = DEFAULT_POLL_INTERVAL_S,
    max_count: Optional[int] = None,
    completion_window: str = "24h",
    client: Optional[AsyncOpenAI] = None,
) -> dict:
    """End-to-end batch run.

    If ``batch_id`` is None: build requests, submit, poll, download, assemble.
    If ``batch_id`` is given: skip submission, poll the existing batch,
    download results, assemble outputs.
    """
    output_dir = Path(output_dir)
    transcripts = list(load_transcripts(input_path))
    if max_count is not None:
        transcripts = transcripts[:max_count]
    if not transcripts:
        return {"transcripts": 0, "scored": 0}

    if client is None:
        client = create_async_client()

    if batch_id is None:
        requests = build_batch_requests(transcripts, judge_model=judge_model)
        log.info("batch: built %d requests for %d transcripts", len(requests), len(transcripts))
        submission = await submit_batch(client, requests, completion_window=completion_window)
        batch_id = submission.batch_id
        log.info("batch: persist this id to resume later: %s", batch_id)

    output_file_id = await wait_for_batch(client, batch_id, poll_interval_s=poll_interval_s)
    results = await download_results(client, output_file_id)
    log.info("batch: downloaded %d completed responses", len(results))

    n_written = assemble_outputs(
        transcripts, results, judge_model=judge_model, output_dir=output_dir
    )
    return {
        "transcripts": len(transcripts),
        "scored": n_written,
        "batch_id": batch_id,
        "results": len(results),
    }


__all__ = [
    "BatchSubmission",
    "assemble_outputs",
    "build_batch_requests",
    "download_results",
    "run_batch",
    "submit_batch",
    "wait_for_batch",
]
