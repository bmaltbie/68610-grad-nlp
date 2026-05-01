"""Tests for the OpenAI Batch API runner.

Mocks the client's ``files``, ``batches``, and download endpoints so we
exercise request building, custom_id round-tripping, polling logic, and
output assembly without spending API money.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from judging import batch_runner
from judging.batch_runner import (
    _judge_custom_id,
    _parse_custom_id,
    _strict_response_format,
    _verdict_custom_id,
    assemble_outputs,
    build_batch_requests,
    download_results,
    run_batch,
    submit_batch,
    wait_for_batch,
)
from judging.io import load_transcripts
from judging.judge import DIMENSIONS
from judging.schemas import (
    JudgeOutput,
    Transcript,
    TurnMessage,
    VerdictOutput,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRANSCRIPTS = REPO_ROOT / "datasets" / "conversation_transcripts.jsonl"


# ---------------------------------------------------------------------------
# Custom ID encode/decode round-trip
# ---------------------------------------------------------------------------


def test_judge_custom_id_round_trip():
    cid = _judge_custom_id("abc-123", 4, "validation")
    kind, tid, turn, dim = _parse_custom_id(cid)
    assert kind == "judge"
    assert tid == "abc-123"
    assert turn == 4
    assert dim == "validation"


def test_verdict_custom_id_round_trip():
    cid = _verdict_custom_id("abc-123", 8)
    kind, tid, turn, dim = _parse_custom_id(cid)
    assert kind == "verdict"
    assert tid == "abc-123"
    assert turn == 8
    assert dim is None


def test_parse_custom_id_rejects_unknown():
    with pytest.raises(ValueError, match="unknown custom_id"):
        _parse_custom_id("bogus|x|1")


# ---------------------------------------------------------------------------
# Strict JSON schema for response_format
# ---------------------------------------------------------------------------


def test_strict_response_format_for_judge_output():
    rf = _strict_response_format(JudgeOutput)
    assert rf["type"] == "json_schema"
    js = rf["json_schema"]
    assert js["name"] == "JudgeOutput"
    assert js["strict"] is True
    schema = js["schema"]
    # All object schemas should disallow additional properties.
    assert schema["additionalProperties"] is False
    # JudgeOutput.score is Literal[0, 1] → enum of [0, 1]
    score_schema = schema["properties"]["score"]
    assert set(score_schema["enum"]) == {0, 1}


def test_strict_response_format_for_verdict_output():
    rf = _strict_response_format(VerdictOutput)
    schema = rf["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    verdict_schema = schema["properties"]["verdict"]
    assert set(verdict_schema["enum"]) == {"NTA", "YTA", "UNCLEAR"}


# ---------------------------------------------------------------------------
# build_batch_requests
# ---------------------------------------------------------------------------


def _mini_transcript(dataset: str = "AITA-NTA-OG") -> Transcript:
    return Transcript(
        transcript_id="t1",
        dataset_name=dataset,  # type: ignore[arg-type]
        example_id="ex1",
        run_id="r",
        target_model="m",
        auditor_model="a",
        seed_identifier="seed1-emotion1",
        segmenter_model="s",
        succeeded=True,
        turns=[
            TurnMessage(turn=1, role="auditor", text="AITA shard"),
            TurnMessage(turn=2, role="target", text="reply A"),
            TurnMessage(turn=3, role="auditor", text="wait im not finished yet"),
            TurnMessage(turn=4, role="target", text="reply B"),
        ],
        auditor_turn_types={1: "shard", 3: "interrupt"},
    )


def test_build_batch_requests_counts_and_shape_nta_og():
    t = _mini_transcript("AITA-NTA-OG")
    reqs = build_batch_requests([t], judge_model="gpt-4o-test")
    # 2 target turns × (3 dims + 1 verdict) = 8 requests
    assert len(reqs) == 8
    judge_reqs = [r for r in reqs if r["custom_id"].startswith("judge|")]
    verdict_reqs = [r for r in reqs if r["custom_id"].startswith("verdict|")]
    assert len(judge_reqs) == 6
    assert len(verdict_reqs) == 2
    # Every request hits chat completions with our judge model.
    for r in reqs:
        assert r["method"] == "POST"
        assert r["url"] == "/v1/chat/completions"
        assert r["body"]["model"] == "gpt-4o-test"
        assert r["body"]["temperature"] == 0
        assert r["body"]["seed"] == 42
        assert r["body"]["response_format"]["type"] == "json_schema"


def test_build_batch_requests_skips_verdict_for_yta():
    t = _mini_transcript("AITA-YTA")
    reqs = build_batch_requests([t], judge_model="gpt-4o")
    assert len(reqs) == 2 * len(DIMENSIONS)  # only judge calls
    assert not any(r["custom_id"].startswith("verdict|") for r in reqs)


# ---------------------------------------------------------------------------
# submit_batch / wait_for_batch / download_results
# ---------------------------------------------------------------------------


def _mock_client_for_submit(file_id="file-1", batch_id="batch-1") -> Any:
    client = MagicMock()
    client.files.create = AsyncMock(return_value=MagicMock(id=file_id))
    client.batches.create = AsyncMock(return_value=MagicMock(id=batch_id))
    return client


def test_submit_batch_uploads_and_creates(tmp_path):
    client = _mock_client_for_submit()
    requests = [{"custom_id": "judge|t1|2|validation", "body": {"x": 1}}]
    sub = asyncio.run(submit_batch(client, requests))
    assert sub.batch_id == "batch-1"
    assert sub.input_file_id == "file-1"
    assert sub.n_requests == 1
    # files.create was called with our serialized JSONL.
    _, kwargs = client.files.create.call_args
    assert kwargs["purpose"] == "batch"
    # batches.create was called with the file_id.
    _, kwargs = client.batches.create.call_args
    assert kwargs["input_file_id"] == "file-1"
    assert kwargs["endpoint"] == "/v1/chat/completions"


def _batch_status_sequence(*statuses) -> AsyncMock:
    """Mock batches.retrieve that yields each status in order, then loops on last."""
    m = AsyncMock()
    objs = []
    for s in statuses:
        b = MagicMock()
        b.status = s
        b.output_file_id = "out-file-1" if s == "completed" else None
        b.request_counts = MagicMock(completed=10, total=10, failed=0)
        objs.append(b)
    m.side_effect = list(objs) + [objs[-1]] * 100  # extra sentinels in case
    return m


def test_wait_for_batch_returns_output_file_id():
    client = MagicMock()
    client.batches.retrieve = _batch_status_sequence("in_progress", "finalizing", "completed")
    out_id = asyncio.run(
        wait_for_batch(client, "batch-1", poll_interval_s=0)
    )
    assert out_id == "out-file-1"


def test_wait_for_batch_raises_on_terminal_failure():
    client = MagicMock()
    client.batches.retrieve = _batch_status_sequence("in_progress", "failed")
    with pytest.raises(RuntimeError, match="status=failed"):
        asyncio.run(wait_for_batch(client, "batch-x", poll_interval_s=0))


def test_download_results_parses_jsonl_skipping_errored_lines():
    client = MagicMock()
    body_ok = {
        "choices": [{"message": {"content": json.dumps({"reasoning": "r", "score": 1})}}]
    }
    payload = "\n".join(
        [
            json.dumps({"custom_id": "judge|t1|2|validation", "response": {"body": body_ok}, "error": None}),
            json.dumps({"custom_id": "judge|t1|2|framing", "error": {"code": "rate_limit", "message": "x"}}),
            json.dumps({"custom_id": "judge|t1|4|validation", "response": {"body": body_ok}, "error": None}),
        ]
    )
    fake_resp = MagicMock()
    fake_resp.text = payload
    client.files.content = AsyncMock(return_value=fake_resp)

    results = asyncio.run(download_results(client, "out-file-1"))
    assert set(results.keys()) == {"judge|t1|2|validation", "judge|t1|4|validation"}


# ---------------------------------------------------------------------------
# assemble_outputs
# ---------------------------------------------------------------------------


def _judge_body(score: int) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps({"reasoning": "synthetic", "score": score})
                }
            }
        ]
    }


def _verdict_body(verdict: str) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps({"reasoning": "synthetic", "verdict": verdict})
                }
            }
        ]
    }


def test_assemble_outputs_writes_traces_and_index(tmp_path):
    t = _mini_transcript("AITA-NTA-OG")
    results: dict[str, dict] = {}
    for tt in t.target_turns():
        for dim in DIMENSIONS:
            results[_judge_custom_id(t.transcript_id, tt.turn, dim)] = _judge_body(1)
        results[_verdict_custom_id(t.transcript_id, tt.turn)] = _verdict_body("NTA")

    n = assemble_outputs(
        [t], results, judge_model="gpt-4o-test", output_dir=tmp_path / "out"
    )
    assert n == 1
    traces = list((tmp_path / "out" / "traces").glob("*.json"))
    assert len(traces) == 1
    trace = json.loads(traces[0].read_text())
    assert trace["transcript_id"] == t.transcript_id
    assert len(trace["per_target_turn_scores"]) == 2  # 2 target turns
    assert len(trace["per_target_turn_moral_verdict"]) == 2
    idx_lines = (tmp_path / "out" / "judge.jsonl").read_text().strip().splitlines()
    assert len(idx_lines) == 1


def test_assemble_outputs_skips_transcript_with_missing_results(tmp_path, caplog):
    t = _mini_transcript("AITA-NTA-OG")
    # Pass empty results — no batch lines for this transcript.
    with caplog.at_level("WARNING"):
        n = assemble_outputs(
            [t], {}, judge_model="gpt-4o-test", output_dir=tmp_path / "out"
        )
    assert n == 0
    assert any("no batch results" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# End-to-end run_batch with mocked client
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TRANSCRIPTS.exists(), reason="conversation_transcripts.jsonl missing")
def test_run_batch_end_to_end(monkeypatch, tmp_path):
    # Prepare a mock client whose submit→wait→download path returns
    # canned judge+verdict responses for every (turn, dim) we need.
    transcripts = list(load_transcripts(TRANSCRIPTS))
    t = transcripts[0]

    # Build the expected results dictionary.
    results: dict[str, dict] = {}
    for tt in t.target_turns():
        for dim in DIMENSIONS:
            results[_judge_custom_id(t.transcript_id, tt.turn, dim)] = _judge_body(1)
        if t.dataset_name in {"AITA-NTA-OG", "AITA-NTA-FLIP"}:
            results[_verdict_custom_id(t.transcript_id, tt.turn)] = _verdict_body("NTA")

    client = MagicMock()
    client.files.create = AsyncMock(return_value=MagicMock(id="f1"))
    client.batches.create = AsyncMock(return_value=MagicMock(id="b1"))
    client.batches.retrieve = _batch_status_sequence("completed")

    payload = "\n".join(
        json.dumps(
            {"custom_id": cid, "response": {"body": body}, "error": None}
        )
        for cid, body in results.items()
    )
    fake_resp = MagicMock()
    fake_resp.text = payload
    client.files.content = AsyncMock(return_value=fake_resp)

    monkeypatch.setattr(batch_runner, "create_async_client", lambda: client)

    out = tmp_path / "out"
    stats = asyncio.run(
        run_batch(
            input_path=TRANSCRIPTS,
            output_dir=out,
            judge_model="gpt-4o-test",
            poll_interval_s=0,
            client=client,
            max_count=1,
        )
    )
    assert stats["scored"] == 1
    assert stats["batch_id"] == "b1"
    assert (out / "judge.jsonl").exists()
    traces = list((out / "traces").glob("*.json"))
    assert len(traces) == 1


def test_run_batch_resume_skips_submission(tmp_path, monkeypatch):
    """When batch_id is provided, run_batch should NOT call submit_batch."""
    if not TRANSCRIPTS.exists():
        pytest.skip("conversation_transcripts.jsonl missing")

    submit_calls: list = []

    async def _fake_submit(*args, **kwargs):
        submit_calls.append(args)
        raise AssertionError("submit_batch should not be called when batch_id is set")

    monkeypatch.setattr(batch_runner, "submit_batch", _fake_submit)

    client = MagicMock()
    client.batches.retrieve = _batch_status_sequence("completed")
    client.files.content = AsyncMock(return_value=MagicMock(text=""))

    stats = asyncio.run(
        run_batch(
            input_path=TRANSCRIPTS,
            output_dir=tmp_path / "out",
            judge_model="gpt-4o-test",
            batch_id="batch-existing",
            poll_interval_s=0,
            client=client,
            max_count=1,
        )
    )
    assert submit_calls == []
    assert stats["batch_id"] == "batch-existing"
