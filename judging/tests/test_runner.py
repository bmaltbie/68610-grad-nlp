"""Tests for runner orchestration.

Patches ``ElephantJudge``, ``VerdictExtractor``, and ``create_async_client``
so we exercise file-handling logic without API calls. Inputs come from
the real ``datasets/conversation_transcripts.jsonl``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from judging import runner
from judging.io import load_transcripts
from judging.schemas import (
    JudgeOutput,
    TargetTurnScore,
    TargetTurnVerdict,
    Transcript,
    TurnDimensionScore,
    VerdictOutput,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRANSCRIPTS = REPO_ROOT / "datasets" / "conversation_transcripts.jsonl"


# ---------------------------------------------------------------------------
# Fake judge / verdict that don't hit the network
# ---------------------------------------------------------------------------


class _FakeJudge:
    def __init__(self, *args, **kwargs):
        self.calls = 0

    async def score_transcript(self, transcript: Transcript) -> list[TargetTurnScore]:
        out = []
        for t in transcript.target_turns():
            self.calls += 1
            dim = TurnDimensionScore(reasoning="r", score=1)
            out.append(
                TargetTurnScore(
                    turn=t.turn,
                    preceded_by=transcript.preceding_auditor_turn_type(t.turn),
                    validation=dim,
                    indirectness=dim,
                    framing=dim,
                )
            )
        return out


class _FakeVerdict:
    def __init__(self, *args, **kwargs):
        self.calls = 0

    async def extract_transcript(self, transcript: Transcript) -> list[TargetTurnVerdict]:
        out = []
        for t in transcript.target_turns():
            self.calls += 1
            out.append(
                TargetTurnVerdict(
                    turn=t.turn,
                    preceded_by=transcript.preceding_auditor_turn_type(t.turn),
                    verdict="NTA",
                    reasoning="r",
                )
            )
        return out


@pytest.fixture
def patched_runner(monkeypatch):
    monkeypatch.setattr(runner, "ElephantJudge", _FakeJudge)
    monkeypatch.setattr(runner, "VerdictExtractor", _FakeVerdict)
    monkeypatch.setattr(runner, "create_async_client", lambda: MagicMock())
    return runner


# ---------------------------------------------------------------------------
# Cost estimator
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TRANSCRIPTS.exists(), reason="conversation_transcripts.jsonl missing")
def test_estimate_calls_counts_dims_and_verdicts():
    transcripts = list(load_transcripts(TRANSCRIPTS))
    stats = runner.estimate_calls(transcripts)
    assert stats["transcripts"] == len(transcripts)
    expected_judge = sum(len(t.target_turns()) * 3 for t in transcripts)
    expected_verdict = sum(
        len(t.target_turns())
        for t in transcripts
        if t.dataset_name in {"AITA-NTA-OG", "AITA-NTA-FLIP"}
    )
    assert stats["judge_calls"] == expected_judge
    assert stats["verdict_calls"] == expected_verdict
    assert stats["total_calls"] == expected_judge + expected_verdict


# ---------------------------------------------------------------------------
# Score runner
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TRANSCRIPTS.exists(), reason="conversation_transcripts.jsonl missing")
def test_run_score_logs_estimate_before_calls(patched_runner, tmp_path, caplog):
    """Estimate must be logged at INFO level before any API call begins."""
    out = tmp_path / "out"
    with caplog.at_level("INFO", logger="judging.runner"):
        asyncio.run(
            patched_runner.run_score(
                input_path=TRANSCRIPTS, output_dir=out, max_count=1
            )
        )
    estimate_msgs = [r for r in caplog.records if "estimate:" in r.message]
    assert estimate_msgs, "expected an INFO log line starting with 'estimate:'"


@pytest.mark.skipif(not TRANSCRIPTS.exists(), reason="conversation_transcripts.jsonl missing")
def test_run_score_writes_trace_and_index(patched_runner, tmp_path):
    out = tmp_path / "out"
    stats = asyncio.run(
        patched_runner.run_score(
            input_path=TRANSCRIPTS,
            output_dir=out,
            max_count=1,
        )
    )
    assert stats["scored"] == 1
    # Exactly one trace file.
    traces = list((out / "traces").glob("*.json"))
    assert len(traces) == 1
    trace = json.loads(traces[0].read_text())
    assert "per_target_turn_scores" in trace
    assert trace["per_target_turn_scores"], "expected at least one scored target turn"
    # Index file has one line.
    idx_path = out / "judge.jsonl"
    idx_lines = idx_path.read_text().strip().splitlines()
    assert len(idx_lines) == 1
    idx = json.loads(idx_lines[0])
    assert idx["transcript_id"] == trace["transcript_id"]
    assert idx["per_turn_summary"]
    # NTA-OG transcripts get a moral verdict block.
    if trace["dataset_name"] in {"AITA-NTA-OG", "AITA-NTA-FLIP"}:
        assert trace["per_target_turn_moral_verdict"]


@pytest.mark.skipif(not TRANSCRIPTS.exists(), reason="conversation_transcripts.jsonl missing")
def test_run_score_resume_skips_existing(patched_runner, tmp_path):
    out = tmp_path / "out"
    asyncio.run(
        patched_runner.run_score(
            input_path=TRANSCRIPTS, output_dir=out, max_count=1
        )
    )
    # Re-run with --resume; should be a no-op.
    stats = asyncio.run(
        patched_runner.run_score(
            input_path=TRANSCRIPTS, output_dir=out, max_count=1, resume=True
        )
    )
    assert stats["scored"] == 0


def test_run_score_batch_dispatches_to_batch_runner(monkeypatch, tmp_path):
    """When batch=True, run_score forwards to batch_runner.run_batch."""
    from judging import batch_runner

    called = {}

    async def _fake_run_batch(**kwargs):
        called.update(kwargs)
        return {"transcripts": 0, "scored": 0}

    monkeypatch.setattr(batch_runner, "run_batch", _fake_run_batch)

    asyncio.run(
        runner.run_score(
            input_path=tmp_path / "x.jsonl",
            output_dir=tmp_path / "out",
            batch=True,
            batch_id="batch_abc",
            poll_interval_s=5,
        )
    )
    assert called["batch_id"] == "batch_abc"
    assert called["poll_interval_s"] == 5


# ---------------------------------------------------------------------------
# Moral runner (second pass)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TRANSCRIPTS.exists(), reason="conversation_transcripts.jsonl missing")
def test_run_moral_writes_jsonl_after_score(patched_runner, tmp_path):
    """End-to-end: score a small batch then run the moral second pass.

    Skips with a clear message if the test transcripts don't include any
    OG⨝FLIP pair (the current pilot file has only OG records).
    """
    out = tmp_path / "out"
    asyncio.run(
        patched_runner.run_score(input_path=TRANSCRIPTS, output_dir=out)
    )
    moral_out = out / "moral.jsonl"
    n = patched_runner.run_moral(judge_jsonl=out / "judge.jsonl", output_path=moral_out)
    if n == 0:
        pytest.skip("no OG⨝FLIP pairs in current test transcripts")
    assert moral_out.exists()
    lines = moral_out.read_text().strip().splitlines()
    assert len(lines) == n


def test_run_moral_errors_when_traces_missing(tmp_path):
    judge_jsonl = tmp_path / "judge.jsonl"
    judge_jsonl.write_text("")
    with pytest.raises(FileNotFoundError, match="traces"):
        runner.run_moral(judge_jsonl=judge_jsonl, output_path=tmp_path / "moral.jsonl")
