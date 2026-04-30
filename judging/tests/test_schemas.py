"""Tests for ``judging.schemas``.

Covers acceptance/rejection of valid/invalid records and JSON
round-tripping for every model that gets serialised to disk.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from judging.schemas import (
    HumanBaselines,
    JudgeIndexRecord,
    JudgeOutput,
    MoralPair,
    MoralTurnComparison,
    TargetTurnScore,
    TargetTurnVerdict,
    Transcript,
    TranscriptScore,
    TurnDimensionScore,
    TurnMessage,
    TurnSummary,
    VerdictOutput,
)


# ---------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------


def _good_turns() -> list[TurnMessage]:
    return [
        TurnMessage(turn=1, role="auditor", text="bullet 1"),
        TurnMessage(turn=2, role="target", text="reply 1"),
        TurnMessage(turn=3, role="auditor", text="wait im not finished yet"),
        TurnMessage(turn=4, role="target", text="reply 2"),
    ]


def _good_transcript(**overrides) -> Transcript:
    base = dict(
        transcript_id="abc-123",
        dataset_name="AITA-NTA-OG",
        example_id="ek759w",
        run_id="run-1",
        target_model="openai/gpt-x",
        auditor_model="openai/gpt-x",
        seed_identifier="seed1-emotion1",
        segmenter_model="claude-x",
        succeeded=True,
        reason="",
        turns=_good_turns(),
        auditor_turn_types={1: "shard", 3: "interrupt"},
    )
    base.update(overrides)
    return Transcript(**base)


def test_transcript_accepts_alternating_roles() -> None:
    t = _good_transcript()
    assert t.transcript_id == "abc-123"
    assert len(t.turns) == 4


def test_transcript_rejects_first_turn_target() -> None:
    bad_turns = [
        TurnMessage(turn=1, role="target", text="reply"),
        TurnMessage(turn=2, role="auditor", text="bullet"),
    ]
    with pytest.raises(ValidationError, match="role='target'"):
        _good_transcript(turns=bad_turns)


def test_transcript_rejects_consecutive_same_role() -> None:
    bad_turns = [
        TurnMessage(turn=1, role="auditor", text="b1"),
        TurnMessage(turn=2, role="auditor", text="b1 dup"),
        TurnMessage(turn=3, role="target", text="r1"),
    ]
    with pytest.raises(ValidationError, match="expected 'target'"):
        _good_transcript(turns=bad_turns)


def test_transcript_rejects_misnumbered_turns() -> None:
    bad_turns = [
        TurnMessage(turn=1, role="auditor", text="b1"),
        TurnMessage(turn=5, role="target", text="r1"),
    ]
    with pytest.raises(ValidationError, match="expected 2"):
        _good_transcript(turns=bad_turns)


def test_transcript_rejects_empty_turns() -> None:
    with pytest.raises(ValidationError, match="has no turns"):
        _good_transcript(turns=[])


def test_transcript_rejects_invalid_dataset_name() -> None:
    with pytest.raises(ValidationError):
        _good_transcript(dataset_name="OEQ")


def test_auditor_target_helpers() -> None:
    t = _good_transcript()
    assert [m.text for m in t.auditor_turns()] == [
        "bullet 1",
        "wait im not finished yet",
    ]
    assert [m.text for m in t.target_turns()] == ["reply 1", "reply 2"]


def test_preceding_auditor_turn_type() -> None:
    t = _good_transcript()
    assert t.preceding_auditor_turn_type(2) == "shard"
    assert t.preceding_auditor_turn_type(4) == "interrupt"


def test_preceding_auditor_turn_type_rejects_non_target_turn() -> None:
    t = _good_transcript()
    with pytest.raises(ValueError, match="not preceded by an auditor turn"):
        t.preceding_auditor_turn_type(3)  # turn 3 is auditor, so target=3 is wrong


def test_preceding_auditor_turn_type_missing_classification() -> None:
    t = _good_transcript(auditor_turn_types={1: "shard"})
    with pytest.raises(ValueError, match="missing auditor_turn_types"):
        t.preceding_auditor_turn_type(4)


def test_human_baselines_optional_fields() -> None:
    b = HumanBaselines(validation_human=1)
    assert b.validation_human == 1
    assert b.indirectness_human is None
    assert b.framing_human is None


def test_transcript_json_round_trip() -> None:
    t = _good_transcript(human_baselines=HumanBaselines(validation_human=0))
    j = t.model_dump_json()
    parsed = Transcript.model_validate_json(j)
    assert parsed == t


# ---------------------------------------------------------------------------
# Structured-output schemas
# ---------------------------------------------------------------------------


def test_judge_output_accepts_binary_score() -> None:
    j = JudgeOutput(reasoning="because foo", score=1)
    assert j.score == 1


def test_judge_output_rejects_non_binary_score() -> None:
    with pytest.raises(ValidationError):
        JudgeOutput(reasoning="x", score=2)


def test_verdict_output_categories() -> None:
    for v in ("NTA", "YTA", "UNCLEAR"):
        assert VerdictOutput(reasoning="x", verdict=v).verdict == v
    with pytest.raises(ValidationError):
        VerdictOutput(reasoning="x", verdict="MAYBE")


# ---------------------------------------------------------------------------
# Output records
# ---------------------------------------------------------------------------


def _good_target_turn_score(turn: int = 2) -> TargetTurnScore:
    dim = TurnDimensionScore(reasoning="r", score=1)
    return TargetTurnScore(
        turn=turn,
        preceded_by="shard",
        validation=dim,
        indirectness=dim,
        framing=dim,
    )


def test_target_turn_score_round_trip() -> None:
    s = _good_target_turn_score()
    j = s.model_dump_json()
    parsed = TargetTurnScore.model_validate_json(j)
    assert parsed == s


def test_turn_dimension_score_from_judge_output() -> None:
    jo = JudgeOutput(reasoning="r", score=0)
    d = TurnDimensionScore.from_judge_output(jo)
    assert d.reasoning == "r"
    assert d.score == 0


def test_target_turn_verdict_from_verdict_output() -> None:
    vo = VerdictOutput(reasoning="r", verdict="NTA")
    v = TargetTurnVerdict.from_verdict_output(turn=2, preceded_by="shard", vo=vo)
    assert v.turn == 2
    assert v.preceded_by == "shard"
    assert v.verdict == "NTA"


def test_transcript_score_round_trip() -> None:
    s = TranscriptScore(
        transcript_id="t-1",
        example_id="ek759w",
        dataset_name="AITA-YTA",
        target_model="m",
        judge_model="gpt-4o",
        rubric_version="v1.0",
        judged_at=datetime(2026, 4, 30, tzinfo=timezone.utc),
        per_target_turn_scores=[_good_target_turn_score(2), _good_target_turn_score(4)],
    )
    j = s.model_dump_json()
    parsed = TranscriptScore.model_validate_json(j)
    assert parsed == s


def test_judge_index_record_round_trip() -> None:
    rec = JudgeIndexRecord(
        transcript_id="t-1",
        example_id="ek759w",
        dataset_name="AITA-YTA",
        target_model="m",
        judge_model="gpt-4o",
        rubric_version="v1.0",
        trace_path="outputs/judging/traces/t-1.json",
        per_turn_summary=[
            TurnSummary(
                turn=2,
                preceded_by="shard",
                validation=1,
                indirectness=1,
                framing=0,
            )
        ],
    )
    parsed = JudgeIndexRecord.model_validate_json(rec.model_dump_json())
    assert parsed == rec


# ---------------------------------------------------------------------------
# Moral pair
# ---------------------------------------------------------------------------


def _verdict(verdict: str = "NTA") -> TargetTurnVerdict:
    return TargetTurnVerdict(
        turn=2, preceded_by="shard", verdict=verdict, reasoning="r"
    )


def test_moral_pair_round_trip() -> None:
    p = MoralPair(
        example_id="ek759w",
        target_model="m",
        og_transcript_id="og-1",
        flip_transcript_id="flip-1",
        per_turn_moral=[
            MoralTurnComparison(
                target_turn_position=1,
                og=_verdict("NTA"),
                flip=_verdict("NTA"),
                moral_sycophancy=1,
            ),
            MoralTurnComparison(
                target_turn_position=2,
                og=_verdict("NTA"),
                flip=_verdict("YTA"),
                moral_sycophancy=0,
            ),
            MoralTurnComparison(
                target_turn_position=3,
                og=_verdict("UNCLEAR"),
                flip=_verdict("NTA"),
                moral_sycophancy=None,
            ),
        ],
    )
    parsed = MoralPair.model_validate_json(p.model_dump_json())
    assert parsed == p
    assert parsed.per_turn_moral[2].moral_sycophancy is None


def test_moral_sycophancy_rejects_non_binary_when_set() -> None:
    with pytest.raises(ValidationError):
        MoralTurnComparison(
            target_turn_position=1,
            og=_verdict(),
            flip=_verdict(),
            moral_sycophancy=2,
        )
