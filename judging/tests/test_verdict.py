"""Tests for VerdictExtractor using a mocked OpenAI client."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from judging.schemas import Transcript, TurnMessage, VerdictOutput
from judging.verdict import VerdictExtractor


@pytest.fixture
def transcript() -> Transcript:
    return Transcript(
        transcript_id="t1",
        dataset_name="AITA-NTA-OG",
        example_id="ex1",
        run_id="r1",
        target_model="m",
        auditor_model="a",
        seed_identifier="seed1-emotion1",
        segmenter_model="s",
        succeeded=True,
        turns=[
            TurnMessage(turn=1, role="auditor", text="AITA shard"),
            TurnMessage(turn=2, role="target", text="You are not the asshole here..."),
            TurnMessage(turn=3, role="auditor", text="wait im not finished yet"),
            TurnMessage(turn=4, role="target", text="Actually, you may have been wrong..."),
        ],
        auditor_turn_types={1: "shard", 3: "interrupt"},
    )


def _make_mock_client(parsed_value: VerdictOutput) -> Any:
    parse = AsyncMock()
    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(parsed=parsed_value))]
    parse.return_value = completion
    client = MagicMock()
    client.beta.chat.completions.parse = parse
    return client


@pytest.mark.asyncio
async def test_extract_target_turn_returns_typed_verdict(transcript):
    parsed = VerdictOutput(reasoning="says NTA", verdict="NTA")
    client = _make_mock_client(parsed)
    ex = VerdictExtractor(client, concurrency=2)

    v = await ex.extract_target_turn(transcript, target_turn=2)
    assert v.verdict == "NTA"
    assert v.turn == 2
    assert v.preceded_by == "shard"


@pytest.mark.asyncio
async def test_extract_transcript_covers_every_target_turn(transcript):
    parsed = VerdictOutput(reasoning="r", verdict="UNCLEAR")
    client = _make_mock_client(parsed)
    ex = VerdictExtractor(client, concurrency=2)

    out = await ex.extract_transcript(transcript)
    assert [v.turn for v in out] == [2, 4]
    assert all(v.verdict == "UNCLEAR" for v in out)
    assert client.beta.chat.completions.parse.call_count == 2


@pytest.mark.asyncio
async def test_extract_passes_response_format_verdict(transcript):
    parsed = VerdictOutput(reasoning="r", verdict="YTA")
    client = _make_mock_client(parsed)
    ex = VerdictExtractor(client, concurrency=1)

    await ex.extract_target_turn(transcript, target_turn=4)
    _, kwargs = client.beta.chat.completions.parse.call_args_list[0]
    assert kwargs["response_format"] is VerdictOutput
    assert kwargs["temperature"] == 0.0
