"""Moral verdict extractor (dim-4 prep).

The official ELEPHANT repo prompts the target to emit ``YTA``/``NTA``
directly and regex-matches on the first 5 chars. Our targets give
free-form multi-turn advice, so we run a structured-output extractor
that returns ``{reasoning, verdict ∈ {NTA, YTA, UNCLEAR}}`` per target
turn. UNCLEAR is excluded from the moral-rate denominator downstream.
"""

from __future__ import annotations

import asyncio
import logging

from openai import AsyncOpenAI

from judging.prompts import build_verdict_messages
from judging.schemas import TargetTurnVerdict, Transcript, VerdictOutput
from judging.utils import RateLimiter, call_structured

log = logging.getLogger(__name__)


class VerdictExtractor:
    """Async per-turn NTA/YTA/UNCLEAR extractor."""

    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        model: str = "gpt-4o",
        concurrency: int = 8,
        temperature: float = 0.0,
        seed: int = 42,
    ):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.limiter = RateLimiter(concurrency)

    async def extract_target_turn(
        self, transcript: Transcript, target_turn: int
    ) -> TargetTurnVerdict:
        messages = build_verdict_messages(transcript, target_turn)
        async with self.limiter:
            vo: VerdictOutput = await call_structured(
                self.client,
                model=self.model,
                messages=messages,
                response_format=VerdictOutput,
                temperature=self.temperature,
                seed=self.seed,
            )
        return TargetTurnVerdict.from_verdict_output(
            turn=target_turn,
            preceded_by=transcript.preceding_auditor_turn_type(target_turn),
            vo=vo,
        )

    async def extract_transcript(
        self, transcript: Transcript
    ) -> list[TargetTurnVerdict]:
        target_turns = [t.turn for t in transcript.target_turns()]
        return await asyncio.gather(
            *[self.extract_target_turn(transcript, t) for t in target_turns]
        )


__all__ = ["VerdictExtractor"]
