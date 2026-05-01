"""ELEPHANT judge for dimensions 1–3 (validation, indirectness, framing).

Per-target-turn × 3-dim async scoring. The three dimension calls fire
concurrently via ``asyncio.gather``; calls across target turns share a
single ``RateLimiter``.

Score semantics (after framing internal flip): 1 = sycophantic, 0 = not.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from openai import AsyncOpenAI

from judging.prompts import (
    RUBRICS,
    RubricName,
    build_judge_messages,
)
from judging.schemas import (
    JudgeOutput,
    TargetTurnScore,
    Transcript,
    TurnDimensionScore,
)
from judging.utils import RateLimiter, call_structured

log = logging.getLogger(__name__)

DIMENSIONS: tuple[RubricName, ...] = ("validation", "indirectness", "framing")


class ElephantJudge:
    """Async judge for ELEPHANT dimensions 1–3."""

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

    async def _score_dim(
        self,
        rubric: RubricName,
        *,
        transcript: Optional[Transcript] = None,
        target_turn: Optional[int] = None,
        question: Optional[str] = None,
        advice: Optional[str] = None,
    ) -> TurnDimensionScore:
        messages = build_judge_messages(
            rubric,
            transcript=transcript,
            target_turn=target_turn,
            question=question,
            advice=advice,
        )
        async with self.limiter:
            jo: JudgeOutput = await call_structured(
                self.client,
                model=self.model,
                messages=messages,
                response_format=JudgeOutput,
                temperature=self.temperature,
                seed=self.seed,
            )

        # Framing rubric scores 0 = challenges, 1 = does not challenge.
        # In our schema 1 = sycophantic on every dimension; framing's
        # raw label already matches that direction (1 = does not
        # challenge = sycophantic), so no flip is needed.
        return TurnDimensionScore.from_judge_output(jo)

    async def score_target_turn(
        self, transcript: Transcript, target_turn: int
    ) -> TargetTurnScore:
        """Score one target turn across all three dimensions concurrently."""
        scores = await asyncio.gather(
            *[
                self._score_dim(
                    dim, transcript=transcript, target_turn=target_turn
                )
                for dim in DIMENSIONS
            ]
        )
        validation, indirectness, framing = scores
        return TargetTurnScore(
            turn=target_turn,
            preceded_by=transcript.preceding_auditor_turn_type(target_turn),
            validation=validation,
            indirectness=indirectness,
            framing=framing,
        )

    async def score_transcript(self, transcript: Transcript) -> list[TargetTurnScore]:
        """Score every target turn in a transcript."""
        target_turns = [t.turn for t in transcript.target_turns()]
        return await asyncio.gather(
            *[self.score_target_turn(transcript, t) for t in target_turns]
        )

    async def score_singleturn(
        self, *, question: str, advice: str
    ) -> dict[RubricName, TurnDimensionScore]:
        """Single-turn calibration mode: score one (question, advice) pair.

        Mirrors the original ELEPHANT scorer's per-row output across the
        three dimensions, but uses structured output (with reasoning)
        instead of ``max_tokens=2``. The calibration script also has a
        raw mode using ``call_raw`` directly; this one keeps reasoning
        for inspection.
        """
        scores = await asyncio.gather(
            *[
                self._score_dim(dim, question=question, advice=advice)
                for dim in DIMENSIONS
            ]
        )
        return dict(zip(DIMENSIONS, scores))


__all__ = ["ElephantJudge", "DIMENSIONS"]
