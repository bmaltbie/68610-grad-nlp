"""Tests for the single-turn calibration runner.

Mocks the OpenAI client to avoid real API spend; the actual one-off
~$0.50 calibration is run manually after the rest of the pipeline is
verified.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from judging import calibrate
from judging.calibrate import (
    PUBLISHED_S_D_GPT4O_AITA_YTA,
    TOLERANCE_PP,
    _judge_singleturn,
    _load_sample,
    run_calibration,
)


# ---------------------------------------------------------------------------
# Sample loader
# ---------------------------------------------------------------------------


def _write_csv(tmp_path, n: int = 5) -> str:
    csv = tmp_path / "yta.csv"
    pd.DataFrame(
        {
            "prompt": [f"AITA prompt {i}" for i in range(n)],
            "validation_human": [1] * n,
            "indirectness_human": [0] * n,
            "framing_human": [1] * n,
        }
    ).to_csv(csv, index=False)
    return str(csv)


def test_load_sample_seeded_is_reproducible(tmp_path):
    csv = _write_csv(tmp_path, n=10)
    a = _load_sample(csv, n=3, seed=42)
    b = _load_sample(csv, n=3, seed=42)
    assert list(a["prompt"]) == list(b["prompt"])


def test_load_sample_rejects_too_large_n(tmp_path):
    csv = _write_csv(tmp_path, n=2)
    with pytest.raises(ValueError, match="only 2 rows"):
        _load_sample(csv, n=10, seed=0)


def test_load_sample_rejects_missing_columns(tmp_path):
    csv = tmp_path / "broken.csv"
    pd.DataFrame({"prompt": ["x"], "validation_human": [1]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        _load_sample(csv, n=1, seed=0)


# ---------------------------------------------------------------------------
# _judge_singleturn parses the integer
# ---------------------------------------------------------------------------


def _mock_chat_client(text: str) -> Any:
    create = AsyncMock()
    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(content=text))]
    create.return_value = completion
    client = MagicMock()
    client.chat.completions.create = create
    return client


@pytest.mark.parametrize("text,expected", [("0", 0), ("1", 1), ("1\n", 1), ("0.", 0), ("the answer is 1", 1)])
def test_judge_singleturn_parses_integer(text, expected):
    client = _mock_chat_client(text)
    out = asyncio.run(
        _judge_singleturn(
            client,
            judge_model="gpt-4o-test",
            rubric="validation",
            question="Q",
            advice="A",
        )
    )
    assert out == expected


def test_judge_singleturn_returns_none_on_unparseable():
    client = _mock_chat_client("nope")
    out = asyncio.run(
        _judge_singleturn(
            client, judge_model="gpt-4o-test", rubric="validation", question="Q", advice="A"
        )
    )
    assert out is None


def test_judge_singleturn_uses_max_tokens_2_and_no_seed():
    client = _mock_chat_client("1")
    asyncio.run(
        _judge_singleturn(
            client,
            judge_model="gpt-4o-test",
            rubric="indirectness",
            question="Q",
            advice="A",
        )
    )
    _, kwargs = client.chat.completions.create.call_args
    assert kwargs["max_tokens"] == 2
    # Upstream parity: no temperature, no seed override.
    assert "seed" not in kwargs
    assert "temperature" not in kwargs


# ---------------------------------------------------------------------------
# Full run_calibration with deterministic mock client
# ---------------------------------------------------------------------------


def _make_dual_client(*, generate_text: str, judge_score: str) -> Any:
    """Mock client whose chat.completions.create returns text that varies by prompt size.

    We want generation calls (long prompt, no max_tokens) to return
    ``generate_text`` and judge calls (short max_tokens=2) to return
    ``judge_score``. Distinguish by inspecting kwargs in the side_effect.
    """

    async def _create(*args, **kwargs):
        completion = MagicMock()
        if kwargs.get("max_tokens") == 2:
            content = judge_score
        else:
            content = generate_text
        completion.choices = [MagicMock(message=MagicMock(content=content))]
        return completion

    create = AsyncMock(side_effect=_create)
    client = MagicMock()
    client.chat.completions.create = create
    return client


def test_run_calibration_passes_when_judge_matches_published(tmp_path):
    """Synthesize a perfect calibration: model_rate − human_rate ≈ published.

    Human labels in CSV: validation_human=1, indirectness_human=0,
    framing_human=1 (set in _write_csv). Judge always returns 1.
    Sample S^d per dim:
      validation:   1 - 1 = 0   (published 0.34, diff 0.34 → FAIL)
      indirectness: 1 - 0 = 1   (published 0.43, diff 0.57 → FAIL)
      framing:      1 - 1 = 0   (published 0.34, diff 0.34 → FAIL)
    So we expect FAIL — not a pass test, just a "result returned cleanly" test.
    """
    csv = _write_csv(tmp_path, n=5)
    client = _make_dual_client(generate_text="some advice", judge_score="1")
    result = asyncio.run(
        run_calibration(csv_path=csv, n=5, seed=0, concurrency=2, client=client)
    )
    assert result.n == 5
    assert len(result.per_dim) == 3
    # All three dims fail this synthetic case (model always 1).
    assert not result.overall_pass
    val = next(d for d in result.per_dim if d.dimension == "validation")
    assert val.model_rate == pytest.approx(1.0)
    assert val.sample_delta == pytest.approx(0.0)
    assert not val.passed


def test_run_calibration_passes_when_perfectly_aligned(tmp_path):
    """Force model and human rates to differ by exactly the published delta.

    Set human_rate to 0.0 for all dims; force judge to emit 1 with
    probability matching the published delta.
    """
    csv_path = tmp_path / "yta.csv"
    n = 100
    pd.DataFrame(
        {
            "prompt": [f"AITA prompt {i}" for i in range(n)],
            "validation_human": [0] * n,
            "indirectness_human": [0] * n,
            "framing_human": [0] * n,
        }
    ).to_csv(csv_path, index=False)

    # Build a side-effect that returns "1" for the first floor(p*n*3)
    # judge calls per dim and "0" after, so per-dim mean lands close to
    # the published value.
    from judging.calibrate import DIMENSIONS

    counters = dict.fromkeys(DIMENSIONS, 0)
    targets = {d: round(PUBLISHED_S_D_GPT4O_AITA_YTA[d] * n) for d in DIMENSIONS}

    async def _create(*args, **kwargs):
        completion = MagicMock()
        if kwargs.get("max_tokens") == 2:
            # Identify the dimension by the rubric body in the user msg.
            user = kwargs["messages"][1]["content"]
            if "challenges the premise" in user:
                dim = "framing"
            elif "suggestive language" in user:
                dim = "indirectness"
            else:
                dim = "validation"
            counters[dim] += 1
            content = "1" if counters[dim] <= targets[dim] else "0"
        else:
            content = "some advice"
        completion.choices = [MagicMock(message=MagicMock(content=content))]
        return completion

    create = AsyncMock(side_effect=_create)
    client = MagicMock()
    client.chat.completions.create = create

    result = asyncio.run(
        run_calibration(
            csv_path=csv_path, n=n, seed=0, concurrency=4, client=client
        )
    )
    # All deltas should be very close to published.
    for d in result.per_dim:
        assert d.diff_pp <= TOLERANCE_PP, f"{d.dimension}: {d}"
    assert result.overall_pass


def test_calibration_report_includes_pass_fail(tmp_path):
    csv = _write_csv(tmp_path, n=3)
    client = _make_dual_client(generate_text="advice", judge_score="0")
    result = asyncio.run(
        run_calibration(csv_path=csv, n=3, seed=0, concurrency=1, client=client)
    )
    report = result.report()
    assert "Calibration vs ELEPHANT Table 3" in report
    assert "OVERALL" in report
    for dim in ("validation", "indirectness", "framing"):
        assert dim in report
