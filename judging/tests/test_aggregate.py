"""Tests for aggregation functions on synthesised DataFrames."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from judging.aggregate import (
    DIMENSION_COLS,
    compute_close_turn_rate,
    compute_delta,
    compute_moral_rate,
    compute_rate,
    load_index_df,
    load_moral_df,
)


# ---------------------------------------------------------------------------
# load_index_df
# ---------------------------------------------------------------------------


def _write_index(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "judge.jsonl"
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return path


def _index_record(
    *,
    transcript_id: str,
    example_id: str,
    dataset_name: str,
    target_model: str,
    summaries: list[dict],
) -> dict:
    return {
        "transcript_id": transcript_id,
        "example_id": example_id,
        "dataset_name": dataset_name,
        "target_model": target_model,
        "judge_model": "gpt-4o-test",
        "rubric_version": "v1",
        "trace_path": f"traces/{transcript_id}.json",
        "per_turn_summary": summaries,
    }


def test_load_index_df_long_shape(tmp_path):
    rec = _index_record(
        transcript_id="t1",
        example_id="ex1",
        dataset_name="AITA-YTA",
        target_model="m",
        summaries=[
            {"turn": 2, "preceded_by": "shard", "validation": 1, "indirectness": 0, "framing": 1},
            {"turn": 4, "preceded_by": "aita_close", "validation": 0, "indirectness": 1, "framing": 1},
        ],
    )
    df = load_index_df(_write_index(tmp_path, [rec]))
    assert len(df) == 2 * len(DIMENSION_COLS)
    assert set(df["dimension"].unique()) == set(DIMENSION_COLS)
    # Spot-check a value: turn=4, dimension=indirectness, score=1.
    row = df[(df["turn"] == 4) & (df["dimension"] == "indirectness")].iloc[0]
    assert row["score"] == 1
    assert row["preceded_by"] == "aita_close"


def test_load_index_df_empty(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    df = load_index_df(path)
    assert df.empty
    assert "score" in df.columns


# ---------------------------------------------------------------------------
# compute_rate / compute_close_turn_rate
# ---------------------------------------------------------------------------


def _long_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_compute_rate_means_per_group():
    df = _long_df(
        [
            {"target_model": "A", "dimension": "validation", "score": 1, "preceded_by": "shard"},
            {"target_model": "A", "dimension": "validation", "score": 0, "preceded_by": "shard"},
            {"target_model": "A", "dimension": "validation", "score": 1, "preceded_by": "shard"},
            {"target_model": "B", "dimension": "validation", "score": 1, "preceded_by": "shard"},
        ]
    )
    out = compute_rate(df, by=["target_model", "dimension"])
    a = out[out["target_model"] == "A"].iloc[0]
    b = out[out["target_model"] == "B"].iloc[0]
    assert a["rate"] == pytest.approx(2 / 3)
    assert a["n"] == 3
    assert b["rate"] == 1.0
    assert b["n"] == 1


def test_compute_close_turn_rate_strict_filters_to_aita_close():
    df = _long_df(
        [
            {"transcript_id": "t1", "turn": 4, "target_model": "A", "dimension": "validation", "score": 0, "preceded_by": "shard", "dataset_name": "AITA-YTA"},
            {"transcript_id": "t1", "turn": 6, "target_model": "A", "dimension": "validation", "score": 1, "preceded_by": "aita_close", "dataset_name": "AITA-YTA"},
            {"transcript_id": "t2", "turn": 6, "target_model": "A", "dimension": "validation", "score": 1, "preceded_by": "aita_close", "dataset_name": "AITA-YTA"},
            {"transcript_id": "t1", "turn": 4, "target_model": "A", "dimension": "framing", "score": 0, "preceded_by": "interrupt", "dataset_name": "AITA-YTA"},
        ]
    )
    out = compute_close_turn_rate(df, fallback_to_last=False)
    # Two close-turn rows for validation (both 1) → rate=1, n=2.
    val = out[(out["dimension"] == "validation")].iloc[0]
    assert val["rate"] == 1.0
    assert val["n"] == 2
    # framing has no close-turn rows in strict mode → not in output.
    assert (out["dimension"] == "framing").sum() == 0


def test_compute_close_turn_rate_fallback_uses_last_turn():
    """When a transcript has no aita_close turn, fall back to its max-turn row."""
    df = _long_df(
        [
            # t1: has aita_close at turn 6
            {"transcript_id": "t1", "turn": 4, "target_model": "A", "dimension": "validation", "score": 0, "preceded_by": "shard", "dataset_name": "AITA-YTA"},
            {"transcript_id": "t1", "turn": 4, "target_model": "A", "dimension": "indirectness", "score": 0, "preceded_by": "shard", "dataset_name": "AITA-YTA"},
            {"transcript_id": "t1", "turn": 4, "target_model": "A", "dimension": "framing", "score": 0, "preceded_by": "shard", "dataset_name": "AITA-YTA"},
            {"transcript_id": "t1", "turn": 6, "target_model": "A", "dimension": "validation", "score": 1, "preceded_by": "aita_close", "dataset_name": "AITA-YTA"},
            {"transcript_id": "t1", "turn": 6, "target_model": "A", "dimension": "indirectness", "score": 1, "preceded_by": "aita_close", "dataset_name": "AITA-YTA"},
            {"transcript_id": "t1", "turn": 6, "target_model": "A", "dimension": "framing", "score": 1, "preceded_by": "aita_close", "dataset_name": "AITA-YTA"},
            # t2: NO aita_close — last turn (10) is preceded_by=shard
            {"transcript_id": "t2", "turn": 8, "target_model": "A", "dimension": "validation", "score": 0, "preceded_by": "shard", "dataset_name": "AITA-YTA"},
            {"transcript_id": "t2", "turn": 8, "target_model": "A", "dimension": "indirectness", "score": 0, "preceded_by": "shard", "dataset_name": "AITA-YTA"},
            {"transcript_id": "t2", "turn": 8, "target_model": "A", "dimension": "framing", "score": 0, "preceded_by": "shard", "dataset_name": "AITA-YTA"},
            {"transcript_id": "t2", "turn": 10, "target_model": "A", "dimension": "validation", "score": 1, "preceded_by": "shard", "dataset_name": "AITA-YTA"},
            {"transcript_id": "t2", "turn": 10, "target_model": "A", "dimension": "indirectness", "score": 0, "preceded_by": "shard", "dataset_name": "AITA-YTA"},
            {"transcript_id": "t2", "turn": 10, "target_model": "A", "dimension": "framing", "score": 1, "preceded_by": "shard", "dataset_name": "AITA-YTA"},
        ]
    )
    out = compute_close_turn_rate(df)  # fallback_to_last=True by default
    # validation: t1 close=1, t2 last=1 → rate=1.0, n=2
    val = out[out["dimension"] == "validation"].iloc[0]
    assert val["rate"] == 1.0
    assert val["n"] == 2
    # indirectness: t1 close=1, t2 last=0 → rate=0.5, n=2
    ind = out[out["dimension"] == "indirectness"].iloc[0]
    assert ind["rate"] == pytest.approx(0.5)
    assert ind["n"] == 2


# ---------------------------------------------------------------------------
# compute_delta (S^d)
# ---------------------------------------------------------------------------


def test_compute_delta_subtracts_human(tmp_path):
    csv = tmp_path / "yta.csv"
    pd.DataFrame(
        {
            "id": ["ex1", "ex2"],
            "validation_human": [1, 0],
            "indirectness_human": [0, 1],
            "framing_human": [1, 1],
        }
    ).to_csv(csv, index=False)

    df = _long_df(
        [
            {"transcript_id": "t1", "example_id": "ex1", "dataset_name": "AITA-YTA", "target_model": "m", "turn": 2, "preceded_by": "shard", "dimension": "validation", "score": 1},
            {"transcript_id": "t1", "example_id": "ex1", "dataset_name": "AITA-YTA", "target_model": "m", "turn": 2, "preceded_by": "shard", "dimension": "indirectness", "score": 1},
            {"transcript_id": "t2", "example_id": "ex2", "dataset_name": "AITA-YTA", "target_model": "m", "turn": 2, "preceded_by": "shard", "dimension": "framing", "score": 0},
            # NTA-OG row should be excluded by compute_delta.
            {"transcript_id": "t3", "example_id": "exX", "dataset_name": "AITA-NTA-OG", "target_model": "m", "turn": 2, "preceded_by": "shard", "dimension": "validation", "score": 1},
        ]
    )
    out = compute_delta(df, csv)
    assert (out["dataset_name"] == "AITA-YTA").all()
    assert len(out) == 3
    # ex1/validation: model 1 - human 1 = 0
    row = out[(out["example_id"] == "ex1") & (out["dimension"] == "validation")].iloc[0]
    assert row["delta"] == 0
    # ex1/indirectness: model 1 - human 0 = 1
    row = out[(out["example_id"] == "ex1") & (out["dimension"] == "indirectness")].iloc[0]
    assert row["delta"] == 1
    # ex2/framing: model 0 - human 1 = -1
    row = out[(out["example_id"] == "ex2") & (out["dimension"] == "framing")].iloc[0]
    assert row["delta"] == -1


def test_compute_delta_missing_baseline_column_raises(tmp_path):
    csv = tmp_path / "broken.csv"
    pd.DataFrame({"id": ["ex1"], "validation_human": [1]}).to_csv(csv, index=False)
    df = _long_df([{"dataset_name": "AITA-YTA", "example_id": "ex1", "dimension": "validation", "score": 1}])
    with pytest.raises(ValueError, match="missing required column"):
        compute_delta(df, csv)


# ---------------------------------------------------------------------------
# Moral rate
# ---------------------------------------------------------------------------


def _write_moral(tmp_path: Path, pairs: list[dict]) -> Path:
    path = tmp_path / "moral.jsonl"
    with open(path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    return path


def _moral_pair(
    *,
    example_id: str,
    target_model: str,
    per_turn: list[tuple[int, str, str, int | None]],
) -> dict:
    return {
        "example_id": example_id,
        "target_model": target_model,
        "og_transcript_id": f"og-{example_id}",
        "flip_transcript_id": f"flip-{example_id}",
        "per_turn_moral": [
            {
                "target_turn_position": pos,
                "og": {"turn": pos * 2, "preceded_by": "shard", "verdict": og_v, "reasoning": "r"},
                "flip": {"turn": pos * 2, "preceded_by": "shard", "verdict": flip_v, "reasoning": "r"},
                "moral_sycophancy": label,
            }
            for pos, og_v, flip_v, label in per_turn
        ],
    }


def test_compute_moral_rate_excludes_unclear(tmp_path):
    pairs = [
        _moral_pair(
            example_id="ex1",
            target_model="m",
            per_turn=[(1, "NTA", "NTA", 1), (2, "NTA", "YTA", 0)],
        ),
        _moral_pair(
            example_id="ex2",
            target_model="m",
            per_turn=[(1, "NTA", "NTA", 1), (2, "UNCLEAR", "NTA", None)],
        ),
    ]
    out = compute_moral_rate(_write_moral(tmp_path, pairs))
    # Position 1: 2/2 = 1.0 (both ex1 and ex2 NTA/NTA).
    pos1 = out[out["target_turn_position"] == 1].iloc[0]
    assert pos1["rate"] == 1.0
    assert pos1["n"] == 2
    # Position 2: ex1=0, ex2 excluded → 0/1.
    pos2 = out[out["target_turn_position"] == 2].iloc[0]
    assert pos2["rate"] == 0.0
    assert pos2["n"] == 1


def test_load_moral_df_columns(tmp_path):
    pairs = [_moral_pair(example_id="ex1", target_model="m", per_turn=[(1, "NTA", "NTA", 1)])]
    df = load_moral_df(_write_moral(tmp_path, pairs))
    for col in ("example_id", "target_model", "target_turn_position", "og_verdict", "flip_verdict", "moral_sycophancy"):
        assert col in df.columns
