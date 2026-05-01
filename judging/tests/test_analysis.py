"""Smoke tests for plotting.

Each plotter takes a judge.jsonl path and an output dir, writes a PNG.
We synthesise a tiny but multi-row index file so seaborn has enough
points to draw meaningful lines, then assert the PNG exists and is
non-empty.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from judging.analysis import plot_accumulation_curves, plot_cross_dataset


def _make_index(tmp_path: Path) -> Path:
    """Synthesise judge.jsonl with 2 datasets × 2 models × 3 turns."""
    records = []
    for dataset in ("AITA-YTA", "AITA-NTA-OG"):
        for model in ("model-A", "model-B"):
            for tr_idx in range(3):
                records.append(
                    {
                        "transcript_id": f"{dataset}-{model}-{tr_idx}",
                        "example_id": f"ex{tr_idx}",
                        "dataset_name": dataset,
                        "target_model": model,
                        "judge_model": "gpt-4o-test",
                        "rubric_version": "v1",
                        "trace_path": "stub",
                        "per_turn_summary": [
                            {"turn": 2, "preceded_by": "shard", "validation": 1, "indirectness": 0, "framing": 1},
                            {"turn": 4, "preceded_by": "shard", "validation": tr_idx % 2, "indirectness": 1, "framing": 0},
                            {"turn": 6, "preceded_by": "aita_close", "validation": 1, "indirectness": 1, "framing": 1},
                        ],
                    }
                )
    path = tmp_path / "judge.jsonl"
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


def test_plot_accumulation_curves_writes_png(tmp_path):
    judge_jsonl = _make_index(tmp_path)
    out = plot_accumulation_curves(judge_jsonl, tmp_path / "plots")
    assert out.exists()
    assert out.suffix == ".png"
    assert out.stat().st_size > 1000  # non-trivial PNG


def test_plot_cross_dataset_writes_png(tmp_path):
    judge_jsonl = _make_index(tmp_path)
    out = plot_cross_dataset(judge_jsonl, tmp_path / "plots")
    assert out.exists()
    assert out.suffix == ".png"
    assert out.stat().st_size > 1000


def test_plot_empty_jsonl_raises(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(ValueError, match="no rows"):
        plot_accumulation_curves(empty, tmp_path / "plots")
