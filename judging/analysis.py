"""Plot generators for the judging stage.

Two plot families:
- ``plot_accumulation_curves`` — per-turn rate trajectory; one panel
  per dimension, one line per ``target_model``, with 95% CI band.
- ``plot_cross_dataset`` — per-turn trajectories side-by-side across
  AITA-YTA / AITA-NTA-OG / AITA-NTA-FLIP.

Both consume ``judge.jsonl`` produced by the runner.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; safe for tests + CI
import matplotlib.pyplot as plt
import seaborn as sns

from judging.aggregate import DIMENSION_COLS, load_index_df

log = logging.getLogger(__name__)


def _save(fig, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_accumulation_curves(
    judge_jsonl: Path | str, output_dir: Path | str
) -> Path:
    """Per-turn rate trajectory, one panel per dimension.

    seaborn's ``lineplot`` with ``errorbar='ci'`` computes the
    bootstrapped 95% CI band per (turn, target_model) cell.
    """
    df = load_index_df(judge_jsonl)
    if df.empty:
        raise ValueError(f"no rows in {judge_jsonl}")
    output_dir = Path(output_dir)

    fig, axes = plt.subplots(1, len(DIMENSION_COLS), figsize=(15, 4), sharey=True)
    for ax, dim in zip(axes, DIMENSION_COLS):
        sub = df[df["dimension"] == dim]
        sns.lineplot(
            data=sub,
            x="turn",
            y="score",
            hue="target_model",
            errorbar=("ci", 95),
            ax=ax,
        )
        ax.set_title(dim)
        ax.set_xlabel("target turn")
        ax.set_ylabel("sycophancy rate")
        ax.set_ylim(0, 1)
    fig.suptitle("ELEPHANT sycophancy rate over multi-turn trajectory")
    fig.tight_layout()
    return _save(fig, output_dir, "accumulation_curves.png")


def plot_cross_dataset(
    judge_jsonl: Path | str, output_dir: Path | str
) -> Path:
    """Cross-dataset comparison: one panel per dimension, one line per dataset."""
    df = load_index_df(judge_jsonl)
    if df.empty:
        raise ValueError(f"no rows in {judge_jsonl}")
    output_dir = Path(output_dir)

    fig, axes = plt.subplots(1, len(DIMENSION_COLS), figsize=(15, 4), sharey=True)
    for ax, dim in zip(axes, DIMENSION_COLS):
        sub = df[df["dimension"] == dim]
        sns.lineplot(
            data=sub,
            x="turn",
            y="score",
            hue="dataset_name",
            errorbar=("ci", 95),
            ax=ax,
        )
        ax.set_title(dim)
        ax.set_xlabel("target turn")
        ax.set_ylabel("sycophancy rate")
        ax.set_ylim(0, 1)
    fig.suptitle("Sycophancy rate by dataset across the multi-turn trajectory")
    fig.tight_layout()
    return _save(fig, output_dir, "cross_dataset.png")


__all__ = ["plot_accumulation_curves", "plot_cross_dataset"]
