"""Tests for the judging CLI surface.

Step 1 commits the wiring; subcommands raise NotImplementedError.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from judging.cli import build_parser, main


def test_help_runs():
    """`python -m judging -h` exits 0 and lists every subcommand."""
    result = subprocess.run(
        [sys.executable, "-m", "judging", "-h"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    for cmd in ("score", "moral", "aggregate", "plot", "calibrate"):
        assert cmd in out


def test_no_subcommand_errors_cleanly():
    """Bare `python -m judging` exits non-zero with a usage message."""
    result = subprocess.run(
        [sys.executable, "-m", "judging"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "usage" in (result.stdout + result.stderr).lower()


def test_score_missing_required_arg_errors():
    """`score` without --input is rejected by argparse, not the stub."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["score"])


def test_calibrate_on_missing_csv_errors_clean(tmp_path):
    """`calibrate` reaches the runner; missing CSV → FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        main(["calibrate", "--csv", str(tmp_path / "missing.csv")])


def test_aggregate_on_missing_input_errors_clean(tmp_path):
    with pytest.raises(FileNotFoundError):
        main([
            "aggregate",
            "--judge-jsonl", str(tmp_path / "missing.jsonl"),
            "--kind", "rate",
        ])


def test_plot_on_missing_input_errors_clean(tmp_path):
    with pytest.raises(FileNotFoundError):
        main([
            "plot",
            "--judge-jsonl", str(tmp_path / "missing.jsonl"),
            "--kind", "accumulation",
            "--output-dir", str(tmp_path / "plots"),
        ])


def test_score_on_missing_input_errors_clean(tmp_path):
    """`score` on a nonexistent input fails fast with FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        main([
            "score",
            "--input", str(tmp_path / "missing.jsonl"),
            "--output-dir", str(tmp_path / "out"),
        ])


def test_score_rejects_single_turn_flag(tmp_path):
    """--single-turn belongs on `calibrate`; rejected by `score`."""
    with pytest.raises(SystemExit):
        main([
            "score",
            "--input", str(tmp_path / "x.jsonl"),
            "--single-turn",
        ])


def test_moral_on_missing_jsonl_errors_clean(tmp_path):
    with pytest.raises(FileNotFoundError):
        main([
            "moral",
            "--judge-jsonl", str(tmp_path / "missing-judge.jsonl"),
            "--output", str(tmp_path / "moral.jsonl"),
        ])


def test_aggregate_kind_choices_enforced():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["aggregate", "--judge-jsonl", "x.jsonl", "--kind", "bogus"])


def test_score_resume_flag_parses():
    parser = build_parser()
    args = parser.parse_args(["score", "--input", "x.jsonl", "--resume"])
    assert args.resume is True
    assert args.command == "score"
