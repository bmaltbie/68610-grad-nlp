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


@pytest.mark.parametrize(
    "argv,expected",
    [
        (["score", "--input", "x.jsonl"], "score"),
        (["moral", "--judge-jsonl", "x.jsonl"], "moral"),
        (["aggregate", "--judge-jsonl", "x.jsonl", "--kind", "rate"], "aggregate"),
        (["plot", "--judge-jsonl", "x.jsonl", "--kind", "accumulation"], "plot"),
        (["calibrate"], "calibrate"),
    ],
)
def test_subcommands_parse_then_stub(argv, expected):
    """Each subcommand parses successfully and reaches the NotImplemented stub."""
    with pytest.raises(NotImplementedError, match=expected):
        main(argv)


def test_aggregate_kind_choices_enforced():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["aggregate", "--judge-jsonl", "x.jsonl", "--kind", "bogus"])


def test_score_dry_run_flag_present():
    parser = build_parser()
    args = parser.parse_args(["score", "--input", "x.jsonl", "--dry-run"])
    assert args.dry_run is True
    assert args.command == "score"
