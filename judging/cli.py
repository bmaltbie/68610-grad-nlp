"""Command-line entry for the judging stage.

Subcommands:
- ``score``     — run the LLM judge over transcripts, write
                  ``judge.jsonl`` index + per-trace JSON files.
- ``moral``     — second-pass: join NTA-OG and NTA-FLIP traces by
                  ``example_id`` and write ``moral.jsonl``.
- ``aggregate`` — compute per-turn rates / S^d deltas / close-turn rate
                  / moral rate from the index outputs.
- ``plot``      — generate accumulation curves / cross-dataset
                  comparisons.
- ``calibrate`` — single-turn calibration vs ELEPHANT Table 3.

Steps 2–5 of TODO.md fill in the stubs. Step 1 just wires the surface
area so the package is callable end-to-end via ``python -m judging``.
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence


def _add_score_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input", required=True, help="path to conversation_transcripts.jsonl")
    p.add_argument("--output-dir", default="outputs/judging", help="where to write judge.jsonl + traces/")
    p.add_argument("--judge-model", default="gpt-4o", help="OpenAI model for the judge")
    p.add_argument("--concurrency", type=int, default=8, help="max concurrent API calls")
    p.add_argument("--batch", action="store_true", help="route through OpenAI Batch API")
    p.add_argument("--resume", action="store_true", help="skip transcripts whose trace file already exists")
    p.add_argument("--max", type=int, default=None, help="cap number of transcripts (smoke test)")
    p.add_argument("--single-turn", action="store_true", help="bypass multi-turn formatter (calibration mode)")


def _add_moral_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--judge-jsonl", required=True, help="path to judge.jsonl from `score`")
    p.add_argument("--output", default="outputs/judging/moral.jsonl", help="output moral.jsonl path")


def _add_aggregate_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--judge-jsonl", required=True)
    p.add_argument(
        "--kind",
        choices=["rate", "delta", "moral", "close"],
        required=True,
        help="rate=per-turn rates; delta=S^d vs human (AITA-YTA); moral=moral rate; close=close-turn rate (vs single-turn baseline)",
    )
    p.add_argument("--baselines-csv", default=None, help="AITA-YTA CSV with *_human columns (delta only)")
    p.add_argument("--moral-jsonl", default=None, help="path to moral.jsonl (kind=moral only)")
    p.add_argument("--output", default=None, help="optional output CSV path")


def _add_plot_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--judge-jsonl", required=True)
    p.add_argument(
        "--kind",
        choices=["accumulation", "cross_dataset"],
        required=True,
    )
    p.add_argument("--output-dir", default="outputs/judging/plots")


def _add_calibrate_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--n", type=int, default=30, help="number of records to sample")
    p.add_argument(
        "--dataset", choices=["AITA-YTA"], default="AITA-YTA",
        help="only AITA-YTA carries human baselines locally",
    )
    p.add_argument("--csv", default="datasets/AITA-YTA.csv", help="path to source AITA-YTA CSV")
    p.add_argument("--judge-model", default="gpt-4o")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m judging",
        description="Judging stage of the multi-turn ELEPHANT pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    _add_score_args(sub.add_parser("score", help="run LLM judge over transcripts"))
    _add_moral_args(sub.add_parser("moral", help="join NTA-OG ⨝ NTA-FLIP traces"))
    _add_aggregate_args(sub.add_parser("aggregate", help="compute rates / deltas / moral / drift"))
    _add_plot_args(sub.add_parser("plot", help="generate plots from judge.jsonl"))
    _add_calibrate_args(sub.add_parser("calibrate", help="single-turn calibration vs ELEPHANT Table 3"))

    return parser


def _stub(command: str) -> None:
    raise NotImplementedError(
        f"`{command}` is wired but not yet implemented; see judging/TODO.md"
    )


def _cmd_score(args) -> None:
    import asyncio

    from judging.runner import run_score

    if args.single_turn:
        # Single-turn mode is owned by the calibrate subcommand;
        # forbid it on `score` to keep semantics clean.
        raise SystemExit(
            "--single-turn is only valid on `calibrate`, not `score`"
        )
    asyncio.run(
        run_score(
            input_path=args.input,
            output_dir=args.output_dir,
            judge_model=args.judge_model,
            concurrency=args.concurrency,
            resume=args.resume,
            max_count=args.max,
            batch=args.batch,
        )
    )


def _cmd_moral(args) -> None:
    from judging.runner import run_moral

    n = run_moral(judge_jsonl=args.judge_jsonl, output_path=args.output)
    print(f"wrote {n} pairs to {args.output}")


def _cmd_aggregate(args) -> None:
    from judging.aggregate import (
        compute_close_turn_rate,
        compute_delta,
        compute_moral_rate,
        compute_rate,
        load_index_df,
    )

    if args.kind == "moral":
        if not args.moral_jsonl:
            raise SystemExit("--moral-jsonl is required for --kind moral")
        out = compute_moral_rate(args.moral_jsonl)
    elif args.kind == "delta":
        if not args.baselines_csv:
            raise SystemExit("--baselines-csv is required for --kind delta")
        out = compute_delta(load_index_df(args.judge_jsonl), args.baselines_csv)
    elif args.kind == "close":
        out = compute_close_turn_rate(load_index_df(args.judge_jsonl))
    else:  # rate
        out = compute_rate(
            load_index_df(args.judge_jsonl),
            by=["dataset_name", "target_model", "turn", "dimension"],
        )

    if args.output:
        out.to_csv(args.output, index=False)
        print(f"wrote {len(out)} rows to {args.output}")
    else:
        print(out.to_string(index=False))


def _cmd_plot(args) -> None:
    from judging.analysis import plot_accumulation_curves, plot_cross_dataset

    if args.kind == "accumulation":
        path = plot_accumulation_curves(args.judge_jsonl, args.output_dir)
    else:  # cross_dataset
        path = plot_cross_dataset(args.judge_jsonl, args.output_dir)
    print(f"wrote {path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "score":
        _cmd_score(args)
    elif args.command == "moral":
        _cmd_moral(args)
    elif args.command == "aggregate":
        _cmd_aggregate(args)
    elif args.command == "plot":
        _cmd_plot(args)
    elif args.command == "calibrate":
        _stub("calibrate")
    else:
        parser.error(f"unknown command {args.command!r}")
    return 0
