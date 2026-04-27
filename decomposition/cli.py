from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import argparse
import json
import sys

from .anthropic_io import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_MAX_TOKENS,
    AnthropicRunError,
    call_anthropic,
    create_anthropic_client,
    raise_for_incomplete_response,
)
from .datasets import DatasetError, load_dataset
from .deterministic import SegmentationConfig, deterministic_segment, utc_now
from .llm_io import LLMIngestError, build_request, load_requests, read_jsonl, record_from_response
from .schema import ValidationError, validate_record_dict


ROOT_DESCRIPTION = """Turn AITA CSV rows into validated decomposition artifacts.

The CLI has two production paths:
  1. deterministic: local baseline segmentation, no model calls.
  2. anthropic: direct native Anthropic segmentation.

The lower-level generate-requests + ingest-responses path remains available for
batch runs, replay, and provider-neutral debugging.

Every emitted shard record is validated before it enters shards.jsonl. Rejected
rows go to run_errors.jsonl so a batch can keep moving without hiding failures.
"""

ROOT_EPILOG = """Examples:
  python -m decomposition.cli deterministic --dataset datasets/AITA-NTA-OG.csv \
--dataset-name AITA-NTA-OG --source-field original_post --run-id pilot \
--out decomposition/artifacts/shards.jsonl \
--errors decomposition/artifacts/run_errors.jsonl

  python -m decomposition.cli validate \
--input decomposition/artifacts/shards.jsonl

Tip:
  From the repository root, prefer:
    uv run --project decomposition python -m decomposition.cli <command> ...
"""


class HelpfulArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that points developers at the next useful command."""

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(
            2,
            "error: %s\n\nFix: run `%s --help` for required arguments and examples.\n"
            % (message, self.prog),
        )


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Run the decomposition CLI and return a process-style exit code."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (
        AnthropicRunError,
        DatasetError,
        FileNotFoundError,
        LLMIngestError,
        ValidationError,
        OSError,
    ) as exc:
        print("error: %s" % exc, file=sys.stderr)
        hint = _hint_for_exception(exc)
        if hint:
            print("Fix: %s" % hint, file=sys.stderr)
        return 1


def deterministic_command(args: argparse.Namespace) -> int:
    """Write validated deterministic baseline shard records for a dataset."""
    config = SegmentationConfig(
        run_id=args.run_id,
        segmentation_version=args.segmentation_version,
        created_at=args.created_at,
        max_source_tokens=args.max_source_tokens,
        max_shard_tokens=args.max_shard_tokens,
    )
    written = 0
    errors = 0
    _ensure_parent(args.out)
    _ensure_parent(args.errors)
    with args.out.open("w", encoding="utf-8") as out_handle, args.errors.open("w", encoding="utf-8") as err_handle:
        for example in load_dataset(args.dataset, args.dataset_name, args.source_field, args.limit):
            try:
                record = deterministic_segment(example, config)
                _write_jsonl(out_handle, record.to_dict())
                written += 1
            except Exception as exc:  # noqa: BLE001 - batch tool logs row-level failures.
                _write_jsonl(
                    err_handle,
                    _run_error(example, args.run_id, args.segmentation_version, "deterministic", exc),
                )
                errors += 1
    print("written=%d errors=%d" % (written, errors))
    return 0


def generate_requests_command(args: argparse.Namespace) -> int:
    """Write provider-neutral JSONL requests for external model execution."""
    _ensure_parent(args.out)
    written = 0
    with args.out.open("w", encoding="utf-8") as out_handle:
        for example in load_dataset(args.dataset, args.dataset_name, args.source_field, args.limit):
            _write_jsonl(
                out_handle,
                build_request(
                    example,
                    run_id=args.run_id,
                    segmentation_version=args.segmentation_version,
                    created_at=args.created_at,
                ),
            )
            written += 1
    print("written=%d" % written)
    return 0


def anthropic_command(args: argparse.Namespace) -> int:
    """Call Anthropic directly and write validated shard records."""
    client = getattr(args, "_anthropic_client", None) or create_anthropic_client()
    _ensure_parent(args.out)
    _ensure_parent(args.errors)
    if args.raw_responses:
        _ensure_parent(args.raw_responses)
    written = 0
    errors = 0
    raw_handle = args.raw_responses.open("w", encoding="utf-8") if args.raw_responses else None
    try:
        with args.out.open("w", encoding="utf-8") as out_handle, args.errors.open(
            "w", encoding="utf-8"
        ) as err_handle:
            for example in load_dataset(args.dataset, args.dataset_name, args.source_field, args.limit):
                request: Optional[Dict[str, Any]] = None
                response: Optional[Dict[str, Any]] = None
                try:
                    request = build_request(
                        example,
                        run_id=args.run_id,
                        segmentation_version=args.segmentation_version,
                        created_at=args.created_at,
                    )
                    response = call_anthropic(
                        request,
                        model=args.model,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        client=client,
                        created_at=args.created_at,
                    )
                    if raw_handle:
                        _write_jsonl(raw_handle, response)
                    raise_for_incomplete_response(response)
                    record = record_from_response(request, response)
                    _write_jsonl(out_handle, record.to_dict())
                    written += 1
                except Exception as exc:  # noqa: BLE001 - batch tool logs row-level failures.
                    error = _run_error(example, args.run_id, args.segmentation_version, "anthropic", exc)
                    if request:
                        error["request_id"] = request.get("request_id")
                    if response:
                        error["stop_reason"] = response.get("stop_reason")
                        error["provider_message_id"] = response.get("provider_message_id")
                    _write_jsonl(err_handle, error)
                    errors += 1
    finally:
        if raw_handle:
            raw_handle.close()
    print("written=%d errors=%d" % (written, errors))
    return 0


def ingest_responses_command(args: argparse.Namespace) -> int:
    """Convert provider-neutral model responses into validated shard records."""
    requests = load_requests(args.requests)
    _ensure_parent(args.out)
    _ensure_parent(args.errors)
    written = 0
    errors = 0
    with args.out.open("w", encoding="utf-8") as out_handle, args.errors.open("w", encoding="utf-8") as err_handle:
        for response in read_jsonl(args.responses):
            request_id = response.get("request_id")
            if not request_id or str(request_id) not in requests:
                _write_jsonl(
                    err_handle,
                    _response_error(
                        response,
                        "ingest",
                        "unknown request_id: %r" % request_id,
                        fix="Use the --requests file that generated this response batch, or regenerate responses.",
                    ),
                )
                errors += 1
                continue
            request = requests[str(request_id)]
            try:
                record = record_from_response(request, response)
                _write_jsonl(out_handle, record.to_dict())
                written += 1
            except Exception as exc:  # noqa: BLE001 - batch tool logs row-level failures.
                error = _response_error(
                    response,
                    "ingest",
                    str(exc),
                    error_type=exc.__class__.__name__,
                    fix=_hint_for_exception(exc),
                )
                error.update(
                    {
                        "example_id": request.get("example_id"),
                        "dataset_name": request.get("dataset_name"),
                        "source_text_field": request.get("source_text_field"),
                        "run_id": request.get("run_id"),
                        "segmentation_version": request.get("segmentation_version"),
                    }
                )
                _write_jsonl(err_handle, error)
                errors += 1
    print("written=%d errors=%d" % (written, errors))
    return 0


def validate_command(args: argparse.Namespace) -> int:
    """Validate an existing shards.jsonl artifact without rewriting it."""
    count = 0
    for record in read_jsonl(args.input):
        validate_record_dict(record)
        count += 1
    print("valid=%d" % count)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = HelpfulArgumentParser(
        prog="python -m decomposition.cli",
        description=ROOT_DESCRIPTION,
        epilog=ROOT_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        dest="command",
        metavar="COMMAND",
        required=True,
        parser_class=HelpfulArgumentParser,
    )

    deterministic_parser = subparsers.add_parser(
        "deterministic",
        help="segment a CSV locally into shards.jsonl",
        description="Run the local deterministic baseline and validate each emitted record.",
        epilog="""Example:
  python -m decomposition.cli deterministic --dataset datasets/AITA-NTA-OG.csv \
--dataset-name AITA-NTA-OG --source-field original_post --run-id pilot \
--out decomposition/artifacts/shards.deterministic.jsonl \
--errors decomposition/artifacts/run_errors.deterministic.jsonl
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_dataset_args(deterministic_parser)
    deterministic_parser.add_argument("--out", type=Path, required=True, help="destination shards.jsonl path")
    deterministic_parser.add_argument("--errors", type=Path, required=True, help="destination run_errors.jsonl path")
    deterministic_parser.add_argument(
        "--max-source-tokens",
        type=int,
        default=2800,
        help="estimated token budget above which a row is marked ineligible",
    )
    deterministic_parser.add_argument(
        "--max-shard-tokens",
        type=int,
        default=900,
        help="estimated token budget above which a deterministic shard is rejected",
    )
    deterministic_parser.set_defaults(func=deterministic_command)

    request_parser = subparsers.add_parser(
        "generate-requests",
        help="write provider-neutral LLM segmentation requests",
        description="Create JSONL requests that any model runner can execute outside this package.",
        epilog="""Example:
  python -m decomposition.cli generate-requests --dataset datasets/AITA-NTA-OG.csv \
--dataset-name AITA-NTA-OG --source-field original_post --run-id pilot \
--out decomposition/artifacts/seg_v1_requests.jsonl
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_dataset_args(request_parser)
    request_parser.add_argument("--out", type=Path, required=True, help="destination request JSONL path")
    request_parser.set_defaults(func=generate_requests_command)

    anthropic_parser = subparsers.add_parser(
        "anthropic",
        help="call Anthropic and write validated shards.jsonl",
        description="Run native Anthropic segmentation, align verbatim spans locally, and validate each record.",
        epilog="""Example:
  python -m decomposition.cli anthropic --dataset datasets/AITA-NTA-OG.csv \
--dataset-name AITA-NTA-OG --source-field original_post --run-id pilot \
--model claude-sonnet-4-5 \
--out decomposition/artifacts/shards.anthropic.jsonl \
--errors decomposition/artifacts/run_errors.anthropic.jsonl \
--raw-responses decomposition/artifacts/seg_v1_anthropic_responses.jsonl
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_dataset_args(anthropic_parser)
    anthropic_parser.add_argument("--out", type=Path, required=True, help="destination shards.jsonl path")
    anthropic_parser.add_argument("--errors", type=Path, required=True, help="destination run_errors.jsonl path")
    anthropic_parser.add_argument(
        "--raw-responses",
        type=Path,
        default=None,
        help="optional replay/audit JSONL containing raw Anthropic response wrappers",
    )
    anthropic_parser.add_argument("--model", default=DEFAULT_ANTHROPIC_MODEL, help="Anthropic model name")
    anthropic_parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="maximum output tokens for each Anthropic response",
    )
    anthropic_parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="sampling temperature for Anthropic generation",
    )
    anthropic_parser.set_defaults(func=anthropic_command)

    ingest_parser = subparsers.add_parser(
        "ingest-responses",
        help="align and validate provider-neutral model responses",
        description="Read model output JSONL, compute local source spans, and reject invalid rows.",
        epilog="""Example:
  python -m decomposition.cli ingest-responses \
--requests decomposition/artifacts/seg_v1_requests.jsonl \
--responses decomposition/artifacts/seg_v1_responses.jsonl \
--out decomposition/artifacts/shards.llm.jsonl \
--errors decomposition/artifacts/run_errors.llm.jsonl
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ingest_parser.add_argument("--requests", type=Path, required=True, help="request JSONL produced by generate-requests")
    ingest_parser.add_argument("--responses", type=Path, required=True, help="provider-neutral response JSONL")
    ingest_parser.add_argument("--out", type=Path, required=True, help="destination shards.jsonl path")
    ingest_parser.add_argument("--errors", type=Path, required=True, help="destination run_errors.jsonl path")
    ingest_parser.set_defaults(func=ingest_responses_command)

    validate_parser = subparsers.add_parser(
        "validate",
        help="check a shard artifact against the decomposition contract",
        description="Validate a shards.jsonl file and exit non-zero on the first invalid record.",
        epilog="""Example:
  python -m decomposition.cli validate --input decomposition/artifacts/shards.llm.jsonl
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    validate_parser.add_argument("--input", type=Path, required=True, help="shards.jsonl file to validate")
    validate_parser.set_defaults(func=validate_command)
    return parser


def _add_dataset_args(parser: argparse.ArgumentParser) -> None:
    """Attach common dataset/run arguments to a subcommand parser."""
    parser.add_argument("--dataset", type=Path, required=True, help="input AITA CSV path")
    parser.add_argument("--dataset-name", required=True, help="dataset provenance label, e.g. AITA-NTA-OG")
    parser.add_argument("--source-field", required=True, help="CSV column containing the raw post text")
    parser.add_argument("--run-id", required=True, help="stable identifier for this segmentation run")
    parser.add_argument("--segmentation-version", default="seg_v1", help="prompt/schema version to use")
    parser.add_argument("--created-at", default=None, help="override artifact timestamp for reproducible tests")
    parser.add_argument("--limit", type=int, default=None, help="optional maximum number of rows to process")


def _ensure_parent(path: Path) -> None:
    """Create the parent directory for an output path if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_jsonl(handle: Any, data: Dict[str, Any]) -> None:
    """Write one deterministic JSON object as a JSONL row."""
    handle.write(json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n")


def _run_error(example: Any, run_id: str, segmentation_version: str, stage: str, exc: Exception) -> Dict[str, Any]:
    """Build a row-level sidecar error without losing provenance."""
    error = {
        "example_id": getattr(example, "example_id", None),
        "dataset_name": getattr(example, "dataset_name", None),
        "source_text_field": getattr(example, "source_text_field", None),
        "run_id": run_id,
        "segmentation_version": segmentation_version,
        "stage": stage,
        "error_type": exc.__class__.__name__,
        "message": str(exc),
        "created_at": utc_now(),
    }
    hint = _hint_for_exception(exc)
    if hint:
        error["fix"] = hint
    return error


def _response_error(
    response: Dict[str, Any],
    stage: str,
    message: str,
    error_type: str = "LLMIngestError",
    fix: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a provider-response sidecar error with request correlation."""
    error = {
        "request_id": response.get("request_id"),
        "stage": stage,
        "error_type": error_type,
        "message": message,
        "created_at": utc_now(),
    }
    if fix:
        error["fix"] = fix
    return error


def _hint_for_exception(exc: Exception) -> Optional[str]:
    """Return a short remediation hint for common developer-facing errors."""
    if isinstance(exc, DatasetError):
        return "Check that --dataset points to the intended CSV and --source-field matches one header exactly."
    if isinstance(exc, AnthropicRunError):
        if "ANTHROPIC_API_KEY" in str(exc):
            return "Export ANTHROPIC_API_KEY before running the anthropic command."
        if "anthropic package is not installed" in str(exc):
            return "Run `uv sync --project decomposition` to install decomposition dependencies."
        if "max_tokens" in str(exc):
            return "Increase --max-tokens and retry this row."
        return "Check ANTHROPIC_API_KEY, --model, --max-tokens, and the raw response sidecar for the failed row."
    if isinstance(exc, FileNotFoundError):
        return "Check the path exists. For prompts, confirm --segmentation-version has a matching prompts/<version>.txt file."
    if isinstance(exc, LLMIngestError):
        return "Ensure each response has request_id, atomic_units, shards, and verbatim source text."
    if isinstance(exc, ValidationError):
        return "Run the validate command on the artifact and inspect the listed field-level contract failures."
    if "text not found after offset" in str(exc):
        return "Make the model return exact verbatim atomic-unit text in source order; local code computes offsets."
    if "source text is empty" in str(exc):
        return "Drop the empty row or choose a --source-field that contains the raw post text."
    return None


if __name__ == "__main__":
    raise SystemExit(main())
