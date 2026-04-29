from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import argparse
import csv
import hashlib
import json
import os
import sys
import threading

from .anthropic_io import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_MAX_TOKENS,
    AnthropicRunError,
    anthropic_message_kwargs,
    call_anthropic,
    create_anthropic_client,
    raise_for_incomplete_response as raise_for_incomplete_anthropic_response,
    response_from_anthropic_message,
)
from .align import AlignmentError
from .datasets import DatasetError, load_dataset
from .deterministic import SegmentationConfig, deterministic_segment, utc_now
from .llm_io import LLMIngestError, build_request, load_requests, read_jsonl, record_from_response
from .llm_schema import SEGMENTATION_OUTPUT_SCHEMA
from .openai_io import (
    DEFAULT_OPENAI_MODEL,
    OpenAIRunError,
    call_openai,
    create_openai_client,
    openai_batch_request,
    raise_for_incomplete_response as raise_for_incomplete_openai_response,
    response_from_openai_batch_result,
)
from .schema import ValidationError, validate_record_dict


ROOT_DESCRIPTION = """Turn AITA CSV rows into validated decomposition artifacts.

The CLI has two production paths:
  1. deterministic: local baseline segmentation, no model calls.
  2. llm: direct native LLM segmentation. OpenAI is the default provider.

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


SUPPORTED_LLM_PROVIDERS = ("openai", "anthropic")
DEFAULT_LLM_PROVIDER = "openai"


def _default_model_for_provider(provider: str) -> str:
    """Return the default model for a supported provider."""
    if provider == "openai":
        return DEFAULT_OPENAI_MODEL
    if provider == "anthropic":
        return DEFAULT_ANTHROPIC_MODEL
    raise ValueError("unsupported LLM provider: %s" % provider)


def _normalize_llm_args(args: argparse.Namespace, default_provider: str = DEFAULT_LLM_PROVIDER) -> argparse.Namespace:
    """Fill provider/model defaults once at the CLI boundary."""
    provider = str(getattr(args, "provider", None) or default_provider)
    if provider not in SUPPORTED_LLM_PROVIDERS:
        raise DatasetError("unsupported provider %r; expected one of %s" % (provider, ", ".join(SUPPORTED_LLM_PROVIDERS)))
    args.provider = provider
    if not getattr(args, "model", None):
        args.model = _default_model_for_provider(provider)
    if not getattr(args, "max_tokens", None):
        args.max_tokens = DEFAULT_MAX_TOKENS
    return args


def _create_llm_client(args: argparse.Namespace) -> Any:
    """Create the configured provider client unless a fake test client was injected."""
    provider = str(getattr(args, "provider", DEFAULT_LLM_PROVIDER))
    if provider == "openai":
        return getattr(args, "_openai_client", None) or create_openai_client(
            max_retries=getattr(args, "openai_max_retries", getattr(args, "provider_max_retries", 2))
        )
    if provider == "anthropic":
        return getattr(args, "_anthropic_client", None) or create_anthropic_client(
            max_retries=getattr(args, "anthropic_max_retries", getattr(args, "provider_max_retries", 2))
        )
    raise DatasetError("unsupported provider %r" % provider)


class LLMAttemptFailure(Exception):
    """Wrap the final per-row failure with retry metadata for error sidecars."""

    def __init__(
        self,
        original: Exception,
        response: Optional[Dict[str, Any]],
        attempts: int,
        raw_responses: int,
        retryable: bool,
        retry_errors: Any,
    ):
        self.original = original
        self.response = response
        self.attempts = attempts
        self.raw_responses = raw_responses
        self.retryable = retryable
        self.retry_errors = retry_errors
        super().__init__(str(original))


@dataclass
class CachedShardRow:
    """A reusable validated shard row plus where it came from."""

    row: Dict[str, Any]
    source: str


@dataclass
class ResumeStats:
    """Counters that explain what resume did and why rows were or were not reused."""

    cached_final: int = 0
    cached_tmp: int = 0
    cached_content: int = 0
    skipped_missing_fingerprint: int = 0
    skipped_invalid: int = 0


@dataclass
class ResumeCache:
    """Validated shard rows indexed by exact request and byte-identical content keys."""

    by_request: Dict[str, CachedShardRow] = field(default_factory=dict)
    by_content: Dict[str, CachedShardRow] = field(default_factory=dict)
    stats: ResumeStats = field(default_factory=ResumeStats)


@dataclass
class LLMWorkItem:
    """One dataset row after request construction and cache lookup."""

    index: int
    total: Optional[int]
    example: Any
    request: Dict[str, Any]
    request_fingerprint: str
    content_fingerprint: str
    cached: Optional[CachedShardRow] = None
    content_cache_hit: bool = False


@dataclass
class LLMWorkResult:
    """One completed row, either cached, generated, or failed."""

    item: LLMWorkItem
    status: str
    row: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    attempt_count: int = 0
    raw_responses: int = 0
    retries: int = 0
    error_type: Optional[str] = None


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Run the decomposition CLI and return a process-style exit code."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (
        AnthropicRunError,
        OpenAIRunError,
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


def llm_command(args: argparse.Namespace) -> int:
    """Call the configured LLM provider directly and write validated shard records."""
    _normalize_llm_args(args)
    llm_retries = max(0, int(getattr(args, "llm_retries", 1)))
    max_attempts = llm_retries + 1
    concurrency = max(1, int(getattr(args, "concurrency", 1)))
    progress_mode = _resolve_progress_mode(getattr(args, "progress", "auto"))
    total = _count_dataset_rows(args.dataset, args.source_field, args.limit) if progress_mode != "off" else None
    _ensure_parent(args.out)
    _ensure_parent(args.errors)
    if args.raw_responses:
        _ensure_parent(args.raw_responses)
    resume = bool(getattr(args, "resume", False))
    include_temp = bool(getattr(args, "resume_include_temp", True))
    resume_cache = getattr(args, "_resume_cache", None) if resume else None
    if resume_cache is None:
        resume_cache = _load_resume_cache(args.out, include_temp=include_temp) if resume else ResumeCache()
    items = _llm_work_items(args, resume_cache, total)
    has_cache_miss = any(item.cached is None for item in items)
    client = None
    if has_cache_miss:
        client = _create_llm_client(args)
    out_path = _resume_temp_path(args.out) if resume else args.out
    raw_mode = getattr(args, "raw_responses_mode", None) or ("append" if resume else "overwrite")
    written = 0
    generated = 0
    cached = 0
    errors = 0
    raw_responses = 0
    retries = 0
    error_types: Counter = Counter()
    progress = _progress_bar(progress_mode, total)
    raw_lock = threading.Lock()
    progress_lock = threading.Lock()
    raw_open_mode = "a" if raw_mode == "append" else "w"
    raw_handle = args.raw_responses.open(raw_open_mode, encoding="utf-8") if args.raw_responses else None
    try:
        with out_path.open("w", encoding="utf-8") as out_handle, args.errors.open(
            "w", encoding="utf-8"
        ) as err_handle:
            results = _run_llm_work_items(
                items,
                args=args,
                client=client,
                max_attempts=max_attempts,
                raw_handle=raw_handle,
                raw_lock=raw_lock,
                progress=progress,
                progress_mode=progress_mode,
                progress_lock=progress_lock,
                concurrency=concurrency,
            )
            for result in results:
                if result.row is not None:
                    _write_jsonl(out_handle, result.row)
                    written += 1
                    if result.status == "cached":
                        cached += 1
                        _count_cache_hit(resume_cache.stats, result.item.cached)
                        if result.item.content_cache_hit:
                            resume_cache.stats.cached_content += 1
                    else:
                        generated += 1
                if result.error is not None:
                    _write_jsonl(err_handle, result.error)
                    errors += 1
                    error_types[str(result.error["error_type"])] += 1
                raw_responses += result.raw_responses
                retries += result.retries
                if result.status != "retry":
                    _progress_update(
                        progress,
                        progress_mode,
                        result.item.index,
                        total,
                        result.item.example.example_id,
                        result.status,
                        written,
                        errors,
                        cached=cached,
                        retries=retries,
                    )
        if resume:
            os.replace(out_path, args.out)
    finally:
        _progress_close(progress)
        if raw_handle:
            raw_handle.close()
    _print_llm_summary(args, written, generated, cached, errors, raw_responses, retries, error_types, resume_cache.stats)
    return 0


def openai_command(args: argparse.Namespace) -> int:
    """Compatibility command for native OpenAI segmentation."""
    args.provider = "openai"
    return llm_command(args)


def anthropic_command(args: argparse.Namespace) -> int:
    """Compatibility command for native Anthropic segmentation."""
    args.provider = "anthropic"
    return llm_command(args)


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


DEFAULT_MANIFEST_SOURCE_FIELDS = {
    "AITA-NTA-OG": "original_post",
    "AITA-NTA-FLIP": "flipped_story",
    "AITA-YTA": "prompt",
    "OEQ": "prompt",
    "SS": "sentence",
}


def llm_bulk_command(args: argparse.Namespace) -> int:
    """Run the realtime LLM command for each JSONL manifest dataset."""
    _normalize_llm_args(args)
    rows = _load_llm_manifest(args.manifest)
    completed = 0
    for manifest_index, row in enumerate(rows, start=1):
        subargs = _manifest_dataset_args(args, row)
        if subargs.resume:
            subargs._resume_cache = _load_manifest_resume_cache(rows, args)
        result = llm_command(subargs)
        if result != 0:
            return result
        completed += 1
    print("datasets=%d" % completed)
    return 0


def openai_bulk_command(args: argparse.Namespace) -> int:
    """Compatibility command for OpenAI manifest segmentation."""
    args.provider = "openai"
    return llm_bulk_command(args)


def anthropic_bulk_command(args: argparse.Namespace) -> int:
    """Compatibility command for Anthropic manifest segmentation."""
    args.provider = "anthropic"
    return llm_bulk_command(args)


def anthropic_batch_submit_command(args: argparse.Namespace) -> int:
    """Submit cache misses from a manifest to Anthropic Message Batches."""
    args.provider = "anthropic"
    _normalize_llm_args(args, default_provider="anthropic")
    rows = _load_llm_manifest(args.manifest)
    resume_cache = _load_manifest_resume_cache(rows, args)
    batch_requests = []
    state_requests = []
    sequence = 0
    for manifest_index, row in enumerate(rows, start=1):
        subargs = _manifest_dataset_args(args, row)
        items = _llm_work_items(subargs, resume_cache, total=None)
        for item in items:
            if item.cached is not None:
                continue
            sequence += 1
            custom_id = _batch_custom_id(sequence, item.request_fingerprint)
            batch_requests.append(
                {
                    "custom_id": custom_id,
                    "params": anthropic_message_kwargs(
                        item.request,
                        model=subargs.model,
                        max_tokens=subargs.max_tokens,
                        temperature=subargs.temperature,
                    ),
                }
            )
            state_requests.append(
                {
                    "custom_id": custom_id,
                    "manifest_index": manifest_index,
                    "row_index": item.index,
                    "request": item.request,
                    "request_fingerprint": item.request_fingerprint,
                    "content_fingerprint": item.content_fingerprint,
                    "out": str(subargs.out),
                    "errors": str(subargs.errors),
                    "raw_responses": str(subargs.raw_responses) if subargs.raw_responses else None,
                }
            )
    _ensure_parent(args.batch_state)
    batch_id = None
    raw_batch = None
    if batch_requests:
        client = getattr(args, "_anthropic_client", None) or create_anthropic_client(
            max_retries=getattr(args, "anthropic_max_retries", 2)
        )
        batch = client.messages.batches.create(requests=batch_requests)
        raw_batch = _object_to_dict(batch)
        batch_id = str(raw_batch.get("id") or getattr(batch, "id", ""))
    state = {
        "type": "anthropic_batch_state",
        "provider": "anthropic",
        "created_at": utc_now(),
        "batch_id": batch_id,
        "batch": raw_batch,
        "manifest": rows,
        "request_count": len(state_requests),
        "requests": state_requests,
        "run_id": args.run_id,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "segmentation_version": args.segmentation_version,
    }
    args.batch_state.write_text(json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print("batch_id=%s requests=%d state=%s" % (batch_id or "none", len(state_requests), args.batch_state))
    return 0


def anthropic_batch_collect_command(args: argparse.Namespace) -> int:
    """Collect Anthropic Message Batch results and write validated artifacts."""
    state = json.loads(args.batch_state.read_text(encoding="utf-8"))
    args.provider = state.get("provider") or "anthropic"
    rows = state.get("manifest") or []
    args.run_id = state.get("run_id")
    args.segmentation_version = state.get("segmentation_version", "seg_v1")
    args.created_at = None
    args.limit = None
    args.model = state.get("model", DEFAULT_ANTHROPIC_MODEL)
    args.max_tokens = state.get("max_tokens", DEFAULT_MAX_TOKENS)
    args.temperature = state.get("temperature", 0.0)
    args.llm_retries = 0
    args.concurrency = 1
    args.progress = "off"
    args.resume = True
    batch_id = state.get("batch_id")
    batch_results = {}
    if batch_id:
        client = getattr(args, "_anthropic_client", None) or create_anthropic_client(
            max_retries=getattr(args, "anthropic_max_retries", 2)
        )
        batch = client.messages.batches.retrieve(str(batch_id))
        batch_dict = _object_to_dict(batch)
        status = str(batch_dict.get("processing_status") or getattr(batch, "processing_status", ""))
        if status and status != "ended":
            print("batch_id=%s status=%s results=not_ready" % (batch_id, status))
            return 0
        request_by_custom = {str(item["custom_id"]): item for item in state.get("requests", [])}
        for result in client.messages.batches.results(str(batch_id)):
            result_dict = _object_to_dict(result)
            custom_id = str(result_dict.get("custom_id") or getattr(result, "custom_id", ""))
            request_state = request_by_custom.get(custom_id)
            if request_state:
                batch_results[str(request_state["request_fingerprint"])] = _batch_result_for_request(
                    result,
                    result_dict,
                    request_state,
                    state,
                    str(batch_id),
                )
    summary = _write_batch_collect_outputs(args, rows, state, batch_results)
    print(
        "written=%d cached=%d errors=%d raw_responses=%d batch_id=%s"
        % (summary["written"], summary["cached"], summary["errors"], summary["raw_responses"], batch_id or "none")
    )
    return 0


def openai_batch_submit_command(args: argparse.Namespace) -> int:
    """Submit cache misses from a manifest to OpenAI Batch for /v1/responses."""
    args.provider = "openai"
    _normalize_llm_args(args)
    rows = _load_llm_manifest(args.manifest)
    resume_cache = _load_manifest_resume_cache(rows, args)
    input_path = getattr(args, "batch_input", None) or args.batch_state.with_name(args.batch_state.stem + ".input.jsonl")
    _ensure_parent(input_path)
    state_requests = []
    sequence = 0
    with input_path.open("w", encoding="utf-8") as input_handle:
        for manifest_index, row in enumerate(rows, start=1):
            subargs = _manifest_dataset_args(args, row)
            items = _llm_work_items(subargs, resume_cache, total=None)
            for item in items:
                if item.cached is not None:
                    continue
                sequence += 1
                custom_id = _batch_custom_id(sequence, item.request_fingerprint)
                _write_jsonl(
                    input_handle,
                    openai_batch_request(
                        custom_id,
                        item.request,
                        model=subargs.model,
                        max_tokens=subargs.max_tokens,
                        temperature=subargs.temperature,
                    ),
                )
                state_requests.append(
                    {
                        "custom_id": custom_id,
                        "manifest_index": manifest_index,
                        "row_index": item.index,
                        "request_fingerprint": item.request_fingerprint,
                        "content_fingerprint": item.content_fingerprint,
                        "out": str(subargs.out),
                        "errors": str(subargs.errors),
                        "raw_responses": str(subargs.raw_responses) if subargs.raw_responses else None,
                    }
                )
    _ensure_parent(args.batch_state)
    batch_id = None
    input_file_id = None
    raw_batch = None
    if state_requests:
        client = _create_llm_client(args)
        with input_path.open("rb") as input_file:
            uploaded = client.files.create(file=input_file, purpose="batch")
        uploaded_dict = _object_to_dict(uploaded)
        input_file_id = str(uploaded_dict.get("id") or getattr(uploaded, "id", ""))
        batch = client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/responses",
            completion_window="24h",
        )
        raw_batch = _object_to_dict(batch)
        batch_id = str(raw_batch.get("id") or getattr(batch, "id", ""))
    state = {
        "type": "openai_batch_state",
        "provider": "openai",
        "created_at": utc_now(),
        "batch_id": batch_id,
        "input_file_id": input_file_id,
        "input_file": str(input_path),
        "batch": raw_batch,
        "manifest": rows,
        "request_count": len(state_requests),
        "requests": state_requests,
        "run_id": args.run_id,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "segmentation_version": args.segmentation_version,
    }
    args.batch_state.write_text(json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print("batch_id=%s requests=%d state=%s input=%s" % (batch_id or "none", len(state_requests), args.batch_state, input_path))
    return 0


def openai_batch_collect_command(args: argparse.Namespace) -> int:
    """Collect OpenAI Batch results and write validated artifacts."""
    state = json.loads(args.batch_state.read_text(encoding="utf-8"))
    rows = state.get("manifest") or []
    args.provider = "openai"
    args.run_id = state.get("run_id")
    args.segmentation_version = state.get("segmentation_version", "seg_v1")
    args.created_at = None
    args.limit = None
    args.model = state.get("model", DEFAULT_OPENAI_MODEL)
    args.max_tokens = state.get("max_tokens", DEFAULT_MAX_TOKENS)
    args.temperature = state.get("temperature", 0.0)
    args.llm_retries = 0
    args.concurrency = 1
    args.progress = "off"
    args.resume = True
    batch_id = state.get("batch_id")
    batch_results = {}
    if batch_id:
        client = getattr(args, "_openai_client", None) or create_openai_client(
            max_retries=getattr(args, "openai_max_retries", 2)
        )
        batch = client.batches.retrieve(str(batch_id))
        batch_dict = _object_to_dict(batch)
        status = str(batch_dict.get("status") or getattr(batch, "status", ""))
        if status and status != "completed":
            print("batch_id=%s status=%s results=not_ready" % (batch_id, status))
            return 0
        output_file_id = batch_dict.get("output_file_id") or getattr(batch, "output_file_id", None)
        if output_file_id:
            request_by_custom = {str(item["custom_id"]): item for item in state.get("requests", [])}
            for result_dict in _openai_batch_output_rows(client, str(output_file_id)):
                custom_id = str(result_dict.get("custom_id", ""))
                request_state = request_by_custom.get(custom_id)
                if request_state:
                    batch_results[_batch_state_row_key(request_state)] = result_dict
    summary = _write_batch_collect_outputs(args, rows, state, batch_results)
    print(
        "written=%d cached=%d errors=%d raw_responses=%d batch_id=%s"
        % (summary["written"], summary["cached"], summary["errors"], summary["raw_responses"], batch_id or "none")
    )
    return 0


def llm_batch_submit_command(args: argparse.Namespace) -> int:
    """Submit provider batch work. OpenAI is the default provider."""
    _normalize_llm_args(args)
    if args.provider == "openai":
        return openai_batch_submit_command(args)
    return anthropic_batch_submit_command(args)


def llm_batch_collect_command(args: argparse.Namespace) -> int:
    """Collect provider batch work using the provider recorded in state."""
    state = json.loads(args.batch_state.read_text(encoding="utf-8"))
    provider = str(state.get("provider") or getattr(args, "provider", DEFAULT_LLM_PROVIDER))
    if provider == "openai":
        return openai_batch_collect_command(args)
    if provider == "anthropic":
        return anthropic_batch_collect_command(args)
    raise DatasetError("unsupported provider %r in batch state; expected one of %s" % (provider, ", ".join(SUPPORTED_LLM_PROVIDERS)))


def _openai_batch_output_rows(client: Any, output_file_id: str) -> Iterable[Dict[str, Any]]:
    """Yield OpenAI Batch output rows from the uploaded JSONL output file."""
    content = client.files.content(output_file_id)
    if hasattr(content, "text"):
        text = str(content.text)
    elif hasattr(content, "read"):
        data = content.read()
        text = data.decode("utf-8") if isinstance(data, bytes) else str(data)
    elif isinstance(content, bytes):
        text = content.decode("utf-8")
    else:
        text = str(content)
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LLMIngestError("OpenAI batch output line %d invalid JSON: %s" % (line_number, exc))
        if not isinstance(row, dict):
            raise LLMIngestError("OpenAI batch output line %d must be a JSON object" % line_number)
        yield row


def _batch_state_row_key(request_state: Dict[str, Any]) -> str:
    """Return the stable collect key for a manifest row recorded in batch state."""
    return "%s:%s" % (request_state.get("manifest_index"), request_state.get("row_index"))


def _load_llm_manifest(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL manifest rows for bulk and batch commands."""
    rows = []
    for index, row in enumerate(read_jsonl(path), start=1):
        if not isinstance(row, dict):
            raise DatasetError("manifest row %d must be a JSON object" % index)
        rows.append(row)
    if not rows:
        raise DatasetError("manifest is empty: %s" % path)
    return rows


def _manifest_dataset_args(args: argparse.Namespace, row: Dict[str, Any]) -> argparse.Namespace:
    """Merge manifest row fields with bulk/batch command defaults."""
    provider = str(row.get("provider") or getattr(args, "provider", DEFAULT_LLM_PROVIDER))
    if provider not in SUPPORTED_LLM_PROVIDERS:
        raise DatasetError("unsupported provider %r in manifest row; expected one of %s" % (provider, ", ".join(SUPPORTED_LLM_PROVIDERS)))
    dataset_name = str(row.get("dataset_name") or row.get("name") or "")
    if not dataset_name:
        raise DatasetError("manifest row missing dataset_name")
    source_field = row.get("source_field") or row.get("source_text_field") or DEFAULT_MANIFEST_SOURCE_FIELDS.get(dataset_name)
    if not source_field:
        raise DatasetError("manifest row for %s missing source_field" % dataset_name)
    dataset = row.get("dataset") or row.get("dataset_path")
    if not dataset:
        raise DatasetError("manifest row for %s missing dataset" % dataset_name)
    run_id = row.get("run_id") or getattr(args, "run_id", None)
    if not run_id:
        raise DatasetError("manifest row for %s missing run_id and no --run-id was provided" % dataset_name)
    out = row.get("out")
    errors = row.get("errors")
    if not out or not errors:
        raise DatasetError("manifest row for %s must include out and errors paths" % dataset_name)
    raw_responses = row.get("raw_responses")
    return argparse.Namespace(
        dataset=Path(str(dataset)),
        dataset_name=dataset_name,
        source_field=str(source_field),
        run_id=str(run_id),
        segmentation_version=str(row.get("segmentation_version") or getattr(args, "segmentation_version", "seg_v1")),
        created_at=row.get("created_at") or getattr(args, "created_at", None),
        limit=row.get("limit", getattr(args, "limit", None)),
        out=Path(str(out)),
        errors=Path(str(errors)),
        raw_responses=Path(str(raw_responses)) if raw_responses else None,
        provider=provider,
        model=str(row.get("model") or getattr(args, "model", None) or _default_model_for_provider(provider)),
        max_tokens=int(row.get("max_tokens", getattr(args, "max_tokens", DEFAULT_MAX_TOKENS))),
        temperature=float(row.get("temperature", getattr(args, "temperature", 0.0))),
        resume=bool(getattr(args, "resume", True)),
        resume_include_temp=bool(getattr(args, "resume_include_temp", True)),
        raw_responses_mode=getattr(args, "raw_responses_mode", None),
        llm_retries=int(getattr(args, "llm_retries", 1)),
        anthropic_max_retries=int(getattr(args, "anthropic_max_retries", getattr(args, "provider_max_retries", 2))),
        openai_max_retries=int(getattr(args, "openai_max_retries", getattr(args, "provider_max_retries", 2))),
        provider_max_retries=int(getattr(args, "provider_max_retries", 2)),
        concurrency=max(1, int(getattr(args, "concurrency", 1))),
        progress=getattr(args, "progress", "auto"),
        _anthropic_client=getattr(args, "_anthropic_client", None),
        _openai_client=getattr(args, "_openai_client", None),
    )


def _load_manifest_resume_cache(rows: Sequence[Dict[str, Any]], args: argparse.Namespace) -> ResumeCache:
    """Load a combined cache from every output path listed in the manifest."""
    combined = ResumeCache()
    seen = set()
    include_temp = bool(getattr(args, "resume_include_temp", True))
    for manifest_index, row in enumerate(rows, start=1):
        out = row.get("out")
        if not out:
            continue
        out_path = Path(str(out))
        if str(out_path) in seen:
            continue
        seen.add(str(out_path))
        _merge_resume_cache(combined, _load_resume_cache(out_path, include_temp=include_temp))
    return combined


def _merge_resume_cache(target: ResumeCache, source: ResumeCache) -> None:
    """Merge source rows and load stats into target."""
    target.by_request.update(source.by_request)
    target.by_content.update(source.by_content)
    target.stats.skipped_invalid += source.stats.skipped_invalid
    target.stats.skipped_missing_fingerprint += source.stats.skipped_missing_fingerprint


def _batch_custom_id(sequence: int, request_fingerprint: str) -> str:
    """Build a short stable custom id for provider batch result correlation."""
    digest = request_fingerprint.split(":", 1)[-1]
    return "req_%06d_%s" % (sequence, digest[:40])


def _batch_result_for_request(
    result: Any,
    result_dict: Dict[str, Any],
    request_state: Dict[str, Any],
    state: Dict[str, Any],
    batch_id: str,
) -> Dict[str, Any]:
    """Normalize one Anthropic batch result into either a response or provider error."""
    result_payload = result_dict.get("result")
    if result_payload is None:
        result_payload = getattr(result, "result", None)
    result_payload_dict = _object_to_dict(result_payload)
    result_type = str(result_payload_dict.get("type") or getattr(result_payload, "type", ""))
    if result_type == "succeeded":
        message = result_payload_dict.get("message") if isinstance(result_payload_dict, dict) else None
        if message is None:
            message = getattr(result_payload, "message", None)
        response = response_from_anthropic_message(
            request_state["request"],
            message,
            model=str(state.get("model") or DEFAULT_ANTHROPIC_MODEL),
            created_at=None,
        )
        response = _response_with_attempt_metadata(
            response,
            attempt=1,
            max_attempts=1,
            request_fingerprint=str(request_state["request_fingerprint"]),
            content_fingerprint=str(request_state["content_fingerprint"]),
        )
        response["batch_id"] = batch_id
        response["batch_custom_id"] = request_state["custom_id"]
        return {"type": "succeeded", "response": response}
    return {
        "type": result_type or "errored",
        "error": result_payload_dict,
        "raw_result": result_dict,
        "batch_id": batch_id,
        "batch_custom_id": request_state["custom_id"],
    }


def _provider_batch_error(provider: str, message: str) -> Exception:
    """Build the provider-specific batch exception type for sidecar errors."""
    if provider == "openai":
        return OpenAIRunError(message)
    return AnthropicRunError(message)


def _batch_result_succeeded(provider: str, batch_result: Dict[str, Any]) -> bool:
    """Return true when a provider batch result contains a successful model response."""
    if provider != "openai":
        return batch_result.get("type") == "succeeded"
    response = batch_result.get("response")
    if not isinstance(response, dict):
        return False
    try:
        status_code = int(response.get("status_code", 0))
    except (TypeError, ValueError):
        status_code = 0
    return 200 <= status_code < 300 and response.get("body") is not None


def _openai_batch_error(batch_result: Dict[str, Any]) -> Any:
    """Extract the useful error payload from an OpenAI Batch result row."""
    if batch_result.get("error") is not None:
        return batch_result.get("error")
    response = batch_result.get("response")
    if isinstance(response, dict):
        return response.get("body") or response
    return batch_result


def _write_batch_collect_outputs(
    args: argparse.Namespace,
    rows: Sequence[Dict[str, Any]],
    state: Dict[str, Any],
    batch_results: Dict[str, Dict[str, Any]],
) -> Dict[str, int]:
    """Write all manifest dataset artifacts from cache hits and collected batch results."""
    provider = str(state.get("provider") or getattr(args, "provider", "anthropic"))
    batch_stage = "%s_batch" % provider
    summary = {"written": 0, "cached": 0, "errors": 0, "raw_responses": 0}
    for manifest_index, row in enumerate(rows, start=1):
        subargs = _manifest_dataset_args(args, row)
        resume_cache = _load_manifest_resume_cache(rows, args)
        items = _llm_work_items(subargs, resume_cache, total=None)
        _ensure_parent(subargs.out)
        _ensure_parent(subargs.errors)
        if subargs.raw_responses:
            _ensure_parent(subargs.raw_responses)
        out_path = _resume_temp_path(subargs.out)
        raw_mode = getattr(args, "raw_responses_mode", None) or "overwrite"
        raw_handle = subargs.raw_responses.open("a" if raw_mode == "append" else "w", encoding="utf-8") if subargs.raw_responses else None
        try:
            with out_path.open("w", encoding="utf-8") as out_handle, subargs.errors.open("w", encoding="utf-8") as err_handle:
                for item in items:
                    if item.cached is not None:
                        result = _cached_llm_result(item)
                        _write_jsonl(out_handle, result.row or {})
                        summary["written"] += 1
                        summary["cached"] += 1
                        continue
                    batch_key = "%d:%d" % (manifest_index, item.index) if provider == "openai" else item.request_fingerprint
                    batch_result = batch_results.get(batch_key)
                    if not batch_result:
                        error = _run_error(
                            item.example,
                            subargs.run_id,
                            subargs.segmentation_version,
                            batch_stage,
                            _provider_batch_error(provider, "batch result missing for request %s" % item.request_fingerprint),
                        )
                        error["request_fingerprint"] = item.request_fingerprint
                        error["content_fingerprint"] = item.content_fingerprint
                        error["attempts"] = 0
                        error["retryable"] = False
                        error["retry_errors"] = []
                        _write_jsonl(err_handle, error)
                        summary["errors"] += 1
                        continue
                    if _batch_result_succeeded(provider, batch_result):
                        if provider == "openai":
                            response = response_from_openai_batch_result(
                                item.request,
                                batch_result,
                                model=str(state.get("model") or DEFAULT_OPENAI_MODEL),
                                created_at=None,
                            )
                            response = _response_with_attempt_metadata(
                                response,
                                attempt=1,
                                max_attempts=1,
                                request_fingerprint=item.request_fingerprint,
                                content_fingerprint=item.content_fingerprint,
                            )
                        else:
                            response = batch_result["response"]
                        if raw_handle:
                            _write_jsonl(raw_handle, response)
                            summary["raw_responses"] += 1
                        try:
                            if provider == "openai":
                                raise_for_incomplete_openai_response(response)
                            else:
                                raise_for_incomplete_anthropic_response(response)
                            record = record_from_response(item.request, response)
                            record.request_fingerprint = item.request_fingerprint
                            record.content_fingerprint = item.content_fingerprint
                            _write_jsonl(out_handle, record.to_dict())
                            summary["written"] += 1
                        except Exception as exc:  # noqa: BLE001 - final batch row failure.
                            error = _run_error(item.example, subargs.run_id, subargs.segmentation_version, batch_stage, exc)
                            error["request_fingerprint"] = item.request_fingerprint
                            error["content_fingerprint"] = item.content_fingerprint
                            error["attempts"] = 1
                            error["retryable"] = _retryable_llm_failure(exc)
                            error["retry_errors"] = [_retry_error_item(exc, 1, response)]
                            error["stop_reason"] = response.get("stop_reason")
                            error["provider_message_id"] = response.get("provider_message_id")
                            error["provider_response_id"] = response.get("provider_response_id")
                            _write_jsonl(err_handle, error)
                            summary["errors"] += 1
                    else:
                        provider_error = _openai_batch_error(batch_result) if provider == "openai" else batch_result.get("error")
                        error = _run_error(
                            item.example,
                            subargs.run_id,
                            subargs.segmentation_version,
                            batch_stage,
                            _provider_batch_error(provider, "batch result %s" % (batch_result.get("type") or "provider_error")),
                        )
                        error["request_fingerprint"] = item.request_fingerprint
                        error["content_fingerprint"] = item.content_fingerprint
                        error["batch_id"] = batch_result.get("batch_id")
                        error["batch_custom_id"] = batch_result.get("batch_custom_id")
                        error["provider_error"] = provider_error
                        error["attempts"] = 1
                        error["retryable"] = False
                        error["retry_errors"] = [
                            {
                                "attempt": 1,
                                "error_type": str(batch_result.get("type") or "provider_error"),
                                "message": _clip(json.dumps(provider_error, sort_keys=True), 500),
                            }
                        ]
                        _write_jsonl(err_handle, error)
                        summary["errors"] += 1
            os.replace(out_path, subargs.out)
        finally:
            if raw_handle:
                raw_handle.close()
    return summary


def _object_to_dict(value: Any) -> Dict[str, Any]:
    """Best-effort conversion for SDK objects and fake test dictionaries."""
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        data = value.model_dump(mode="json")
        if isinstance(data, dict):
            return data
    if hasattr(value, "to_dict"):
        data = value.to_dict()
        if isinstance(data, dict):
            return data
    return {"repr": repr(value)}


def _llm_work_items(args: argparse.Namespace, resume_cache: ResumeCache, total: Optional[int]) -> List[LLMWorkItem]:
    """Build all row work units up front so cache lookup and output order are explicit."""
    items = []
    for index, example in enumerate(
        load_dataset(args.dataset, args.dataset_name, args.source_field, args.limit),
        start=1,
    ):
        request = build_request(
            example,
            run_id=args.run_id,
            segmentation_version=args.segmentation_version,
            created_at=args.created_at,
        )
        request_fingerprint = _request_fingerprint(
            request,
            provider=args.provider,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        content_fingerprint = _content_fingerprint(
            request,
            provider=args.provider,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        cached = resume_cache.by_request.get(request_fingerprint)
        content_cache_hit = False
        if cached is None:
            cached = resume_cache.by_content.get(content_fingerprint)
            content_cache_hit = cached is not None
        items.append(
            LLMWorkItem(
                index=index,
                total=total,
                example=example,
                request=request,
                request_fingerprint=request_fingerprint,
                content_fingerprint=content_fingerprint,
                cached=cached,
                content_cache_hit=content_cache_hit,
            )
        )
    return items


def _run_llm_work_items(
    items: Sequence[LLMWorkItem],
    args: argparse.Namespace,
    client: Any,
    max_attempts: int,
    raw_handle: Any,
    raw_lock: threading.Lock,
    progress: Any,
    progress_mode: str,
    progress_lock: threading.Lock,
    concurrency: int,
) -> Iterable[LLMWorkResult]:
    """Run cached and generated rows, yielding results in dataset order."""
    if concurrency <= 1:
        for item in items:
            yield _run_one_llm_item(
                item,
                args,
                client,
                max_attempts,
                raw_handle,
                raw_lock,
                progress,
                progress_mode,
                progress_lock,
            )
        return

    completed: Dict[int, LLMWorkResult] = {}
    next_index = 1
    futures = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for item in items:
            if item.cached is not None:
                completed[item.index] = _cached_llm_result(item)
            else:
                futures.append(
                    executor.submit(
                        _run_one_llm_item,
                        item,
                        args,
                        client,
                        max_attempts,
                        raw_handle,
                        raw_lock,
                        None,
                        "off",
                        progress_lock,
                    )
                )
            while next_index in completed:
                yield completed.pop(next_index)
                next_index += 1

        for future in as_completed(futures):
            result = future.result()
            completed[result.item.index] = result
            while next_index in completed:
                yield completed.pop(next_index)
                next_index += 1


def _run_one_llm_item(
    item: LLMWorkItem,
    args: argparse.Namespace,
    client: Any,
    max_attempts: int,
    raw_handle: Any,
    raw_lock: threading.Lock,
    progress: Any,
    progress_mode: str,
    progress_lock: threading.Lock,
) -> LLMWorkResult:
    """Produce one cached or generated LLM result without writing final artifacts."""
    if item.cached is not None:
        return _cached_llm_result(item)

    response: Optional[Dict[str, Any]] = None
    try:
        record, response, attempt_count, _retryable, _retry_errors = _call_provider_with_llm_retries(
            provider=args.provider,
            request=item.request,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            client=client,
            created_at=args.created_at,
            raw_handle=raw_handle,
            raw_lock=raw_lock,
            request_fingerprint=item.request_fingerprint,
            content_fingerprint=item.content_fingerprint,
            max_attempts=max_attempts,
            progress=progress,
            progress_mode=progress_mode,
            progress_lock=progress_lock,
            index=item.index,
            total=item.total,
            example_id=item.example.example_id,
            written=0,
            errors=0,
            cached=0,
            retries=0,
        )
        record.request_fingerprint = item.request_fingerprint
        record.content_fingerprint = item.content_fingerprint
        return LLMWorkResult(
            item=item,
            status="ok",
            row=record.to_dict(),
            attempt_count=attempt_count,
            raw_responses=attempt_count,
            retries=max(0, attempt_count - 1),
        )
    except Exception as exc:  # noqa: BLE001 - batch tool logs row-level failures.
        error, raw_count, retry_count = _llm_error_from_exception(item, args, exc, response)
        return LLMWorkResult(
            item=item,
            status="error",
            error=error,
            raw_responses=raw_count,
            retries=retry_count,
            error_type=str(error["error_type"]),
        )


def _cached_llm_result(item: LLMWorkItem) -> LLMWorkResult:
    """Return a cache hit row rewritten only when the hit came from the content cache."""
    row = deepcopy(item.cached.row) if item.cached else {}
    if item.content_cache_hit:
        row = _rewrite_cached_row_for_request(row, item.request, item.request_fingerprint, item.content_fingerprint)
    else:
        row["request_fingerprint"] = item.request_fingerprint
        row["content_fingerprint"] = item.content_fingerprint
    validate_record_dict(row)
    return LLMWorkResult(item=item, status="cached", row=row)


def _rewrite_cached_row_for_request(
    row: Dict[str, Any],
    request: Dict[str, Any],
    request_fingerprint: str,
    content_fingerprint: str,
) -> Dict[str, Any]:
    """Reuse byte-identical segmentation while replacing target-row provenance."""
    rewritten = deepcopy(row)
    rewritten.update(
        {
            "example_id": str(request["example_id"]),
            "dataset_name": str(request["dataset_name"]),
            "source_text_field": str(request["source_text_field"]),
            "run_id": str(request["run_id"]),
            "segmentation_version": str(request["segmentation_version"]),
            "created_at": str(request.get("created_at") or rewritten.get("created_at") or utc_now()),
            "raw_source_text": str(request["raw_source_text"]),
            "request_fingerprint": request_fingerprint,
            "content_fingerprint": content_fingerprint,
        }
    )
    rewritten.pop("gold_label", None)
    validate_record_dict(rewritten)
    return rewritten


def _llm_error_from_exception(
    item: LLMWorkItem,
    args: argparse.Namespace,
    exc: Exception,
    response: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], int, int]:
    """Build the final row failure while preserving retry metadata."""
    final_exc = exc
    raw_responses = 0
    retries = 0
    if isinstance(exc, LLMAttemptFailure):
        final_exc = exc.original
        response = exc.response
        raw_responses = exc.raw_responses
        retries = max(0, exc.attempts - 1)
    provider = str(getattr(args, "provider", DEFAULT_LLM_PROVIDER))
    error = _run_error(item.example, args.run_id, args.segmentation_version, provider, final_exc)
    error["request_id"] = item.request.get("request_id")
    error["request_fingerprint"] = item.request_fingerprint
    error["content_fingerprint"] = item.content_fingerprint
    if response:
        error["stop_reason"] = response.get("stop_reason")
        error["provider_message_id"] = response.get("provider_message_id")
        error["provider_response_id"] = response.get("provider_response_id")
    if isinstance(exc, LLMAttemptFailure):
        error["attempts"] = exc.attempts
        error["retryable"] = exc.retryable
        error["retry_errors"] = exc.retry_errors
    return error, raw_responses, retries


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

    llm_parser = subparsers.add_parser(
        "llm",
        help="call the default LLM provider and write validated shards.jsonl",
        description="Run native LLM segmentation, align verbatim spans locally, and validate each record. OpenAI is the default provider.",
        epilog="""Example:
  python -m decomposition.cli llm --dataset datasets/AITA-YTA.csv \
--dataset-name AITA-YTA --source-field prompt --run-id pilot \
--provider openai --model gpt-5.4-mini \
--resume --llm-retries 1 --provider-max-retries 2 \
--out decomposition/artifacts/shards.AITA-YTA.openai.jsonl \
--errors decomposition/artifacts/run_errors.AITA-YTA.openai.jsonl \
--raw-responses decomposition/artifacts/seg_v1_openai_responses.AITA-YTA.jsonl
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_dataset_args(llm_parser)
    llm_parser.add_argument("--provider", choices=SUPPORTED_LLM_PROVIDERS, default=DEFAULT_LLM_PROVIDER, help="LLM provider")
    llm_parser.add_argument("--out", type=Path, required=True, help="destination shards.jsonl path")
    llm_parser.add_argument("--errors", type=Path, required=True, help="destination run_errors.jsonl path")
    llm_parser.add_argument("--raw-responses", type=Path, default=None, help="optional replay/audit JSONL containing raw response wrappers")
    llm_parser.add_argument("--model", default=None, help="model name; defaults to the provider default")
    llm_parser.add_argument("--resume", action="store_true", help="reuse matching valid rows already present in --out")
    llm_parser.add_argument("--resume-include-temp", dest="resume_include_temp", action="store_true", default=True)
    llm_parser.add_argument("--no-resume-include-temp", dest="resume_include_temp", action="store_false")
    llm_parser.add_argument("--raw-responses-mode", choices=("append", "overwrite"), default=None)
    llm_parser.add_argument("--concurrency", type=int, default=1, help="maximum concurrent uncached row calls")
    llm_parser.add_argument("--llm-retries", type=int, default=1, help="extra model-output attempts for retryable failures")
    llm_parser.add_argument("--provider-max-retries", type=int, default=2, help="provider SDK transport/API retry count")
    llm_parser.add_argument("--openai-max-retries", type=int, default=2, help="OpenAI SDK transport/API retry count")
    llm_parser.add_argument("--anthropic-max-retries", type=int, default=2, help="Anthropic SDK transport/API retry count")
    llm_parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="maximum output tokens")
    llm_parser.add_argument("--temperature", type=float, default=0.0, help="sampling temperature")
    llm_parser.add_argument("--progress", choices=("auto", "bar", "log", "off"), default="auto", help="progress display")
    llm_parser.set_defaults(func=llm_command)

    openai_parser = subparsers.add_parser(
        "openai",
        help="call OpenAI and write validated shards.jsonl",
        description="Run native OpenAI segmentation through the Responses API and local validation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_dataset_args(openai_parser)
    openai_parser.add_argument("--out", type=Path, required=True, help="destination shards.jsonl path")
    openai_parser.add_argument("--errors", type=Path, required=True, help="destination run_errors.jsonl path")
    openai_parser.add_argument("--raw-responses", type=Path, default=None, help="optional replay/audit JSONL containing raw OpenAI response wrappers")
    openai_parser.add_argument("--model", default=DEFAULT_OPENAI_MODEL, help="OpenAI model name")
    openai_parser.add_argument("--resume", action="store_true", help="reuse matching valid rows already present in --out")
    openai_parser.add_argument("--resume-include-temp", dest="resume_include_temp", action="store_true", default=True)
    openai_parser.add_argument("--no-resume-include-temp", dest="resume_include_temp", action="store_false")
    openai_parser.add_argument("--raw-responses-mode", choices=("append", "overwrite"), default=None)
    openai_parser.add_argument("--concurrency", type=int, default=1, help="maximum concurrent OpenAI row calls for uncached realtime work")
    openai_parser.add_argument("--llm-retries", type=int, default=1, help="extra model-output attempts for retryable failures")
    openai_parser.add_argument("--openai-max-retries", type=int, default=2, help="OpenAI SDK transport/API retry count")
    openai_parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="maximum output tokens for each OpenAI response")
    openai_parser.add_argument("--temperature", type=float, default=0.0, help="sampling temperature for OpenAI generation")
    openai_parser.add_argument("--progress", choices=("auto", "bar", "log", "off"), default="auto", help="progress display")
    openai_parser.set_defaults(func=openai_command)

    anthropic_parser = subparsers.add_parser(
        "anthropic",
        help="call Anthropic and write validated shards.jsonl",
        description="Run native Anthropic segmentation, align verbatim spans locally, and validate each record.",
        epilog="""Example:
  python -m decomposition.cli anthropic --dataset datasets/AITA-NTA-OG.csv \
--dataset-name AITA-NTA-OG --source-field original_post --run-id pilot \
--model claude-sonnet-4-5 \
--resume --llm-retries 1 --anthropic-max-retries 2 \
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
        "--resume",
        action="store_true",
        help="reuse matching valid rows already present in --out and query only uncached examples",
    )
    anthropic_parser.add_argument(
        "--resume-include-temp",
        dest="resume_include_temp",
        action="store_true",
        default=True,
        help="include a same-directory --out.tmp file when loading resume cache",
    )
    anthropic_parser.add_argument(
        "--no-resume-include-temp",
        dest="resume_include_temp",
        action="store_false",
        help="ignore --out.tmp while loading resume cache",
    )
    anthropic_parser.add_argument(
        "--raw-responses-mode",
        choices=("append", "overwrite"),
        default=None,
        help="raw sidecar write mode; default is append with --resume and overwrite otherwise",
    )
    anthropic_parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="maximum concurrent Anthropic row calls for uncached realtime work",
    )
    anthropic_parser.add_argument(
        "--llm-retries",
        type=int,
        default=1,
        help="extra model-output attempts for retryable ingest failures",
    )
    anthropic_parser.add_argument(
        "--anthropic-max-retries",
        type=int,
        default=2,
        help="Anthropic SDK transport/API retry count",
    )
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
    anthropic_parser.add_argument(
        "--progress",
        choices=("auto", "bar", "log", "off"),
        default="auto",
        help="progress display: auto picks bar for terminals and log for redirected output",
    )
    anthropic_parser.set_defaults(func=anthropic_command)

    llm_bulk_parser = subparsers.add_parser(
        "llm-bulk",
        help="run LLM segmentation for every dataset in a JSONL manifest",
        description="Run the realtime provider path across multiple datasets while reusing manifest-wide caches.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_llm_manifest_args(llm_bulk_parser)
    llm_bulk_parser.add_argument("--provider", choices=SUPPORTED_LLM_PROVIDERS, default=DEFAULT_LLM_PROVIDER, help="LLM provider")
    llm_bulk_parser.set_defaults(func=llm_bulk_command)

    openai_bulk_parser = subparsers.add_parser(
        "openai-bulk",
        help="run OpenAI segmentation for every dataset in a JSONL manifest",
        description="Run the realtime OpenAI path across multiple datasets while reusing manifest-wide caches.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_llm_manifest_args(openai_bulk_parser)
    openai_bulk_parser.set_defaults(func=openai_bulk_command)

    bulk_parser = subparsers.add_parser(
        "anthropic-bulk",
        help="run Anthropic segmentation for every dataset in a JSONL manifest",
        description="Run the realtime Anthropic path across multiple datasets while reusing manifest-wide caches.",
        epilog="""Example:
  python -m decomposition.cli anthropic-bulk --manifest decomposition/artifacts/anthropic_manifest.jsonl \
--run-id pilot --resume --concurrency 2
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_llm_manifest_args(bulk_parser)
    bulk_parser.set_defaults(func=anthropic_bulk_command)

    llm_batch_parser = subparsers.add_parser(
        "llm-batch",
        help="submit or collect LLM provider batch jobs",
        description="Use provider batch APIs for asynchronous multi-dataset decomposition. OpenAI is the default provider.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    llm_batch_subparsers = llm_batch_parser.add_subparsers(
        dest="batch_command",
        metavar="BATCH_COMMAND",
        required=True,
        parser_class=HelpfulArgumentParser,
    )
    llm_batch_submit = llm_batch_subparsers.add_parser("submit", help="submit manifest cache misses")
    _add_llm_manifest_args(llm_batch_submit)
    llm_batch_submit.add_argument("--provider", choices=SUPPORTED_LLM_PROVIDERS, default=DEFAULT_LLM_PROVIDER, help="LLM provider")
    llm_batch_submit.add_argument("--batch-state", type=Path, required=True, help="JSON state sidecar to write")
    llm_batch_submit.add_argument("--batch-input", type=Path, default=None, help="OpenAI batch input JSONL path")
    llm_batch_submit.set_defaults(func=llm_batch_submit_command)

    llm_batch_collect = llm_batch_subparsers.add_parser("collect", help="collect a completed provider batch")
    llm_batch_collect.add_argument("--batch-state", type=Path, required=True, help="JSON state sidecar from submit")
    llm_batch_collect.add_argument("--raw-responses-mode", choices=("append", "overwrite"), default="overwrite")
    llm_batch_collect.add_argument("--resume-include-temp", dest="resume_include_temp", action="store_true", default=True)
    llm_batch_collect.add_argument("--no-resume-include-temp", dest="resume_include_temp", action="store_false")
    llm_batch_collect.add_argument("--provider-max-retries", type=int, default=2, help="provider SDK transport/API retry count")
    llm_batch_collect.add_argument("--openai-max-retries", type=int, default=2, help="OpenAI SDK transport/API retry count")
    llm_batch_collect.add_argument("--anthropic-max-retries", type=int, default=2, help="Anthropic SDK transport/API retry count")
    llm_batch_collect.set_defaults(func=llm_batch_collect_command)

    openai_batch_parser = subparsers.add_parser(
        "openai-batch",
        help="submit or collect OpenAI Batch jobs",
        description="Use OpenAI Batch for asynchronous multi-dataset decomposition.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    openai_batch_subparsers = openai_batch_parser.add_subparsers(
        dest="batch_command",
        metavar="BATCH_COMMAND",
        required=True,
        parser_class=HelpfulArgumentParser,
    )
    openai_batch_submit = openai_batch_subparsers.add_parser("submit", help="submit manifest cache misses to OpenAI Batch")
    _add_llm_manifest_args(openai_batch_submit)
    openai_batch_submit.add_argument("--batch-state", type=Path, required=True, help="JSON state sidecar to write")
    openai_batch_submit.add_argument("--batch-input", type=Path, default=None, help="OpenAI batch input JSONL path")
    openai_batch_submit.set_defaults(func=openai_batch_submit_command)

    openai_batch_collect = openai_batch_subparsers.add_parser("collect", help="collect a completed OpenAI Batch into artifacts")
    openai_batch_collect.add_argument("--batch-state", type=Path, required=True, help="JSON state sidecar from submit")
    openai_batch_collect.add_argument("--raw-responses-mode", choices=("append", "overwrite"), default="overwrite")
    openai_batch_collect.add_argument("--resume-include-temp", dest="resume_include_temp", action="store_true", default=True)
    openai_batch_collect.add_argument("--no-resume-include-temp", dest="resume_include_temp", action="store_false")
    openai_batch_collect.add_argument("--openai-max-retries", type=int, default=2, help="OpenAI SDK transport/API retry count")
    openai_batch_collect.set_defaults(func=openai_batch_collect_command)

    batch_parser = subparsers.add_parser(
        "anthropic-batch",
        help="submit or collect Anthropic Message Batch jobs",
        description="Use Anthropic Message Batches for asynchronous multi-dataset decomposition.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    batch_subparsers = batch_parser.add_subparsers(
        dest="batch_command",
        metavar="BATCH_COMMAND",
        required=True,
        parser_class=HelpfulArgumentParser,
    )
    batch_submit = batch_subparsers.add_parser("submit", help="submit manifest cache misses to Message Batches")
    _add_llm_manifest_args(batch_submit)
    batch_submit.add_argument("--batch-state", type=Path, required=True, help="JSON state sidecar to write")
    batch_submit.set_defaults(func=anthropic_batch_submit_command)

    batch_collect = batch_subparsers.add_parser("collect", help="collect a completed Message Batch into artifacts")
    batch_collect.add_argument("--batch-state", type=Path, required=True, help="JSON state sidecar from submit")
    batch_collect.add_argument(
        "--raw-responses-mode",
        choices=("append", "overwrite"),
        default="overwrite",
        help="raw sidecar write mode while collecting batch results",
    )
    batch_collect.add_argument(
        "--resume-include-temp",
        dest="resume_include_temp",
        action="store_true",
        default=True,
        help="include existing --out.tmp rows when collecting batch results",
    )
    batch_collect.add_argument(
        "--no-resume-include-temp",
        dest="resume_include_temp",
        action="store_false",
        help="ignore existing --out.tmp rows while collecting batch results",
    )
    batch_collect.add_argument(
        "--anthropic-max-retries",
        type=int,
        default=2,
        help="Anthropic SDK transport/API retry count",
    )
    batch_collect.set_defaults(func=anthropic_batch_collect_command)

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


def _add_llm_manifest_args(parser: argparse.ArgumentParser) -> None:
    """Attach common multi-dataset LLM arguments."""
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL dataset manifest")
    parser.add_argument("--run-id", default=None, help="run id used when a manifest row omits run_id")
    parser.add_argument("--segmentation-version", default="seg_v1", help="prompt/schema version to use")
    parser.add_argument("--created-at", default=None, help="override artifact timestamp for reproducible tests")
    parser.add_argument("--limit", type=int, default=None, help="optional maximum rows per dataset")
    parser.add_argument("--model", default=None, help="model name; defaults to the provider default")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="maximum output tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="sampling temperature")
    parser.add_argument("--llm-retries", type=int, default=1, help="extra model-output attempts for retryable failures")
    parser.add_argument("--provider-max-retries", type=int, default=2, help="provider SDK transport/API retry count")
    parser.add_argument("--openai-max-retries", type=int, default=2, help="OpenAI SDK transport/API retry count")
    parser.add_argument("--anthropic-max-retries", type=int, default=2, help="Anthropic SDK transport/API retry count")
    parser.add_argument("--concurrency", type=int, default=1, help="maximum concurrent realtime provider row calls")
    parser.add_argument("--progress", choices=("auto", "bar", "log", "off"), default="auto", help="progress display")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True, help="reuse valid cached rows")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="ignore cached rows")
    parser.add_argument(
        "--resume-include-temp",
        dest="resume_include_temp",
        action="store_true",
        default=True,
        help="include --out.tmp files when loading resume caches",
    )
    parser.add_argument("--no-resume-include-temp", dest="resume_include_temp", action="store_false")
    parser.add_argument(
        "--raw-responses-mode",
        choices=("append", "overwrite"),
        default=None,
        help="raw sidecar write mode; default is append with resume",
    )


def _ensure_parent(path: Path) -> None:
    """Create the parent directory for an output path if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_jsonl(handle: Any, data: Dict[str, Any]) -> None:
    """Write one deterministic JSON object as a JSONL row."""
    handle.write(json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n")
    if hasattr(handle, "flush"):
        handle.flush()


def _resume_temp_path(path: Path) -> Path:
    """Build a same-directory temp output path so replace is atomic on one filesystem."""
    return path.with_name(path.name + ".tmp")


def _load_resume_cache(path: Path, include_temp: bool = True) -> ResumeCache:
    """Load valid fingerprinted shard rows from final and optional temp artifacts."""
    cache = ResumeCache()
    paths = [(path, "final")]
    temp_path = _resume_temp_path(path)
    if include_temp and temp_path != path:
        paths.append((temp_path, "tmp"))
    for candidate, source in paths:
        if not candidate.exists():
            continue
        for row in _read_jsonl_lenient(candidate, cache.stats):
            fingerprint = row.get("request_fingerprint")
            if not fingerprint:
                cache.stats.skipped_missing_fingerprint += 1
                continue
            try:
                validate_record_dict(row)
            except ValidationError:
                cache.stats.skipped_invalid += 1
                continue
            cached = CachedShardRow(row=row, source=source)
            cache.by_request[str(fingerprint)] = cached
            content_fingerprint = row.get("content_fingerprint")
            if content_fingerprint:
                cache.by_content[str(content_fingerprint)] = cached
    return cache


def _read_jsonl_lenient(path: Path, stats: ResumeStats) -> Iterable[Dict[str, Any]]:
    """Yield parseable JSONL rows while counting bad cache rows instead of aborting."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats.skipped_invalid += 1
                continue
            if not isinstance(row, dict):
                stats.skipped_invalid += 1
                continue
            yield row


def _count_cache_hit(stats: ResumeStats, cached: Optional[CachedShardRow]) -> None:
    """Track which cache artifact produced a row."""
    if cached is None:
        return
    if cached.source == "tmp":
        stats.cached_tmp += 1
    else:
        stats.cached_final += 1


def _request_fingerprint(
    request: Dict[str, Any],
    provider: str,
    model: str,
    max_tokens: int,
    temperature: float,
    output_schema: Optional[Dict[str, Any]] = None,
) -> str:
    """Fingerprint all request settings that make a cached shard row reusable."""
    components = {
        "dataset_name": str(request.get("dataset_name", "")),
        "example_id": str(request.get("example_id", "")),
        "source_text_field": str(request.get("source_text_field", "")),
        "run_id": str(request.get("run_id", "")),
        "segmentation_version": str(request.get("segmentation_version", "")),
        "raw_source_text_sha256": _sha256_text(str(request.get("raw_source_text", ""))),
        "prompt_sha256": _sha256_text(_prompt_text_from_request(request)),
        "output_schema_sha256": _sha256_json(output_schema or SEGMENTATION_OUTPUT_SCHEMA),
        "provider": str(provider),
        "model": str(model),
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    return "sha256:" + _sha256_json(components)


def _content_fingerprint(
    request: Dict[str, Any],
    provider: str,
    model: str,
    max_tokens: int,
    temperature: float,
    output_schema: Optional[Dict[str, Any]] = None,
) -> str:
    """Fingerprint byte-identical source text and generation settings across datasets."""
    components = {
        "raw_source_text_sha256": _sha256_text(str(request.get("raw_source_text", ""))),
        "prompt_sha256": _sha256_text(_prompt_text_from_request(request)),
        "output_schema_sha256": _sha256_json(output_schema or SEGMENTATION_OUTPUT_SCHEMA),
        "provider": str(provider),
        "model": str(model),
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    return "sha256:" + _sha256_json(components)


def _prompt_text_from_request(request: Dict[str, Any]) -> str:
    """Extract prompt text from provider-neutral request messages."""
    prompt_parts = []
    for message in request.get("messages", []):
        if isinstance(message, dict) and str(message.get("role", "")) in ("system", "developer"):
            prompt_parts.append(str(message.get("content", "")))
    return "\n\n".join(prompt_parts)


def _sha256_text(value: str) -> str:
    """Return a stable UTF-8 SHA-256 digest."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    """Return a stable JSON SHA-256 digest."""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_text(encoded)


def _call_provider_with_llm_retries(
    provider: str,
    request: Dict[str, Any],
    model: str,
    max_tokens: int,
    temperature: float,
    client: Any,
    created_at: Optional[str],
    raw_handle: Any,
    raw_lock: Optional[threading.Lock],
    request_fingerprint: str,
    content_fingerprint: str,
    max_attempts: int,
    progress: Any,
    progress_mode: str,
    progress_lock: Optional[threading.Lock],
    index: int,
    total: Optional[int],
    example_id: str,
    written: int,
    errors: int,
    cached: int,
    retries: int,
) -> Any:
    """Call the configured provider until the row validates or retries are exhausted."""
    retry_errors = []
    raw_response_count = 0
    response: Optional[Dict[str, Any]] = None
    for attempt in range(1, max_attempts + 1):
        response = None
        try:
            if provider == "openai":
                response = call_openai(
                    request,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    client=client,
                    created_at=created_at,
                )
            elif provider == "anthropic":
                response = call_anthropic(
                    request,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    client=client,
                    created_at=created_at,
                )
            else:
                raise DatasetError("unsupported provider %r" % provider)
            response = _response_with_attempt_metadata(
                response,
                attempt,
                max_attempts,
                request_fingerprint,
                content_fingerprint,
            )
            if raw_handle:
                if raw_lock:
                    with raw_lock:
                        _write_jsonl(raw_handle, response)
                else:
                    _write_jsonl(raw_handle, response)
            raw_response_count += 1
            if provider == "openai":
                raise_for_incomplete_openai_response(response)
            else:
                raise_for_incomplete_anthropic_response(response)
            record = record_from_response(request, response)
            return record, response, attempt, False, retry_errors
        except Exception as exc:  # noqa: BLE001 - row-level retry classification.
            retryable = _retryable_llm_failure(exc)
            retry_errors.append(_retry_error_item(exc, attempt, response))
            if not retryable or attempt >= max_attempts:
                raise LLMAttemptFailure(
                    exc,
                    response,
                    attempts=attempt,
                    raw_responses=raw_response_count,
                    retryable=retryable,
                    retry_errors=retry_errors,
                ) from exc
            retries += 1
            if progress_lock:
                with progress_lock:
                    _progress_update(
                        progress,
                        progress_mode,
                        index,
                        total,
                        example_id,
                        "retry",
                        written,
                        errors,
                        cached=cached,
                        retries=retries,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        advance=False,
                    )
            else:
                _progress_update(
                    progress,
                    progress_mode,
                    index,
                    total,
                    example_id,
                    "retry",
                    written,
                    errors,
                    cached=cached,
                    retries=retries,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    advance=False,
                )
    raise LLMIngestError("unreachable retry state")


def _response_with_attempt_metadata(
    response: Dict[str, Any],
    attempt: int,
    max_attempts: int,
    request_fingerprint: str,
    content_fingerprint: str,
) -> Dict[str, Any]:
    """Annotate raw response sidecars with retry/cache metadata."""
    annotated = dict(response)
    annotated["attempt"] = attempt
    annotated["max_attempts"] = max_attempts
    annotated["request_fingerprint"] = request_fingerprint
    annotated["content_fingerprint"] = content_fingerprint
    return annotated


def _retryable_llm_failure(exc: Exception) -> bool:
    """Return true for model-output failures likely to improve on another sample."""
    if isinstance(exc, (LLMIngestError, AlignmentError, ValidationError)):
        return True
    if isinstance(exc, (AnthropicRunError, OpenAIRunError)):
        message = str(exc)
        if "contained no text content" in message:
            return True
        if "was incomplete" in message:
            return False
        return False
    return False


def _retry_error_item(exc: Exception, attempt: int, response: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Compact per-attempt failure details for the final error row."""
    item: Dict[str, Any] = {
        "attempt": attempt,
        "error_type": exc.__class__.__name__,
        "message": _clip(str(exc), 500),
    }
    if response:
        item["stop_reason"] = response.get("stop_reason")
        item["provider_message_id"] = response.get("provider_message_id")
        item["provider_response_id"] = response.get("provider_response_id")
    return item


def _count_dataset_rows(dataset_path: Path, source_text_field: str, limit: Optional[int]) -> int:
    """Count rows once so progress output can show a useful ETA."""
    with dataset_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise DatasetError(
                "dataset has no header: %s. Fix: provide a CSV with a header row containing the source text column."
                % dataset_path
            )
        if source_text_field not in reader.fieldnames:
            raise DatasetError(
                "source field %r not found in %s; fields=%s. Fix: pass one of these names to --source-field."
                % (source_text_field, dataset_path, reader.fieldnames)
            )
        count = 0
        for count, _row in enumerate(reader, start=1):
            if limit is not None and count >= limit:
                return limit
        return count


def _resolve_progress_mode(progress: str) -> str:
    """Choose progress behavior that works in terminals and non-interactive logs."""
    if progress != "auto":
        return progress
    return "bar" if sys.stderr.isatty() else "log"


def _progress_bar(progress_mode: str, total: Optional[int]) -> Any:
    """Create a tqdm progress bar when requested."""
    if progress_mode != "bar":
        return None
    from tqdm import tqdm

    return tqdm(total=total, desc="Sharding", unit="post", file=sys.stderr)


def _progress_update(
    progress: Any,
    progress_mode: str,
    index: int,
    total: Optional[int],
    example_id: str,
    status: str,
    written: int,
    errors: int,
    cached: int = 0,
    retries: int = 0,
    attempt: Optional[int] = None,
    max_attempts: Optional[int] = None,
    advance: bool = True,
) -> None:
    """Report per-row progress without mixing artifact output into stdout."""
    if progress_mode == "off":
        return
    if progress_mode == "bar" and progress:
        progress.set_postfix(ok=written, errors=errors, cached=cached, retries=retries, last=example_id, status=status)
        if advance:
            progress.update(1)
        return
    if progress_mode == "log":
        total_text = str(total) if total is not None else "?"
        attempt_text = ""
        if attempt is not None and max_attempts is not None:
            attempt_text = " attempt=%d/%d" % (attempt, max_attempts)
        print(
            "sharding %d/%s example_id=%s status=%s ok=%d errors=%d cached=%d retries=%d%s"
            % (index, total_text, example_id, status, written, errors, cached, retries, attempt_text),
            file=sys.stderr,
        )


def _progress_close(progress: Any) -> None:
    """Close a progress bar if one was created."""
    if progress:
        progress.close()


def _print_llm_summary(
    args: argparse.Namespace,
    written: int,
    generated: int,
    cached: int,
    errors: int,
    raw_responses: int,
    retries: int,
    error_types: Counter,
    resume_stats: Optional[ResumeStats] = None,
) -> None:
    """Print the final LLM run summary with artifact paths and top failures."""
    print(
        "written=%d generated=%d cached=%d errors=%d raw_responses=%d retries=%d"
        % (written, generated, cached, errors, raw_responses, retries)
    )
    if error_types:
        print("top_errors:")
        for error_type, count in error_types.most_common(5):
            print("  %s: %d" % (error_type, count))
    print("outputs:")
    print("  shards: %s" % args.out)
    print("  errors: %s" % args.errors)
    if args.raw_responses:
        print("  raw: %s" % args.raw_responses)
    if resume_stats:
        print(
            "resume_cache cached_final=%d cached_tmp=%d cached_content=%d skipped_missing_fingerprint=%d skipped_invalid=%d"
            % (
                resume_stats.cached_final,
                resume_stats.cached_tmp,
                resume_stats.cached_content,
                resume_stats.skipped_missing_fingerprint,
                resume_stats.skipped_invalid,
            )
        )


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
    if isinstance(exc, OpenAIRunError):
        if "OPENAI_API_KEY" in str(exc):
            return "Export OPENAI_API_KEY before running OpenAI-backed decomposition commands."
        if "openai package is not installed" in str(exc):
            return "Run `uv sync --project decomposition` to install decomposition dependencies."
        if "max-tokens" in str(exc) or "max_tokens" in str(exc):
            return "Increase --max-tokens and retry this row."
        return "Check OPENAI_API_KEY, --model, --max-tokens, and the raw response sidecar for the failed row."
    if isinstance(exc, FileNotFoundError):
        return "Check the path exists. For prompts, confirm --segmentation-version has a matching prompts/<version>.txt file."
    if "text not found after offset" in str(exc):
        return "Make the model return exact verbatim atomic-unit text in source order; local code computes offsets."
    if "unit 0" in str(exc) or "zero-based" in str(exc):
        return "Use 1-based atomic unit ids: the first atomic unit is 1, and shard unit_ids must reference those ids."
    if isinstance(exc, LLMIngestError):
        return "Ensure each response has request_id, atomic_units, shards, and verbatim source text."
    if isinstance(exc, ValidationError):
        return "Run the validate command on the artifact and inspect the listed field-level contract failures."
    if "source text is empty" in str(exc):
        return "Drop the empty row or choose a --source-field that contains the raw post text."
    return None


def _clip(value: str, limit: int) -> str:
    """Keep JSONL error sidecars readable."""
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
