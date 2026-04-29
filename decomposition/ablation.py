from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import csv
import hashlib
import json
import os
import sys
import threading

from .align import AlignmentError
from .datasets import DatasetError, DatasetExample, load_dataset
from .deterministic import utc_now
from .llm_io import (
    AtomicUnitsRecord,
    LLMIngestError,
    atomic_record_from_response,
    atomic_record_from_shard_record,
    build_atomic_request,
    read_jsonl,
    validate_atomic_record,
)
from .llm_schema import ATOMIC_UNITS_OUTPUT_SCHEMA
from .openai_io import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_OPENAI_MODEL,
    OpenAIRunError,
    call_openai_atomic,
    create_openai_client,
    openai_atomic_batch_request,
    raise_for_incomplete_response as raise_for_incomplete_openai_response,
    response_from_openai_batch_result,
)
from .schema import ShardRecord, ValidationError, WarningItem, validate_record
from .shard_planner import DEFAULT_SHARD_POLICY, ShardPlanningError, plan_shards


DEFAULT_TARGET_TURNS = (4, 6, 8)
DEFAULT_PROVIDER = "openai"


@dataclass
class AblationPaths:
    """Default artifact paths for one dataset-level ablation run."""

    atomic: Path
    atomic_errors: Path
    raw_responses: Path
    summary: Path
    shards_by_target: Dict[int, Path]


@dataclass
class AtomicCachedRow:
    """A validated atomic cache row plus the artifact source that produced it."""

    record: AtomicUnitsRecord
    source: str


@dataclass
class AtomicCacheStats:
    """Counters explaining cache recovery behavior."""

    cached_final: int = 0
    cached_tmp: int = 0
    cached_raw_response: int = 0
    cached_seed_shards: int = 0
    cached_content: int = 0
    cached_raw_hash: int = 0
    skipped_invalid: int = 0
    skipped_missing_fingerprint: int = 0


@dataclass
class AtomicResumeCache:
    """Validated atomic rows indexed by exact request, content, and explicit seeds."""

    by_request: Dict[str, AtomicCachedRow] = field(default_factory=dict)
    by_content: Dict[str, AtomicCachedRow] = field(default_factory=dict)
    by_raw_hash: Dict[str, AtomicCachedRow] = field(default_factory=dict)
    stats: AtomicCacheStats = field(default_factory=AtomicCacheStats)


@dataclass
class AtomicWorkItem:
    """One dataset row after request construction and cache lookup."""

    index: int
    total: Optional[int]
    example: DatasetExample
    request: Dict[str, Any]
    atomic_request_fingerprint: str
    atomic_content_fingerprint: str
    cached: Optional[AtomicCachedRow] = None
    content_cache_hit: bool = False
    raw_hash_cache_hit: bool = False


@dataclass
class AtomicWorkResult:
    """One completed atomic row, either cached, generated, or failed."""

    item: AtomicWorkItem
    status: str
    record: Optional[AtomicUnitsRecord] = None
    error: Optional[Dict[str, Any]] = None
    raw_responses: int = 0
    retries: int = 0


@dataclass
class AtomicAttemptFailure(Exception):
    """Wrap the final per-row failure with retry metadata."""

    original: Exception
    response: Optional[Dict[str, Any]]
    attempts: int
    raw_responses: int
    retryable: bool
    retry_errors: List[Dict[str, Any]]

    def __str__(self) -> str:
        return str(self.original)


def run_shard_ablation(args: Any) -> int:
    """Run realtime OpenAI atomic extraction and write all requested shard counts."""
    _normalize_ablation_args(args)
    targets = parse_target_turns(args.target_turns)
    paths = ablation_paths(args.out_dir, args.dataset_name, args.provider, args.segmentation_version)
    include_temp = bool(getattr(args, "resume_include_temp", True))
    resume = bool(getattr(args, "resume", True))
    progress_mode = _resolve_progress_mode(getattr(args, "progress", "auto"))
    total = _count_dataset_rows(args.dataset, args.source_field, args.limit) if progress_mode != "off" else None
    items = _atomic_work_items(args, targets, total, paths, include_temp=include_temp, resume=resume)

    has_cache_miss = any(item.cached is None for item in items)
    client = None
    if has_cache_miss:
        client = getattr(args, "_openai_client", None) or create_openai_client(
            max_retries=getattr(args, "openai_max_retries", getattr(args, "provider_max_retries", 2))
        )

    results = _run_atomic_realtime_items(items, args, client, paths, progress_mode, total)
    records, atomic_summary = _write_atomic_outputs(results, paths, resume=resume)
    shard_summary = write_shard_artifacts(
        records,
        target_turns=targets,
        paths=paths,
        run_id=args.run_id,
        policy=getattr(args, "shard_policy", DEFAULT_SHARD_POLICY),
    )
    summary = {
        "type": "shard_ablation_summary",
        "provider": args.provider,
        "dataset_name": args.dataset_name,
        "run_id": args.run_id,
        "segmentation_version": args.segmentation_version,
        "model": args.model,
        "target_turns": list(targets),
        "created_at": utc_now(),
        "atomic": atomic_summary,
        "shards": shard_summary,
        "outputs": _paths_summary(paths),
    }
    _write_summary(paths.summary, summary)
    _print_realtime_summary(paths, targets, atomic_summary, shard_summary)
    return 0


def submit_shard_ablation_batch(args: Any) -> int:
    """Submit OpenAI Batch work for only atomic cache misses."""
    _normalize_ablation_args(args)
    targets = parse_target_turns(args.target_turns)
    paths = ablation_paths(args.out_dir, args.dataset_name, args.provider, args.segmentation_version)
    include_temp = bool(getattr(args, "resume_include_temp", True))
    items = _atomic_work_items(args, targets, total=None, paths=paths, include_temp=include_temp, resume=True)
    input_path = getattr(args, "batch_input", None) or args.batch_state.with_name(args.batch_state.stem + ".input.jsonl")
    _ensure_parent(input_path)
    _ensure_parent(args.batch_state)

    state_requests = []
    sequence = 0
    with input_path.open("w", encoding="utf-8") as input_handle:
        for item in items:
            if item.cached is not None:
                continue
            sequence += 1
            custom_id = _batch_custom_id(sequence, item.atomic_request_fingerprint)
            _write_jsonl(
                input_handle,
                openai_atomic_batch_request(
                    custom_id,
                    item.request,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                ),
            )
            state_requests.append(
                {
                    "custom_id": custom_id,
                    "row_index": item.index,
                    "request": item.request,
                    "atomic_request_fingerprint": item.atomic_request_fingerprint,
                    "atomic_content_fingerprint": item.atomic_content_fingerprint,
                }
            )

    state = _batch_state(args, targets, paths, input_path, state_requests, stage="prepared")
    _write_state(args.batch_state, state)
    batch_id = None
    input_file_id = None
    raw_batch = None
    if state_requests:
        client = getattr(args, "_openai_client", None) or create_openai_client(
            max_retries=getattr(args, "openai_max_retries", getattr(args, "provider_max_retries", 2))
        )
        with input_path.open("rb") as input_file:
            uploaded = client.files.create(file=input_file, purpose="batch")
        uploaded_dict = _object_to_dict(uploaded)
        input_file_id = str(uploaded_dict.get("id") or getattr(uploaded, "id", ""))
        state.update({"stage": "uploaded", "input_file_id": input_file_id, "uploaded_file": uploaded_dict})
        _write_state(args.batch_state, state)
        batch = client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/responses",
            completion_window="24h",
        )
        raw_batch = _object_to_dict(batch)
        batch_id = str(raw_batch.get("id") or getattr(batch, "id", ""))
        state.update({"stage": "submitted", "batch_id": batch_id, "batch": raw_batch})
        _write_state(args.batch_state, state)
    print("batch_id=%s requests=%d state=%s input=%s" % (batch_id or "none", len(state_requests), args.batch_state, input_path))
    return 0


def collect_shard_ablation_batch(args: Any) -> int:
    """Collect OpenAI Batch atomic results and write all requested shard counts."""
    state = json.loads(args.batch_state.read_text(encoding="utf-8"))
    if state.get("type") != "openai_shard_ablation_batch_state":
        raise DatasetError("batch state is not a shard-ablation OpenAI state: %s" % args.batch_state)
    rebuilt = _args_from_batch_state(args, state)
    paths = ablation_paths(rebuilt.out_dir, rebuilt.dataset_name, rebuilt.provider, rebuilt.segmentation_version)
    targets = tuple(int(value) for value in state.get("target_turns", DEFAULT_TARGET_TURNS))
    include_temp = bool(getattr(args, "resume_include_temp", True))
    items = _atomic_work_items(rebuilt, targets, total=None, paths=paths, include_temp=include_temp, resume=True)

    batch_results = {}
    batch_id = state.get("batch_id")
    if batch_id:
        client = getattr(args, "_openai_client", None) or create_openai_client(
            max_retries=getattr(args, "openai_max_retries", state.get("provider_max_retries", 2))
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
                request_state = request_by_custom.get(str(result_dict.get("custom_id", "")))
                if request_state:
                    batch_results[str(request_state["atomic_request_fingerprint"])] = result_dict

    results = _collect_atomic_batch_items(items, rebuilt, paths, batch_results, raw_mode=getattr(args, "raw_responses_mode", "append"))
    records, atomic_summary = _write_atomic_outputs(results, paths, resume=True)
    shard_summary = write_shard_artifacts(
        records,
        target_turns=targets,
        paths=paths,
        run_id=rebuilt.run_id,
        policy=getattr(rebuilt, "shard_policy", DEFAULT_SHARD_POLICY),
    )
    summary = {
        "type": "shard_ablation_summary",
        "provider": rebuilt.provider,
        "dataset_name": rebuilt.dataset_name,
        "run_id": rebuilt.run_id,
        "segmentation_version": rebuilt.segmentation_version,
        "model": rebuilt.model,
        "target_turns": list(targets),
        "created_at": utc_now(),
        "batch_id": batch_id,
        "atomic": atomic_summary,
        "shards": shard_summary,
        "outputs": _paths_summary(paths),
    }
    _write_summary(paths.summary, summary)
    print(
        "written=%d cached=%d errors=%d batch_id=%s"
        % (atomic_summary["written"], atomic_summary["cached"], atomic_summary["errors"], batch_id or "none")
    )
    return 0


def write_shard_artifacts(
    atomic_records: Sequence[AtomicUnitsRecord],
    target_turns: Sequence[int],
    paths: AblationPaths,
    run_id: str,
    policy: str = DEFAULT_SHARD_POLICY,
) -> Dict[str, Dict[str, int]]:
    """Generate deterministic shard files for every target count from atomic cache rows."""
    _validate_shard_policy(policy)
    summary: Dict[str, Dict[str, int]] = {}
    for target in target_turns:
        out_path = paths.shards_by_target[target]
        temp_path = _resume_temp_path(out_path)
        _ensure_parent(out_path)
        counts = {"written": 0, "ok": 0, "ineligible": 0, "errors": 0}
        try:
            with temp_path.open("w", encoding="utf-8") as out_handle:
                for atomic_record in atomic_records:
                    shard_record = _shard_record_for_target(atomic_record, target, run_id, policy)
                    validate_record(shard_record)
                    _write_jsonl(out_handle, shard_record.to_dict())
                    counts["written"] += 1
                    if shard_record.status == "ok":
                        counts["ok"] += 1
                    else:
                        counts["ineligible"] += 1
        except Exception as exc:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
            example_id = getattr(locals().get("atomic_record", None), "example_id", "unknown")
            raise DatasetError(
                "failed to generate k%d shard artifact for example_id=%s: %s" % (target, example_id, exc)
            ) from exc
        os.replace(temp_path, out_path)
        summary["k%d" % target] = counts
    return summary


def parse_target_turns(value: Any) -> Tuple[int, ...]:
    """Parse and validate comma-separated shard counts."""
    if value is None:
        return DEFAULT_TARGET_TURNS
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    else:
        parts = [str(part).strip() for part in value if str(part).strip()]
    targets = tuple(int(part) for part in parts)
    if not targets:
        raise DatasetError("--target-turns must contain at least one target count")
    unsupported = sorted(set(targets).difference(DEFAULT_TARGET_TURNS))
    if unsupported:
        raise DatasetError("--target-turns supports only %s; got unsupported %s" % (DEFAULT_TARGET_TURNS, unsupported))
    return targets


def _validate_shard_policy(policy: str) -> None:
    if policy != DEFAULT_SHARD_POLICY:
        raise DatasetError("unsupported --shard-policy %r; supported policies: %s" % (policy, DEFAULT_SHARD_POLICY))


def ablation_paths(out_dir: Path, dataset_name: str, provider: str, segmentation_version: str = "seg_v1") -> AblationPaths:
    """Return the default dataset-level ablation artifact paths."""
    return AblationPaths(
        atomic=out_dir / ("atomic_units.%s.%s.jsonl" % (dataset_name, provider)),
        atomic_errors=out_dir / ("run_errors.%s.%s.atomic.jsonl" % (dataset_name, provider)),
        raw_responses=out_dir / ("%s_%s_atomic_responses.%s.jsonl" % (segmentation_version, provider, dataset_name)),
        summary=out_dir / ("shard_ablation.%s.%s.summary.json" % (dataset_name, provider)),
        shards_by_target={
            target: out_dir / ("shards.%s.%s.k%d.jsonl" % (dataset_name, provider, target))
            for target in DEFAULT_TARGET_TURNS
        },
    )


def _normalize_ablation_args(args: Any) -> None:
    provider = str(getattr(args, "provider", None) or DEFAULT_PROVIDER)
    if provider != "openai":
        raise DatasetError("shard-ablation currently supports provider=openai")
    args.provider = provider
    if not getattr(args, "model", None):
        args.model = DEFAULT_OPENAI_MODEL
    if not getattr(args, "max_tokens", None):
        args.max_tokens = DEFAULT_MAX_OUTPUT_TOKENS
    if not hasattr(args, "temperature"):
        args.temperature = 0.0
    if not hasattr(args, "segmentation_version"):
        args.segmentation_version = "seg_v1"
    if not hasattr(args, "created_at"):
        args.created_at = None
    if not hasattr(args, "shard_policy"):
        args.shard_policy = DEFAULT_SHARD_POLICY
    _validate_shard_policy(str(args.shard_policy))


def _atomic_work_items(
    args: Any,
    targets: Sequence[int],
    total: Optional[int],
    paths: AblationPaths,
    include_temp: bool,
    resume: bool,
) -> List[AtomicWorkItem]:
    cache = _load_atomic_cache_for_args(args, paths, include_temp=include_temp, resume=resume)
    items: List[AtomicWorkItem] = []
    for index, example in enumerate(load_dataset(args.dataset, args.dataset_name, args.source_field, args.limit), start=1):
        request = build_atomic_request(
            example,
            run_id=args.run_id,
            segmentation_version=args.segmentation_version,
            created_at=args.created_at,
        )
        request_fingerprint = atomic_request_fingerprint(
            request,
            provider=args.provider,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        content_fingerprint = atomic_content_fingerprint(
            request,
            provider=args.provider,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        cached = cache.by_request.get(request_fingerprint)
        content_hit = False
        raw_hit = False
        if cached is None:
            cached = cache.by_content.get(content_fingerprint)
            content_hit = cached is not None
        if cached is None:
            cached = cache.by_raw_hash.get(_raw_hash(example.raw_source_text))
            raw_hit = cached is not None
        items.append(
            AtomicWorkItem(
                index=index,
                total=total,
                example=example,
                request=request,
                atomic_request_fingerprint=request_fingerprint,
                atomic_content_fingerprint=content_fingerprint,
                cached=cached,
                content_cache_hit=content_hit,
                raw_hash_cache_hit=raw_hit,
            )
        )
    return items


def _load_atomic_cache_for_args(args: Any, paths: AblationPaths, include_temp: bool, resume: bool) -> AtomicResumeCache:
    cache = AtomicResumeCache()
    if resume:
        _merge_atomic_cache(cache, _load_atomic_cache(paths.atomic, include_temp=include_temp))
        _merge_atomic_cache(cache, _load_raw_response_cache(paths.raw_responses, args))
    for seed_path in _seed_shards(args):
        _merge_atomic_cache(cache, _load_seed_shards_cache(seed_path))
    return cache


def _load_atomic_cache(path: Path, include_temp: bool = True) -> AtomicResumeCache:
    cache = AtomicResumeCache()
    candidates = [(path, "final")]
    temp = _resume_temp_path(path)
    if include_temp and temp != path:
        candidates.append((temp, "tmp"))
    for candidate, source in candidates:
        if not candidate.exists():
            continue
        for row in _read_jsonl_lenient(candidate, cache.stats):
            try:
                record = AtomicUnitsRecord.from_dict(row)
                validate_atomic_record(record)
            except Exception:
                cache.stats.skipped_invalid += 1
                continue
            _add_atomic_cache_row(cache, record, source=source, allow_raw_hash=False)
    return cache


def _load_raw_response_cache(path: Path, args: Any) -> AtomicResumeCache:
    cache = AtomicResumeCache()
    if not path.exists():
        return cache
    requests_by_id: Dict[str, Dict[str, Any]] = {}
    fingerprints_by_id: Dict[str, Tuple[str, str]] = {}
    for example in load_dataset(args.dataset, args.dataset_name, args.source_field, args.limit):
        request = build_atomic_request(
            example,
            run_id=args.run_id,
            segmentation_version=args.segmentation_version,
            created_at=args.created_at,
        )
        request_fingerprint = atomic_request_fingerprint(
            request,
            provider=args.provider,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        content_fingerprint = atomic_content_fingerprint(
            request,
            provider=args.provider,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        requests_by_id[str(request["request_id"])] = request
        fingerprints_by_id[str(request["request_id"])] = (request_fingerprint, content_fingerprint)
    for row in _read_jsonl_lenient(path, cache.stats):
        request_id = str(row.get("request_id") or "")
        request = requests_by_id.get(request_id)
        if request is None:
            continue
        try:
            raise_for_incomplete_openai_response(row)
            record = atomic_record_from_response(request, row)
            request_fingerprint, content_fingerprint = fingerprints_by_id[request_id]
            record.atomic_request_fingerprint = request_fingerprint
            record.atomic_content_fingerprint = content_fingerprint
            _add_atomic_cache_row(cache, record, source="raw_response", allow_raw_hash=False)
        except Exception:
            cache.stats.skipped_invalid += 1
    return cache


def _load_seed_shards_cache(path: Path) -> AtomicResumeCache:
    cache = AtomicResumeCache()
    if not path.exists():
        raise FileNotFoundError("seed shard artifact not found: %s" % path)
    for row in _read_jsonl_lenient(path, cache.stats):
        try:
            record = atomic_record_from_shard_record(ShardRecord.from_dict(row))
        except Exception:
            cache.stats.skipped_invalid += 1
            continue
        _add_atomic_cache_row(cache, record, source="seed_shards", allow_raw_hash=True)
    return cache


def _merge_atomic_cache(target: AtomicResumeCache, source: AtomicResumeCache) -> None:
    target.by_request.update(source.by_request)
    target.by_content.update(source.by_content)
    target.by_raw_hash.update(source.by_raw_hash)
    target.stats.skipped_invalid += source.stats.skipped_invalid
    target.stats.skipped_missing_fingerprint += source.stats.skipped_missing_fingerprint


def _add_atomic_cache_row(
    cache: AtomicResumeCache,
    record: AtomicUnitsRecord,
    source: str,
    allow_raw_hash: bool,
) -> None:
    cached = AtomicCachedRow(record=record, source=source)
    if record.atomic_request_fingerprint:
        cache.by_request[str(record.atomic_request_fingerprint)] = cached
    else:
        cache.stats.skipped_missing_fingerprint += 1
    if record.atomic_content_fingerprint:
        cache.by_content[str(record.atomic_content_fingerprint)] = cached
    if allow_raw_hash:
        cache.by_raw_hash[_raw_hash(record.raw_source_text)] = cached


def _run_atomic_realtime_items(
    items: Sequence[AtomicWorkItem],
    args: Any,
    client: Any,
    paths: AblationPaths,
    progress_mode: str,
    total: Optional[int],
) -> List[AtomicWorkResult]:
    max_attempts = max(1, int(getattr(args, "llm_retries", 1)) + 1)
    concurrency = max(1, int(getattr(args, "concurrency", 1)))
    raw_mode = "a" if bool(getattr(args, "resume", True)) else "w"
    _ensure_parent(paths.raw_responses)
    raw_handle = paths.raw_responses.open(raw_mode, encoding="utf-8")
    raw_lock = threading.Lock()
    progress = _progress_bar(progress_mode, total)
    progress_lock = threading.Lock()
    try:
        results = list(
            _run_atomic_items_ordered(
                items,
                args,
                client,
                max_attempts=max_attempts,
                raw_handle=raw_handle,
                raw_lock=raw_lock,
                progress=progress,
                progress_mode=progress_mode,
                progress_lock=progress_lock,
                concurrency=concurrency,
            )
        )
    finally:
        raw_handle.close()
        _progress_close(progress)
    return results


def _run_atomic_items_ordered(
    items: Sequence[AtomicWorkItem],
    args: Any,
    client: Any,
    max_attempts: int,
    raw_handle: Any,
    raw_lock: threading.Lock,
    progress: Any,
    progress_mode: str,
    progress_lock: threading.Lock,
    concurrency: int,
) -> Iterable[AtomicWorkResult]:
    if concurrency <= 1:
        written = cached = errors = retries = 0
        for item in items:
            result = _run_one_atomic_item(item, args, client, max_attempts, raw_handle, raw_lock, progress, progress_mode, progress_lock)
            written, cached, errors, retries = _progress_counts(result, written, cached, errors, retries)
            _progress_update(progress, progress_mode, item.index, item.total, item.example.example_id, result.status, written, errors, cached, retries)
            yield result
        return

    completed: Dict[int, AtomicWorkResult] = {}
    next_index = 1
    futures = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for item in items:
            if item.cached is not None:
                completed[item.index] = _cached_atomic_result(item)
            else:
                futures.append(
                    executor.submit(
                        _run_one_atomic_item,
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


def _run_one_atomic_item(
    item: AtomicWorkItem,
    args: Any,
    client: Any,
    max_attempts: int,
    raw_handle: Any,
    raw_lock: Optional[threading.Lock],
    progress: Any,
    progress_mode: str,
    progress_lock: Optional[threading.Lock],
) -> AtomicWorkResult:
    if item.cached is not None:
        return _cached_atomic_result(item)
    response: Optional[Dict[str, Any]] = None
    try:
        record, response, attempt_count, _retryable, _retry_errors = _call_openai_atomic_with_retries(
            item,
            args,
            client,
            max_attempts=max_attempts,
            raw_handle=raw_handle,
            raw_lock=raw_lock,
            progress=progress,
            progress_mode=progress_mode,
            progress_lock=progress_lock,
        )
        record.atomic_request_fingerprint = item.atomic_request_fingerprint
        record.atomic_content_fingerprint = item.atomic_content_fingerprint
        return AtomicWorkResult(
            item=item,
            status="ok",
            record=record,
            raw_responses=attempt_count,
            retries=max(0, attempt_count - 1),
        )
    except Exception as exc:
        error, raw_count, retry_count = _atomic_error_from_exception(item, args, exc, response)
        return AtomicWorkResult(item=item, status="error", error=error, raw_responses=raw_count, retries=retry_count)


def _call_openai_atomic_with_retries(
    item: AtomicWorkItem,
    args: Any,
    client: Any,
    max_attempts: int,
    raw_handle: Any,
    raw_lock: Optional[threading.Lock],
    progress: Any,
    progress_mode: str,
    progress_lock: Optional[threading.Lock],
) -> Tuple[AtomicUnitsRecord, Dict[str, Any], int, bool, List[Dict[str, Any]]]:
    retry_errors: List[Dict[str, Any]] = []
    raw_response_count = 0
    response: Optional[Dict[str, Any]] = None
    for attempt in range(1, max_attempts + 1):
        response = None
        try:
            response = call_openai_atomic(
                item.request,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                client=client,
                created_at=args.created_at,
            )
            response = _atomic_response_with_attempt_metadata(response, item, attempt, max_attempts)
            if raw_handle:
                if raw_lock:
                    with raw_lock:
                        _write_jsonl(raw_handle, response)
                else:
                    _write_jsonl(raw_handle, response)
            raw_response_count += 1
            raise_for_incomplete_openai_response(response)
            record = atomic_record_from_response(item.request, response)
            return record, response, attempt, False, retry_errors
        except Exception as exc:
            retryable = _retryable_atomic_failure(exc)
            retry_errors.append(_retry_error_item(exc, attempt, response))
            if not retryable or attempt >= max_attempts:
                raise AtomicAttemptFailure(
                    exc,
                    response,
                    attempts=attempt,
                    raw_responses=raw_response_count,
                    retryable=retryable,
                    retry_errors=retry_errors,
                ) from exc
            if progress_lock:
                with progress_lock:
                    _progress_update(progress, progress_mode, item.index, item.total, item.example.example_id, "retry", 0, 0, 0, attempt, advance=False)
            else:
                _progress_update(progress, progress_mode, item.index, item.total, item.example.example_id, "retry", 0, 0, 0, attempt, advance=False)
    raise LLMIngestError("unreachable atomic retry state")


def _cached_atomic_result(item: AtomicWorkItem) -> AtomicWorkResult:
    record = _rewrite_atomic_record_for_request(
        item.cached.record,
        item.request,
        item.atomic_request_fingerprint,
        item.atomic_content_fingerprint,
    )
    return AtomicWorkResult(item=item, status="cached", record=record)


def _rewrite_atomic_record_for_request(
    record: AtomicUnitsRecord,
    request: Dict[str, Any],
    request_fingerprint: str,
    content_fingerprint: str,
) -> AtomicUnitsRecord:
    rewritten = AtomicUnitsRecord.from_dict(record.to_dict())
    rewritten.example_id = str(request["example_id"])
    rewritten.dataset_name = str(request["dataset_name"])
    rewritten.source_text_field = str(request["source_text_field"])
    rewritten.run_id = str(request["run_id"])
    rewritten.segmentation_version = str(request["segmentation_version"])
    rewritten.created_at = str(request.get("created_at") or record.created_at or utc_now())
    rewritten.raw_source_text = str(request["raw_source_text"])
    rewritten.atomic_request_fingerprint = request_fingerprint
    rewritten.atomic_content_fingerprint = content_fingerprint
    validate_atomic_record(rewritten)
    return rewritten


def _collect_atomic_batch_items(
    items: Sequence[AtomicWorkItem],
    args: Any,
    paths: AblationPaths,
    batch_results: Dict[str, Dict[str, Any]],
    raw_mode: str,
) -> List[AtomicWorkResult]:
    _ensure_parent(paths.raw_responses)
    raw_handle = None
    results: List[AtomicWorkResult] = []
    try:
        for item in items:
            if item.cached is not None:
                results.append(_cached_atomic_result(item))
                continue
            batch_result = batch_results.get(item.atomic_request_fingerprint)
            if not batch_result:
                exc = OpenAIRunError("batch result missing for request %s" % item.atomic_request_fingerprint)
                error = _run_error(item.example, args.run_id, args.segmentation_version, "openai_atomic_batch", exc)
                error["atomic_request_fingerprint"] = item.atomic_request_fingerprint
                error["atomic_content_fingerprint"] = item.atomic_content_fingerprint
                results.append(AtomicWorkResult(item=item, status="error", error=error))
                continue
            if _openai_batch_result_succeeded(batch_result):
                response = response_from_openai_batch_result(item.request, batch_result, model=args.model, created_at=None)
                response = _atomic_response_with_attempt_metadata(response, item, 1, 1)
                if raw_handle is None:
                    raw_handle = paths.raw_responses.open("a" if raw_mode == "append" else "w", encoding="utf-8")
                _write_jsonl(raw_handle, response)
                try:
                    raise_for_incomplete_openai_response(response)
                    record = atomic_record_from_response(item.request, response)
                    record.atomic_request_fingerprint = item.atomic_request_fingerprint
                    record.atomic_content_fingerprint = item.atomic_content_fingerprint
                    results.append(AtomicWorkResult(item=item, status="ok", record=record, raw_responses=1))
                except Exception as exc:
                    error = _run_error(item.example, args.run_id, args.segmentation_version, "openai_atomic_batch", exc)
                    error["atomic_request_fingerprint"] = item.atomic_request_fingerprint
                    error["atomic_content_fingerprint"] = item.atomic_content_fingerprint
                    error["retryable"] = _retryable_atomic_failure(exc)
                    error["retry_errors"] = [_retry_error_item(exc, 1, response)]
                    results.append(AtomicWorkResult(item=item, status="error", error=error, raw_responses=1))
            else:
                provider_error = _openai_batch_error(batch_result)
                exc = OpenAIRunError("batch result provider_error")
                error = _run_error(item.example, args.run_id, args.segmentation_version, "openai_atomic_batch", exc)
                error["atomic_request_fingerprint"] = item.atomic_request_fingerprint
                error["atomic_content_fingerprint"] = item.atomic_content_fingerprint
                error["provider_error"] = provider_error
                error["batch_custom_id"] = batch_result.get("custom_id")
                results.append(AtomicWorkResult(item=item, status="error", error=error))
    finally:
        if raw_handle is not None:
            raw_handle.close()
    return results


def _write_atomic_outputs(
    results: Sequence[AtomicWorkResult],
    paths: AblationPaths,
    resume: bool,
) -> Tuple[List[AtomicUnitsRecord], Dict[str, Any]]:
    _ensure_parent(paths.atomic)
    _ensure_parent(paths.atomic_errors)
    out_path = _resume_temp_path(paths.atomic) if resume else paths.atomic
    records: List[AtomicUnitsRecord] = []
    counts = Counter()
    retries = 0
    raw_responses = 0
    with out_path.open("w", encoding="utf-8") as out_handle, paths.atomic_errors.open("w", encoding="utf-8") as err_handle:
        for result in results:
            if result.record is not None:
                _write_jsonl(out_handle, result.record.to_dict())
                records.append(result.record)
                counts["written"] += 1
                if result.status == "cached":
                    _count_atomic_cache_hit(counts, result)
                else:
                    counts[result.status] += 1
            if result.error is not None:
                _write_jsonl(err_handle, result.error)
                counts["errors"] += 1
                counts["error_%s" % result.error.get("error_type", "unknown")] += 1
            retries += result.retries
            raw_responses += result.raw_responses
    if resume:
        os.replace(out_path, paths.atomic)
    summary = {
        "written": counts["written"],
        "generated": counts["ok"],
        "cached": counts["cached"],
        "errors": counts["errors"],
        "raw_responses": raw_responses,
        "retries": retries,
        "cache_sources": {
            "final": counts["cache_final"],
            "tmp": counts["cache_tmp"],
            "raw_response": counts["cache_raw_response"],
            "seed_shards": counts["cache_seed_shards"],
            "content": counts["cache_content"],
            "raw_hash": counts["cache_raw_hash"],
        },
        "error_types": {key[len("error_") :]: value for key, value in counts.items() if key.startswith("error_")},
    }
    return records, summary


def _count_atomic_cache_hit(counts: Counter, result: AtomicWorkResult) -> None:
    if result.status != "cached" or result.item.cached is None:
        return
    counts["cached"] += 1
    source = result.item.cached.source
    counts["cache_%s" % source] += 1
    if result.item.content_cache_hit:
        counts["cache_content"] += 1
    if result.item.raw_hash_cache_hit:
        counts["cache_raw_hash"] += 1


def _shard_record_for_target(
    atomic_record: AtomicUnitsRecord,
    target_turns: int,
    run_id: str,
    policy: str,
) -> ShardRecord:
    warnings = [WarningItem.from_dict(warning.to_dict()) for warning in atomic_record.warnings]
    if len(atomic_record.atomic_units) < target_turns:
        warnings.append(
            WarningItem(
                code="too_few_atomic_units_for_target",
                field="status",
                severity="warning",
                message="Only %d atomic units are available for target_turns=%d."
                % (len(atomic_record.atomic_units), target_turns),
            )
        )
        return ShardRecord(
            example_id=atomic_record.example_id,
            dataset_name=atomic_record.dataset_name,
            source_text_field=atomic_record.source_text_field,
            run_id=run_id,
            segmentation_version=atomic_record.segmentation_version,
            segmenter_model=atomic_record.segmenter_model + "+" + policy,
            created_at=atomic_record.created_at,
            raw_source_text=atomic_record.raw_source_text,
            target_turns=target_turns,
            status="ineligible_target_shards",
            atomic_units=list(atomic_record.atomic_units),
            shards=[],
            warnings=warnings,
            request_fingerprint=_shard_request_fingerprint(atomic_record, target_turns, policy),
            content_fingerprint=_shard_content_fingerprint(atomic_record, target_turns, policy),
        )
    shards = plan_shards(atomic_record.raw_source_text, atomic_record.atomic_units, target_turns, policy=policy)
    return ShardRecord(
        example_id=atomic_record.example_id,
        dataset_name=atomic_record.dataset_name,
        source_text_field=atomic_record.source_text_field,
        run_id=run_id,
        segmentation_version=atomic_record.segmentation_version,
        segmenter_model=atomic_record.segmenter_model + "+" + policy,
        created_at=atomic_record.created_at,
        raw_source_text=atomic_record.raw_source_text,
        target_turns=target_turns,
        status="ok",
        atomic_units=list(atomic_record.atomic_units),
        shards=shards,
        warnings=warnings,
        request_fingerprint=_shard_request_fingerprint(atomic_record, target_turns, policy),
        content_fingerprint=_shard_content_fingerprint(atomic_record, target_turns, policy),
    )


def atomic_request_fingerprint(
    request: Dict[str, Any],
    provider: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    components = {
        "dataset_name": str(request.get("dataset_name", "")),
        "example_id": str(request.get("example_id", "")),
        "source_text_field": str(request.get("source_text_field", "")),
        "run_id": str(request.get("run_id", "")),
        "segmentation_version": str(request.get("segmentation_version", "")),
        "raw_source_text_sha256": _sha256_text(str(request.get("raw_source_text", ""))),
        "prompt_sha256": _sha256_text(_prompt_text_from_request(request)),
        "output_schema_sha256": _sha256_json(ATOMIC_UNITS_OUTPUT_SCHEMA),
        "provider": provider,
        "model": model,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    return "sha256:" + _sha256_json(components)


def atomic_content_fingerprint(
    request: Dict[str, Any],
    provider: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    components = {
        "raw_source_text_sha256": _sha256_text(str(request.get("raw_source_text", ""))),
        "prompt_sha256": _sha256_text(_prompt_text_from_request(request)),
        "output_schema_sha256": _sha256_json(ATOMIC_UNITS_OUTPUT_SCHEMA),
        "provider": provider,
        "model": model,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    return "sha256:" + _sha256_json(components)


def _shard_request_fingerprint(atomic_record: AtomicUnitsRecord, target_turns: int, policy: str) -> str:
    components = {
        "atomic_request_fingerprint": atomic_record.atomic_request_fingerprint,
        "run_id": atomic_record.run_id,
        "target_turns": target_turns,
        "policy": policy,
    }
    return "sha256:" + _sha256_json(components)


def _shard_content_fingerprint(atomic_record: AtomicUnitsRecord, target_turns: int, policy: str) -> str:
    components = {
        "atomic_content_fingerprint": atomic_record.atomic_content_fingerprint,
        "raw_source_text_sha256": _sha256_text(atomic_record.raw_source_text),
        "target_turns": target_turns,
        "policy": policy,
    }
    return "sha256:" + _sha256_json(components)


def _batch_state(
    args: Any,
    targets: Sequence[int],
    paths: AblationPaths,
    input_path: Path,
    state_requests: Sequence[Dict[str, Any]],
    stage: str,
) -> Dict[str, Any]:
    return {
        "type": "openai_shard_ablation_batch_state",
        "stage": stage,
        "provider": args.provider,
        "created_at": utc_now(),
        "batch_id": None,
        "input_file_id": None,
        "input_file": str(input_path),
        "request_count": len(state_requests),
        "requests": list(state_requests),
        "dataset": str(args.dataset),
        "dataset_name": args.dataset_name,
        "source_field": args.source_field,
        "run_id": args.run_id,
        "segmentation_version": args.segmentation_version,
        "created_at_override": args.created_at,
        "limit": args.limit,
        "out_dir": str(args.out_dir),
        "target_turns": list(targets),
        "seed_shards": [str(path) for path in _seed_shards(args)],
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "provider_max_retries": getattr(args, "provider_max_retries", 2),
        "openai_max_retries": getattr(args, "openai_max_retries", getattr(args, "provider_max_retries", 2)),
        "shard_policy": getattr(args, "shard_policy", DEFAULT_SHARD_POLICY),
        "outputs": _paths_summary(paths),
    }


def _args_from_batch_state(args: Any, state: Dict[str, Any]) -> Any:
    class RebuiltArgs:
        pass

    rebuilt = RebuiltArgs()
    rebuilt.dataset = Path(str(state["dataset"]))
    rebuilt.dataset_name = str(state["dataset_name"])
    rebuilt.source_field = str(state["source_field"])
    rebuilt.run_id = str(state["run_id"])
    rebuilt.segmentation_version = str(state.get("segmentation_version") or "seg_v1")
    rebuilt.created_at = state.get("created_at_override")
    rebuilt.limit = state.get("limit")
    rebuilt.out_dir = Path(str(state["out_dir"]))
    rebuilt.target_turns = ",".join(str(value) for value in state.get("target_turns", DEFAULT_TARGET_TURNS))
    rebuilt.provider = str(state.get("provider") or "openai")
    rebuilt.model = str(state.get("model") or DEFAULT_OPENAI_MODEL)
    rebuilt.max_tokens = int(state.get("max_tokens") or DEFAULT_MAX_OUTPUT_TOKENS)
    rebuilt.temperature = float(state.get("temperature") or 0.0)
    rebuilt.seed_shards = [Path(str(path)) for path in state.get("seed_shards", [])]
    rebuilt.resume = True
    rebuilt.resume_include_temp = bool(getattr(args, "resume_include_temp", True))
    rebuilt.provider_max_retries = int(state.get("provider_max_retries") or 2)
    rebuilt.openai_max_retries = int(state.get("openai_max_retries") or 2)
    rebuilt.shard_policy = str(state.get("shard_policy") or DEFAULT_SHARD_POLICY)
    return rebuilt


def _openai_batch_output_rows(client: Any, output_file_id: str) -> Iterable[Dict[str, Any]]:
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


def _openai_batch_result_succeeded(batch_result: Dict[str, Any]) -> bool:
    response = batch_result.get("response")
    if not isinstance(response, dict):
        return False
    try:
        status_code = int(response.get("status_code", 0))
    except (TypeError, ValueError):
        status_code = 0
    return 200 <= status_code < 300 and response.get("body") is not None


def _openai_batch_error(batch_result: Dict[str, Any]) -> Any:
    if batch_result.get("error") is not None:
        return batch_result.get("error")
    response = batch_result.get("response")
    if isinstance(response, dict):
        return response.get("body") or response
    return batch_result


def _batch_custom_id(sequence: int, request_fingerprint: str) -> str:
    digest = request_fingerprint.split(":", 1)[-1]
    return "atomic_%06d_%s" % (sequence, digest[:40])


def _atomic_response_with_attempt_metadata(
    response: Dict[str, Any],
    item: AtomicWorkItem,
    attempt: int,
    max_attempts: int,
) -> Dict[str, Any]:
    annotated = dict(response)
    annotated["attempt"] = attempt
    annotated["max_attempts"] = max_attempts
    annotated["atomic_request_fingerprint"] = item.atomic_request_fingerprint
    annotated["atomic_content_fingerprint"] = item.atomic_content_fingerprint
    annotated["request_fingerprint"] = item.atomic_request_fingerprint
    annotated["content_fingerprint"] = item.atomic_content_fingerprint
    return annotated


def _atomic_error_from_exception(
    item: AtomicWorkItem,
    args: Any,
    exc: Exception,
    response: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], int, int]:
    final_exc = exc
    raw_responses = 0
    retries = 0
    if isinstance(exc, AtomicAttemptFailure):
        final_exc = exc.original
        response = exc.response
        raw_responses = exc.raw_responses
        retries = max(0, exc.attempts - 1)
    error = _run_error(item.example, args.run_id, args.segmentation_version, "openai_atomic", final_exc)
    error["request_id"] = item.request.get("request_id")
    error["atomic_request_fingerprint"] = item.atomic_request_fingerprint
    error["atomic_content_fingerprint"] = item.atomic_content_fingerprint
    if response:
        error["provider_response_id"] = response.get("provider_response_id")
    if isinstance(exc, AtomicAttemptFailure):
        error["attempts"] = exc.attempts
        error["retryable"] = exc.retryable
        error["retry_errors"] = exc.retry_errors
    return error, raw_responses, retries


def _retryable_atomic_failure(exc: Exception) -> bool:
    if isinstance(exc, (LLMIngestError, AlignmentError, ValidationError)):
        return True
    if isinstance(exc, OpenAIRunError):
        if "contained no text content" in str(exc):
            return True
        return False
    return False


def _retry_error_item(exc: Exception, attempt: int, response: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "attempt": attempt,
        "error_type": exc.__class__.__name__,
        "message": _clip(str(exc), 500),
    }
    if response:
        item["provider_response_id"] = response.get("provider_response_id")
    return item


def _run_error(example: DatasetExample, run_id: str, segmentation_version: str, stage: str, exc: Exception) -> Dict[str, Any]:
    error = {
        "example_id": example.example_id,
        "dataset_name": example.dataset_name,
        "source_text_field": example.source_text_field,
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


def _hint_for_exception(exc: Exception) -> Optional[str]:
    if isinstance(exc, DatasetError):
        message = str(exc)
        if "--target-turns" in message:
            return "Use supported shard counts: 4,6,8."
        if "--shard-policy" in message:
            return "Use the default shard planner policy natural_dp_v1."
        if "failed to generate k" in message:
            return "Inspect the atomic cache row, target_turns, and deterministic shard planner policy."
        return "Check that --dataset points to the intended CSV and --source-field matches one header exactly."
    if isinstance(exc, OpenAIRunError):
        if "OPENAI_API_KEY" in str(exc):
            return "Export OPENAI_API_KEY before running OpenAI-backed decomposition commands."
        if "max-tokens" in str(exc) or "max_tokens" in str(exc):
            return "Increase --max-tokens and retry this row."
        return "Check OPENAI_API_KEY, --model, --max-tokens, and the raw response sidecar for the failed row."
    if isinstance(exc, (LLMIngestError, AlignmentError)):
        return "Make the model return exact verbatim atomic-unit text in source order; local code computes offsets."
    if isinstance(exc, ValidationError):
        return "Inspect the field-level atomic cache validation failure and regenerate that row."
    if isinstance(exc, ShardPlanningError):
        return "Check target_turns and atomic-unit coverage for this row."
    return None


def _read_jsonl_lenient(path: Path, stats: AtomicCacheStats) -> Iterable[Dict[str, Any]]:
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


def _seed_shards(args: Any) -> List[Path]:
    raw = getattr(args, "seed_shards", None) or []
    if isinstance(raw, (str, Path)):
        return [Path(str(raw))]
    return [Path(str(path)) for path in raw]


def _prompt_text_from_request(request: Dict[str, Any]) -> str:
    prompt_parts = []
    for message in request.get("messages", []):
        if isinstance(message, dict) and str(message.get("role", "")) in ("system", "developer"):
            prompt_parts.append(str(message.get("content", "")))
    return "\n\n".join(prompt_parts)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_text(encoded)


def _raw_hash(value: str) -> str:
    return _sha256_text(value)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_jsonl(handle: Any, data: Dict[str, Any]) -> None:
    handle.write(json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n")
    if hasattr(handle, "flush"):
        handle.flush()


def _resume_temp_path(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def _write_state(path: Path, state: Dict[str, Any]) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_summary(path: Path, summary: Dict[str, Any]) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _paths_summary(paths: AblationPaths) -> Dict[str, Any]:
    return {
        "atomic_units": str(paths.atomic),
        "atomic_errors": str(paths.atomic_errors),
        "raw_responses": str(paths.raw_responses),
        "summary": str(paths.summary),
        "shards": {str(target): str(path) for target, path in sorted(paths.shards_by_target.items())},
    }


def _object_to_dict(value: Any) -> Dict[str, Any]:
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


def _count_dataset_rows(dataset_path: Path, source_text_field: str, limit: Optional[int]) -> int:
    with dataset_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise DatasetError("dataset has no header: %s" % dataset_path)
        if source_text_field not in reader.fieldnames:
            raise DatasetError("source field %r not found in %s; fields=%s" % (source_text_field, dataset_path, reader.fieldnames))
        count = 0
        for count, _row in enumerate(reader, start=1):
            if limit is not None and count >= limit:
                return limit
        return count


def _resolve_progress_mode(progress: str) -> str:
    if progress != "auto":
        return progress
    return "bar" if sys.stderr.isatty() else "log"


def _progress_bar(progress_mode: str, total: Optional[int]) -> Any:
    if progress_mode != "bar":
        return None
    from tqdm import tqdm

    return tqdm(total=total, desc="Atomic units", unit="post", file=sys.stderr)


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
    advance: bool = True,
) -> None:
    if progress_mode == "off":
        return
    if progress_mode == "bar" and progress:
        progress.set_postfix(ok=written, errors=errors, cached=cached, retries=retries, last=example_id, status=status)
        if advance:
            progress.update(1)
        return
    if progress_mode == "log":
        total_text = str(total) if total is not None else "?"
        print(
            "atomic %d/%s example_id=%s status=%s ok=%d errors=%d cached=%d retries=%d"
            % (index, total_text, example_id, status, written, errors, cached, retries),
            file=sys.stderr,
        )


def _progress_close(progress: Any) -> None:
    if progress:
        progress.close()


def _progress_counts(
    result: AtomicWorkResult,
    written: int,
    cached: int,
    errors: int,
    retries: int,
) -> Tuple[int, int, int, int]:
    if result.record is not None:
        written += 1
    if result.status == "cached":
        cached += 1
    if result.error is not None:
        errors += 1
    retries += result.retries
    return written, cached, errors, retries


def _print_realtime_summary(
    paths: AblationPaths,
    targets: Sequence[int],
    atomic_summary: Dict[str, Any],
    shard_summary: Dict[str, Dict[str, int]],
) -> None:
    print(
        "atomic_written=%d generated=%d cached=%d errors=%d raw_responses=%d retries=%d"
        % (
            atomic_summary["written"],
            atomic_summary["generated"],
            atomic_summary["cached"],
            atomic_summary["errors"],
            atomic_summary["raw_responses"],
            atomic_summary["retries"],
        )
    )
    for target in targets:
        counts = shard_summary["k%d" % target]
        print(
            "k%d written=%d ok=%d ineligible=%d errors=%d"
            % (target, counts["written"], counts["ok"], counts["ineligible"], counts["errors"])
        )
    print("outputs:")
    print("  atomic: %s" % paths.atomic)
    print("  errors: %s" % paths.atomic_errors)
    print("  raw: %s" % paths.raw_responses)
    for target in targets:
        print("  k%d: %s" % (target, paths.shards_by_target[target]))
    print("  summary: %s" % paths.summary)


def _clip(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."
