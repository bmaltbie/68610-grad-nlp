from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import hashlib
import json
import os

from .datasets import DatasetError
from .deterministic import utc_now
from .schema import SUPPORTED_TARGET_TURNS, validate_record_dict


DEFAULT_COHORT_NAME = "eligible_all"
CONSISTENCY_FIELDS = (
    "dataset_name",
    "source_text_field",
    "run_id",
    "segmentation_version",
    "segmenter_model",
    "raw_source_text",
    "normalized_source_text",
    "gold_label",
    "atomic_units",
    "warnings",
)


@dataclass
class TargetShardRows:
    """Validated shard rows for one target count, indexed by example id."""

    target_turns: int
    path: Path
    rows: List[Dict[str, Any]]
    by_id: Dict[str, Dict[str, Any]]
    status_counts: Counter


def run_shard_ablation_cohort(args: Any) -> int:
    """Write matched-cohort shard artifacts from existing k-target shard files."""
    targets = parse_target_turns(getattr(args, "target_turns", None))
    summary = write_eligible_all_cohort(
        artifacts_dir=Path(args.artifacts_dir),
        out_dir=Path(args.out_dir),
        dataset_name=str(args.dataset_name),
        provider=str(getattr(args, "provider", "openai")),
        target_turns=targets,
        allow_empty=bool(getattr(args, "allow_empty", False)),
    )
    print(
        "cohort=%s retained=%d target_turns=%s out_dir=%s"
        % (
            summary["cohort"]["name"],
            summary["cohort"]["retained"],
            ",".join(str(target) for target in targets),
            args.out_dir,
        )
    )
    return 0


def write_eligible_all_cohort(
    artifacts_dir: Path,
    out_dir: Path,
    dataset_name: str,
    provider: str,
    target_turns: Sequence[int],
    allow_empty: bool = False,
) -> Dict[str, Any]:
    """Filter existing shard artifacts to examples with ok rows for every target."""
    targets = tuple(int(target) for target in target_turns)
    if not targets:
        raise DatasetError("--target-turns must contain at least one target count")

    loaded = [_load_target_rows(_input_shard_path(artifacts_dir, dataset_name, provider, target), target) for target in targets]
    _validate_same_example_universe(loaded)

    ok_by_target = {item.target_turns: _ok_ids(item) for item in loaded}
    retained_ids = set.intersection(*(ids for ids in ok_by_target.values())) if ok_by_target else set()
    if not retained_ids and not allow_empty:
        raise DatasetError(
            "eligible-all cohort is empty for %s %s targets=%s; pass --allow-empty to write empty outputs"
            % (dataset_name, provider, ",".join(str(target) for target in targets))
        )

    canonical = min(loaded, key=lambda item: item.target_turns)
    retained_order = [example_id for example_id in _row_ids(canonical.rows) if example_id in retained_ids]
    _validate_retained_consistency(loaded, retained_order)

    out_dir.mkdir(parents=True, exist_ok=True)
    cohort_slug = _cohort_slug(targets)
    output_paths = {
        item.target_turns: _output_shard_path(out_dir, dataset_name, provider, item.target_turns, cohort_slug)
        for item in loaded
    }
    summary_path = out_dir / ("cohort.%s.%s.%s.summary.json" % (dataset_name, provider, cohort_slug))

    written_counts = {}
    for item in loaded:
        output_path = output_paths[item.target_turns]
        written_counts[item.target_turns] = _write_filtered_rows(output_path, item, retained_order)

    summary = _summary(
        dataset_name=dataset_name,
        provider=provider,
        target_turns=targets,
        loaded=loaded,
        retained_order=retained_order,
        ok_by_target=ok_by_target,
        output_paths=output_paths,
        summary_path=summary_path,
        written_counts=written_counts,
        cohort_slug=cohort_slug,
    )
    _write_json(summary_path, summary)
    return summary


def parse_target_turns(value: Any) -> Tuple[int, ...]:
    """Parse and validate comma-separated target counts for cohort export."""
    if value is None:
        parts = ["4", "6", "8"]
    elif isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    else:
        parts = [str(part).strip() for part in value if str(part).strip()]
    if not parts:
        raise DatasetError("--target-turns must contain at least one target count")
    try:
        raw_targets = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise DatasetError("--target-turns must be comma-separated integers") from exc
    unsupported = sorted(set(raw_targets).difference(SUPPORTED_TARGET_TURNS))
    if unsupported:
        raise DatasetError(
            "--target-turns supports only %s; got unsupported %s"
            % (tuple(sorted(SUPPORTED_TARGET_TURNS)), unsupported)
        )
    if len(set(raw_targets)) != len(raw_targets):
        raise DatasetError("--target-turns must not repeat counts")
    return tuple(sorted(raw_targets))


def _load_target_rows(path: Path, target_turns: int) -> TargetShardRows:
    if not path.exists():
        raise DatasetError("missing shard artifact: %s" % path)

    rows = []
    by_id: Dict[str, Dict[str, Any]] = {}
    status_counts: Counter = Counter()
    for line_number, row in _read_jsonl(path):
        validate_record_dict(row)
        observed_target = int(row.get("target_turns", 0))
        if observed_target != target_turns:
            raise DatasetError(
                "%s:%d expected target_turns=%d, got %d"
                % (path, line_number, target_turns, observed_target)
            )
        example_id = str(row.get("example_id", ""))
        if example_id in by_id:
            raise DatasetError("%s contains duplicate example_id=%s" % (path, example_id))
        by_id[example_id] = row
        rows.append(row)
        status_counts[str(row.get("status", ""))] += 1
    return TargetShardRows(target_turns=target_turns, path=path, rows=rows, by_id=by_id, status_counts=status_counts)


def _read_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError("%s:%d invalid JSON: %s" % (path, line_number, exc)) from exc
            if not isinstance(row, dict):
                raise DatasetError("%s:%d expected one JSON object per line" % (path, line_number))
            yield line_number, row


def _validate_same_example_universe(loaded: Sequence[TargetShardRows]) -> None:
    if not loaded:
        return
    expected = set(loaded[0].by_id)
    for item in loaded[1:]:
        observed = set(item.by_id)
        if observed == expected:
            continue
        missing = sorted(expected.difference(observed))[:5]
        extra = sorted(observed.difference(expected))[:5]
        raise DatasetError(
            "%s has a different example_id universe than %s; missing=%s extra=%s"
            % (item.path, loaded[0].path, missing, extra)
        )


def _validate_retained_consistency(loaded: Sequence[TargetShardRows], retained_order: Sequence[str]) -> None:
    if not loaded:
        return
    base = loaded[0]
    for example_id in retained_order:
        base_row = base.by_id[example_id]
        for item in loaded[1:]:
            row = item.by_id[example_id]
            for field in CONSISTENCY_FIELDS:
                if row.get(field) != base_row.get(field):
                    raise DatasetError(
                        "example_id=%s differs across target files for field=%s (%s vs %s)"
                        % (example_id, field, base.path, item.path)
                    )


def _ok_ids(item: TargetShardRows) -> set:
    return {example_id for example_id, row in item.by_id.items() if row.get("status") == "ok"}


def _row_ids(rows: Sequence[Dict[str, Any]]) -> List[str]:
    return [str(row.get("example_id", "")) for row in rows]


def _write_filtered_rows(path: Path, item: TargetShardRows, retained_order: Sequence[str]) -> int:
    temp_path = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            for example_id in retained_order:
                row = item.by_id[example_id]
                validate_record_dict(row)
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
    except Exception:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise
    os.replace(temp_path, path)
    return count


def _summary(
    dataset_name: str,
    provider: str,
    target_turns: Sequence[int],
    loaded: Sequence[TargetShardRows],
    retained_order: Sequence[str],
    ok_by_target: Dict[int, set],
    output_paths: Dict[int, Path],
    summary_path: Path,
    written_counts: Dict[int, int],
    cohort_slug: str,
) -> Dict[str, Any]:
    retained_ids = set(retained_order)
    ok_union = set.union(*(ids for ids in ok_by_target.values())) if ok_by_target else set()
    return {
        "type": "shard_ablation_cohort_summary",
        "dataset_name": dataset_name,
        "provider": provider,
        "target_turns": list(target_turns),
        "created_at": utc_now(),
        "cohort": {
            "name": DEFAULT_COHORT_NAME,
            "slug": cohort_slug,
            "retained": len(retained_order),
            "dropped_from_ok_union": len(ok_union.difference(retained_ids)),
            "example_ids_sha256": _example_ids_hash(retained_order),
        },
        "inputs": {
            "k%d" % item.target_turns: {
                "path": str(item.path),
                "total": len(item.rows),
                "ok": item.status_counts.get("ok", 0),
                "not_ok": len(item.rows) - item.status_counts.get("ok", 0),
                "statuses": dict(sorted(item.status_counts.items())),
                "ok_not_in_cohort": len(ok_by_target[item.target_turns].difference(retained_ids)),
            }
            for item in loaded
        },
        "outputs": {
            "summary": str(summary_path),
            "shards": {
                "k%d" % target: {
                    "path": str(path),
                    "written": written_counts[target],
                }
                for target, path in sorted(output_paths.items())
            },
        },
    }


def _example_ids_hash(example_ids: Sequence[str]) -> str:
    payload = json.dumps(list(example_ids), ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cohort_slug(targets: Sequence[int]) -> str:
    return DEFAULT_COHORT_NAME + "_" + "_".join("k%d" % target for target in targets)


def _input_shard_path(artifacts_dir: Path, dataset_name: str, provider: str, target: int) -> Path:
    return artifacts_dir / ("shards.%s.%s.k%d.jsonl" % (dataset_name, provider, target))


def _output_shard_path(out_dir: Path, dataset_name: str, provider: str, target: int, cohort_slug: str) -> Path:
    return out_dir / ("shards.%s.%s.k%d.%s.jsonl" % (dataset_name, provider, target, cohort_slug))


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
