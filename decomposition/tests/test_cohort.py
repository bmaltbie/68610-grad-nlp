from pathlib import Path
import argparse
import json
import subprocess
import sys

import pytest

from decomposition import cohort
from decomposition.datasets import DatasetError
from decomposition.schema import AtomicUnit, ShardRecord, WarningItem, validate_record
from decomposition.shard_planner import plan_shards


REPO_ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_cli(*args: str, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "decomposition.cli", *args],
        cwd=str(cwd),
        text=True,
        capture_output=True,
    )


def raw_for(prefix: str, unit_count: int) -> str:
    return " ".join("%s%d" % (prefix, index) for index in range(1, unit_count + 1))


def units_for(raw: str):
    units = []
    cursor = 0
    for index, token in enumerate(raw.split(" "), start=1):
        start = raw.find(token, cursor)
        end = start + len(token)
        units.append(AtomicUnit(index, raw[start:end], start, end, "title" if index == 1 else "body"))
        cursor = end
    return units


def shard_row(example_id: str, unit_count: int, target_turns: int, raw: str = None):
    raw = raw or raw_for(example_id, unit_count)
    units = units_for(raw)
    warnings = []
    if len(units) >= target_turns:
        status = "ok"
        shards = plan_shards(raw, units, target_turns)
    else:
        status = "ineligible_target_shards"
        shards = []
        warnings = [
            WarningItem(
                code="too_few_atomic_units_for_target",
                field="status",
                severity="warning",
                message="Only %d atomic units are available for target_turns=%d."
                % (len(units), target_turns),
            )
        ]
    record = ShardRecord(
        example_id=example_id,
        dataset_name="AITA-YTA",
        source_text_field="prompt",
        run_id="run1",
        segmentation_version="seg_v1",
        segmenter_model="gpt-test+natural_dp_v1",
        created_at="2026-04-30T00:00:00Z",
        raw_source_text=raw,
        target_turns=target_turns,
        status=status,
        atomic_units=units,
        shards=shards,
        warnings=warnings,
        request_fingerprint="sha256:request-%s-k%d" % (example_id, target_turns),
        content_fingerprint="sha256:content-%s-k%d" % (example_id, target_turns),
    )
    validate_record(record)
    return record.to_dict()


def write_target_file(artifacts_dir: Path, target_turns: int, rows) -> Path:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / ("shards.AITA-YTA.openai.k%d.jsonl" % target_turns)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def write_standard_target_files(artifacts_dir: Path) -> None:
    specs = [("long", 8), ("medium", 6), ("short", 4)]
    for target in (4, 6, 8):
        write_target_file(artifacts_dir, target, [shard_row(example_id, units, target) for example_id, units in specs])


def args_for(artifacts_dir: Path, out_dir: Path, allow_empty: bool = False):
    return argparse.Namespace(
        artifacts_dir=artifacts_dir,
        out_dir=out_dir,
        dataset_name="AITA-YTA",
        provider="openai",
        target_turns="4,6,8",
        allow_empty=allow_empty,
    )


def test_shard_ablation_cohort_writes_only_common_ok_ids_and_summary(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    out_dir = tmp_path / "cohorts"
    write_standard_target_files(artifacts_dir)

    assert cohort.run_shard_ablation_cohort(args_for(artifacts_dir, out_dir)) == 0

    for target in (4, 6, 8):
        rows = read_jsonl(out_dir / ("shards.AITA-YTA.openai.k%d.eligible_all_k4_k6_k8.jsonl" % target))
        assert [row["example_id"] for row in rows] == ["long"]
        assert len(rows[0]["shards"]) == target

    summary = json.loads(
        (out_dir / "cohort.AITA-YTA.openai.eligible_all_k4_k6_k8.summary.json").read_text(encoding="utf-8")
    )
    assert summary["cohort"]["retained"] == 1
    assert summary["inputs"]["k4"]["ok"] == 3
    assert summary["inputs"]["k6"]["ok"] == 2
    assert summary["inputs"]["k8"]["ok"] == 1
    assert summary["inputs"]["k4"]["ok_not_in_cohort"] == 2
    assert summary["inputs"]["k8"]["ok_not_in_cohort"] == 0


def test_shard_ablation_cohort_rejects_duplicate_example_ids(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    out_dir = tmp_path / "cohorts"
    write_target_file(artifacts_dir, 4, [shard_row("dup", 8, 4), shard_row("dup", 8, 4)])
    write_target_file(artifacts_dir, 6, [shard_row("dup", 8, 6)])
    write_target_file(artifacts_dir, 8, [shard_row("dup", 8, 8)])

    with pytest.raises(DatasetError, match="duplicate example_id=dup"):
        cohort.run_shard_ablation_cohort(args_for(artifacts_dir, out_dir))


def test_shard_ablation_cohort_rejects_mismatched_retained_source(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    out_dir = tmp_path / "cohorts"
    write_target_file(artifacts_dir, 4, [shard_row("long", 8, 4)])
    write_target_file(artifacts_dir, 6, [shard_row("long", 8, 6, raw=raw_for("changed", 8))])
    write_target_file(artifacts_dir, 8, [shard_row("long", 8, 8)])

    with pytest.raises(DatasetError, match="field=raw_source_text"):
        cohort.run_shard_ablation_cohort(args_for(artifacts_dir, out_dir))


def test_shard_ablation_cohort_requires_common_ids_unless_allowed(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    out_dir = tmp_path / "cohorts"
    specs = [("medium", 6), ("short", 4)]
    for target in (4, 6, 8):
        write_target_file(artifacts_dir, target, [shard_row(example_id, units, target) for example_id, units in specs])

    with pytest.raises(DatasetError, match="eligible-all cohort is empty"):
        cohort.run_shard_ablation_cohort(args_for(artifacts_dir, out_dir))

    assert cohort.run_shard_ablation_cohort(args_for(artifacts_dir, out_dir, allow_empty=True)) == 0
    assert read_jsonl(out_dir / "shards.AITA-YTA.openai.k4.eligible_all_k4_k6_k8.jsonl") == []


def test_cli_shard_ablation_cohort_writes_outputs_without_provider_calls(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    out_dir = tmp_path / "cohorts"
    write_standard_target_files(artifacts_dir)

    result = run_cli(
        "shard-ablation-cohort",
        "--artifacts-dir",
        str(artifacts_dir),
        "--dataset-name",
        "AITA-YTA",
        "--provider",
        "openai",
        "--target-turns",
        "4,6,8",
        "--out-dir",
        str(out_dir),
    )

    assert result.returncode == 0, result.stderr
    assert "cohort=eligible_all retained=1" in result.stdout
    assert (out_dir / "cohort.AITA-YTA.openai.eligible_all_k4_k6_k8.summary.json").exists()


def test_parse_target_turns_normalizes_order_and_rejects_bad_values() -> None:
    assert cohort.parse_target_turns("8,4,6") == (4, 6, 8)
    with pytest.raises(DatasetError, match="comma-separated integers"):
        cohort.parse_target_turns("4,nope")
