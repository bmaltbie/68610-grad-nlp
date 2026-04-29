import pytest

from decomposition.schema import (
    AtomicUnit,
    Shard,
    ShardRecord,
    ValidationError,
    WarningItem,
    validate_record,
)


def valid_record() -> ShardRecord:
    raw = "One. Two. Three. Four."
    units = [
        AtomicUnit(1, "One.", 0, 4, "body"),
        AtomicUnit(2, "Two.", 5, 9, "body"),
        AtomicUnit(3, "Three.", 10, 16, "body"),
        AtomicUnit(4, "Four.", 17, 22, "body"),
    ]
    return ShardRecord(
        example_id="ex1",
        dataset_name="AITA-NTA-OG",
        source_text_field="original_post",
        run_id="run1",
        segmentation_version="seg_v1",
        segmenter_model="deterministic-baseline-v1",
        created_at="2026-04-27T00:00:00Z",
        raw_source_text=raw,
        normalized_source_text=None,
        target_turns=4,
        status="ok",
        atomic_units=units,
        shards=[
            Shard(1, [1], "One.", "setup"),
            Shard(2, [2], "Two.", "main_event"),
            Shard(3, [3], "Three.", "background_context"),
            Shard(4, [4], "Four.", "current_conflict"),
        ],
        warnings=[],
    )


def test_valid_record_passes_with_whitespace_gaps() -> None:
    validate_record(valid_record())


def test_wrong_shard_count_fails() -> None:
    record = valid_record()
    record.shards = record.shards[:3]

    with pytest.raises(ValidationError):
        validate_record(record)


def test_invalid_enum_fails() -> None:
    record = valid_record()
    record.atomic_units[0].section_type = "intro"

    with pytest.raises(ValidationError):
        validate_record(record)


def test_other_enum_requires_warning() -> None:
    record = valid_record()
    record.shards[1].section_role = "other"

    with pytest.raises(ValidationError):
        validate_record(record)

    record.warnings.append(
        WarningItem(
            code="enum_other",
            field="shards[1].section_role",
            severity="warning",
            message="No role fit this shard.",
        )
    )
    validate_record(record)


def test_non_whitespace_gap_fails() -> None:
    record = valid_record()
    record.atomic_units[1].start_char = 6
    record.atomic_units[1].text = "wo."
    record.shards[1].text = "wo."

    with pytest.raises(ValidationError) as exc_info:
        validate_record(record)

    assert "uncovered non-whitespace text before atomic_units[1]: 'T'" in str(exc_info.value)


def test_ineligible_record_passes_with_warning_and_empty_shards() -> None:
    record = ShardRecord(
        example_id="ex1",
        dataset_name="AITA-NTA-OG",
        source_text_field="original_post",
        run_id="run1",
        segmentation_version="seg_v1",
        segmenter_model="deterministic-baseline-v1",
        created_at="2026-04-27T00:00:00Z",
        raw_source_text="One sentence only.",
        normalized_source_text=None,
        target_turns=4,
        status="ineligible_primary_fixed4",
        atomic_units=[],
        shards=[],
        warnings=[WarningItem("too_few_atomic_units", "status", "warning", "Too short.")],
    )

    validate_record(record)


def test_ineligible_target_shards_preserves_atomic_units() -> None:
    raw = "One. Two. Three."
    record = ShardRecord(
        example_id="ex1",
        dataset_name="AITA-YTA",
        source_text_field="prompt",
        run_id="run1",
        segmentation_version="seg_v1",
        segmenter_model="gpt-test+natural_dp_v1",
        created_at="2026-04-29T00:00:00Z",
        raw_source_text=raw,
        normalized_source_text=None,
        target_turns=4,
        status="ineligible_target_shards",
        atomic_units=[
            AtomicUnit(1, "One.", 0, 4, "body"),
            AtomicUnit(2, "Two.", 5, 9, "body"),
            AtomicUnit(3, "Three.", 10, 16, "body"),
        ],
        shards=[],
        warnings=[WarningItem("too_few_atomic_units_for_target", "status", "warning", "Too short for k4.")],
    )

    validate_record(record)
