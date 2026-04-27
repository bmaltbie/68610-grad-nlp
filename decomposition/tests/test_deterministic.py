from decomposition.datasets import DatasetExample
from decomposition.deterministic import SegmentationConfig, deterministic_segment, extract_atomic_units


def example(text: str) -> DatasetExample:
    return DatasetExample("ex1", "AITA-NTA-OG", "original_post", text, 1)


def test_deterministic_segment_produces_four_valid_shards() -> None:
    text = "AITA for refusing? I live with my sister. She asked me to cook. I said no."

    record = deterministic_segment(example(text), SegmentationConfig("run1", created_at="2026-04-27T00:00:00Z"))

    assert record.status == "ok"
    assert len(record.shards) == 4
    assert record.atomic_units[0].section_type == "title"
    assert any(warning.code == "inferred_title" for warning in record.warnings)


def test_deterministic_segment_marks_long_source_ineligible() -> None:
    text = "AITA for this? " + ("Long sentence. " * 1000)

    record = deterministic_segment(
        example(text),
        SegmentationConfig("run1", created_at="2026-04-27T00:00:00Z", max_source_tokens=10),
    )

    assert record.status == "ineligible_primary_fixed4"
    assert record.shards == []
    assert record.warnings[0].code == "source_too_long"


def test_extract_atomic_units_labels_edit_and_tldr_sections() -> None:
    text = "AITA for this? First body sentence.\n\nEDIT: Added context.\nTL;DR: Short summary."

    units, _warnings = extract_atomic_units(text)

    assert [unit.section_type for unit in units] == ["title", "body", "edit", "tldr"]
