import pytest

from decomposition.align import AlignmentError
from decomposition.datasets import DatasetExample
from decomposition.llm_io import LLMIngestError, atomic_record_from_response, build_atomic_request, record_from_response


def request(raw: str):
    return {
        "example_id": "ex1",
        "dataset_name": "AITA-NTA-OG",
        "source_text_field": "original_post",
        "run_id": "run1",
        "segmentation_version": "seg_v1",
        "created_at": "2026-04-27T00:00:00Z",
        "raw_source_text": raw,
    }


def response(raw: str, shard_unit_ids):
    parts = raw.split(" ")
    return {
        "segmenter_model": "test-model",
        "response": {
            "atomic_units": [
                {"unit_id": 1, "text": parts[0], "section_type": "title"},
                {"unit_id": 2, "text": parts[1], "section_type": "body"},
                {"unit_id": 3, "text": parts[2], "section_type": "body"},
                {"unit_id": 4, "text": parts[3], "section_type": "body"},
            ],
            "shards": [
                {"unit_ids": shard_unit_ids[0], "section_role": "setup"},
                {"unit_ids": shard_unit_ids[1], "section_role": "main_event"},
                {"unit_ids": shard_unit_ids[2], "section_role": "background_context"},
                {"unit_ids": shard_unit_ids[3], "section_role": "current_conflict"},
            ],
            "warnings": [],
        },
    }


def test_record_from_response_normalizes_perfect_zero_based_shard_ids() -> None:
    raw = "AITA? First. Second. Third."

    record = record_from_response(request(raw), response(raw, [[0], [1], [2], [3]]))

    assert [shard.unit_ids for shard in record.shards] == [[1], [2], [3], [4]]
    assert [warning.code for warning in record.warnings] == ["normalized_zero_based_unit_ids"]
    assert "zero-based" in record.warnings[0].message


def test_record_from_response_projects_safe_noncontiguous_shards_to_ranges() -> None:
    raw = "AITA? First. Second. Third. Fourth. Fifth."
    model_response = {
        "segmenter_model": "test-model",
        "response": {
            "atomic_units": [
                {"unit_id": 1, "text": "AITA?", "section_type": "title"},
                {"unit_id": 2, "text": "First.", "section_type": "body"},
                {"unit_id": 3, "text": "Second.", "section_type": "body"},
                {"unit_id": 4, "text": "Third.", "section_type": "body"},
                {"unit_id": 5, "text": "Fourth.", "section_type": "body"},
                {"unit_id": 6, "text": "Fifth.", "section_type": "body"},
            ],
            "shards": [
                {"unit_ids": [1, 3], "section_role": "setup"},
                {"unit_ids": [2, 4], "section_role": "main_event"},
                {"unit_ids": [5], "section_role": "background_context"},
                {"unit_ids": [6], "section_role": "current_conflict"},
            ],
            "warnings": [],
        },
    }

    record = record_from_response(request(raw), model_response)

    assert [shard.unit_ids for shard in record.shards] == [[1, 2, 3], [4], [5], [6]]
    assert [warning.code for warning in record.warnings] == ["normalized_noncontiguous_shards"]


def test_record_from_response_rejects_ambiguous_zero_based_shard_ids() -> None:
    raw = "AITA? First. Second. Third."

    with pytest.raises(LLMIngestError, match="unit 0"):
        record_from_response(request(raw), response(raw, [[0], [2], [3], [4]]))


def test_record_from_response_rejects_contradictory_atomic_unit_id() -> None:
    raw = "AITA? First. Second. Third."
    bad_response = response(raw, [[1], [2], [3], [4]])
    bad_response["response"]["atomic_units"][0]["unit_id"] = 0

    with pytest.raises(LLMIngestError, match="must be 1"):
        record_from_response(request(raw), bad_response)


def test_record_from_response_alignment_error_includes_source_context() -> None:
    raw = "Privileged isn’t even the word. Then done. Last. Final."
    bad_response = {
        "segmenter_model": "test-model",
        "response": {
            "atomic_units": [
                {"unit_id": 1, "text": "Privileged is not even the word.", "section_type": "body"}
            ],
            "shards": [{"unit_ids": [1], "section_role": "setup"}],
            "warnings": [],
        },
    }

    with pytest.raises(AlignmentError) as exc_info:
        record_from_response(request(raw), bad_response)

    message = str(exc_info.value)
    assert "Offending unit text" in message
    assert "Raw text near offset 0" in message
    assert "isn’t" in message
    assert "copy exact source text" in message


def test_record_from_response_repairs_typography_and_control_character_alignment() -> None:
    raw = "AITA for boyfriend’s friend? Fine.\u202c\n\nThe next morning he didn’t reply. Final. Done."
    model_response = {
        "segmenter_model": "test-model",
        "response": {
            "atomic_units": [
                {"unit_id": 1, "text": "AITA for boyfriend's friend?", "section_type": "title"},
                {"unit_id": 2, "text": "Fine.", "section_type": "body"},
                {"unit_id": 3, "text": "\u202cThe next morning he didn't reply.", "section_type": "body"},
                {"unit_id": 4, "text": "Final.", "section_type": "body"},
                {"unit_id": 5, "text": "Done.", "section_type": "body"},
            ],
            "shards": [
                {"unit_ids": [1], "section_role": "setup"},
                {"unit_ids": [2], "section_role": "main_event"},
                {"unit_ids": [3, 4], "section_role": "background_context"},
                {"unit_ids": [5], "section_role": "current_conflict"},
            ],
            "warnings": [],
        },
    }

    record = record_from_response(request(raw), model_response)

    assert record.atomic_units[0].text == "AITA for boyfriend’s friend?"
    assert record.atomic_units[2].text == "The next morning he didn’t reply."
    assert [warning.code for warning in record.warnings] == ["normalized_alignment_text"]


def test_record_from_response_allows_zero_width_html_entity_gaps() -> None:
    raw = "AITA? First.&#x200B;\n\nSecond. Third."
    model_response = {
        "segmenter_model": "test-model",
        "response": {
            "atomic_units": [
                {"unit_id": 1, "text": "AITA?", "section_type": "title"},
                {"unit_id": 2, "text": "First.", "section_type": "body"},
                {"unit_id": 3, "text": "Second.", "section_type": "body"},
                {"unit_id": 4, "text": "Third.", "section_type": "body"},
            ],
            "shards": [
                {"unit_ids": [1], "section_role": "setup"},
                {"unit_ids": [2], "section_role": "main_event"},
                {"unit_ids": [3], "section_role": "background_context"},
                {"unit_ids": [4], "section_role": "current_conflict"},
            ],
            "warnings": [],
        },
    }

    record = record_from_response(request(raw), model_response)

    assert len(record.shards) == 4


def test_build_atomic_request_uses_atomic_prompt_and_shape() -> None:
    example = DatasetExample("ex1", "AITA-YTA", "prompt", "AITA? First. Second.", 1)

    atomic_request = build_atomic_request(
        example,
        run_id="run1",
        segmentation_version="seg_v1",
        created_at="2026-04-29T00:00:00Z",
    )

    assert atomic_request["request_id"].endswith(":atomic")
    assert atomic_request["task"] == "atomic_units"
    assert "Do not create shards" in atomic_request["messages"][0]["content"]
    assert "shards" not in atomic_request["expected_response"]


def test_atomic_record_from_response_validates_source_alignment_without_shards() -> None:
    raw = "AITA? First. Second."
    atomic_request = request(raw)
    atomic_response = {
        "segmenter_model": "gpt-test",
        "response": {
            "atomic_units": [
                {"unit_id": 1, "text": "AITA?", "section_type": "title"},
                {"unit_id": 2, "text": "First.", "section_type": "body"},
                {"unit_id": 3, "text": "Second.", "section_type": "body"},
            ],
            "warnings": [],
        },
    }

    record = atomic_record_from_response(atomic_request, atomic_response)

    assert record.segmenter_model == "gpt-test"
    assert [unit.text for unit in record.atomic_units] == ["AITA?", "First.", "Second."]
