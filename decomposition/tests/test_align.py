import pytest

from decomposition.align import AlignmentError, UnitSpec, align_units


def test_align_units_preserves_duplicate_occurrence_order() -> None:
    raw = "Repeat. Middle. Repeat. End."

    units = align_units(
        raw,
        [
            UnitSpec("Repeat."),
            UnitSpec("Middle."),
            UnitSpec("Repeat."),
            UnitSpec("End."),
        ],
    )

    assert units[0].start_char == 0
    assert units[2].start_char == raw.rfind("Repeat.")


def test_align_units_rejects_missing_text() -> None:
    with pytest.raises(AlignmentError):
        align_units("First. Second.", [UnitSpec("First."), UnitSpec("Missing.")])


def test_align_units_rejects_unique_reordered_text() -> None:
    with pytest.raises(AlignmentError):
        align_units("First. Second. Third.", [UnitSpec("Second."), UnitSpec("First.")])
