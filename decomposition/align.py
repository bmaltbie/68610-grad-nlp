from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .schema import AtomicUnit


class AlignmentError(Exception):
    """Raised when model-proposed text cannot be located in the raw source."""

    pass


@dataclass
class UnitSpec:
    """A model-proposed atomic unit before local span authority is applied."""

    text: str
    section_type: str = "body"


def align_units(raw_source_text: str, unit_specs: Iterable[UnitSpec]) -> List[AtomicUnit]:
    """Convert verbatim unit text into monotonic raw-text spans.

    The model is allowed to propose text boundaries, but not offsets. Searching
    from `cursor` forces later duplicate phrases to map to later source
    occurrences and rejects reordered or invented text.
    """
    units: List[AtomicUnit] = []
    cursor = 0
    for index, spec in enumerate(unit_specs, start=1):
        text = spec.text
        if not text:
            raise AlignmentError(
                "unit %d has empty text. Fix: every atomic_units entry must contain verbatim source text."
                % index
            )
        start = raw_source_text.find(text, cursor)
        if start < 0:
            raise AlignmentError(
                (
                    "unit %d text not found after offset %d. Offending unit text: %r. "
                    "Raw text near offset %d: %r. Fix: copy exact Unicode characters from the source in source order; "
                    "do not normalize curly quotes/apostrophes, summarize, or rewrite."
                )
                % (
                    index,
                    cursor,
                    _clip(text, 180),
                    cursor,
                    _clip(raw_source_text[max(0, cursor - 80) : cursor + 180], 260),
                )
            )
        end = start + len(text)
        units.append(
            AtomicUnit(
                unit_id=index,
                text=text,
                start_char=start,
                end_char=end,
                section_type=spec.section_type,
            )
        )
        cursor = end
    return units


def _clip(value: str, limit: int) -> str:
    """Keep alignment diagnostics readable for long posts."""
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."
