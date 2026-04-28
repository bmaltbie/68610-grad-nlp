from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Sequence, Tuple
import math
import re

from .datasets import DatasetExample
from .schema import AtomicUnit, Shard, ShardRecord, WarningItem, validate_record

SOURCE_TOKEN_BUDGET = 2800
SHARD_TOKEN_BUDGET = 900
SEGMENTATION_VERSION = "seg_v1"
SEGMENTER_MODEL = "deterministic-baseline-v1"
_MARKER_RE = re.compile(r"(?i)(?:^|\n\s*)(EDIT\s*\d*\s*:|UPDATE\s*\d*\s*:|TL\s*;?\s*DR\s*:|TLDR\s*:)")
_SENTENCE_RE = re.compile(r"\S.*?(?:[.!?](?=\s+|$)|$)", re.DOTALL)
_TITLE_RE = re.compile(r"(?i)^\s*(?:aita|wibta|am i the asshole|would i be the asshole|am i wrong)\b.*\?")


@dataclass
class SegmentationConfig:
    """Runtime settings shared by deterministic segmentation commands."""

    run_id: str
    segmentation_version: str = SEGMENTATION_VERSION
    segmenter_model: str = SEGMENTER_MODEL
    created_at: Optional[str] = None
    max_source_tokens: int = SOURCE_TOKEN_BUDGET
    max_shard_tokens: int = SHARD_TOKEN_BUDGET


def utc_now() -> str:
    """Return a compact UTC timestamp for artifact provenance."""
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def estimate_tokens(text: str) -> int:
    """Estimate model tokens with a conservative character-count heuristic."""
    return int(math.ceil(len(text) / 4.0))


def deterministic_segment(example: DatasetExample, config: SegmentationConfig) -> ShardRecord:
    """Build one validated fixed-four shard record without model assistance."""
    created_at = config.created_at or utc_now()
    raw = example.raw_source_text
    if not raw.strip():
        raise ValueError("source text is empty. Fix: choose a populated --source-field or remove the row.")
    if estimate_tokens(raw) > config.max_source_tokens:
        record = _ineligible(
            example,
            config,
            created_at,
            "source_too_long",
            "Estimated source tokens exceed primary fixed-4 budget.",
        )
        validate_record(record)
        return record

    units, warnings = extract_atomic_units(raw)
    if len(units) < 4:
        record = _ineligible(
            example,
            config,
            created_at,
            "too_few_atomic_units",
            "Source text does not split naturally into four atomic units.",
        )
        record.atomic_units = units
        record.warnings.extend(warnings)
        validate_record(record)
        return record

    shards = build_fixed_four_shards(raw, units)
    for shard in shards:
        if estimate_tokens(shard.text) > config.max_shard_tokens:
            record = _ineligible(
                example,
                config,
                created_at,
                "shard_too_long",
                "A deterministic shard exceeds the primary shard budget.",
            )
            validate_record(record)
            return record

    record = ShardRecord(
        example_id=example.example_id,
        dataset_name=example.dataset_name,
        source_text_field=example.source_text_field,
        run_id=config.run_id,
        segmentation_version=config.segmentation_version,
        segmenter_model=config.segmenter_model,
        created_at=created_at,
        raw_source_text=raw,
        normalized_source_text=None,
        target_turns=4,
        status="ok",
        atomic_units=units,
        shards=shards,
        warnings=warnings,
    )
    validate_record(record)
    return record


def extract_atomic_units(raw_source_text: str) -> Tuple[List[AtomicUnit], List[WarningItem]]:
    """Split raw source text into traceable atomic units.

    This baseline deliberately uses transparent text rules instead of clever NLP:
    detect visible AITA markers, then split each section into sentence-like units.
    That keeps failures auditable when downstream modules consume the artifacts.
    """
    units: List[AtomicUnit] = []
    warnings: List[WarningItem] = []
    for section_start, section_end, section_type in _sections(raw_source_text):
        section_text = raw_source_text[section_start:section_end]
        for match in _SENTENCE_RE.finditer(section_text):
            text = match.group(0).strip()
            if not text:
                continue
            start = section_start + match.start() + len(match.group(0)) - len(match.group(0).lstrip())
            end = start + len(text)
            unit_section_type = section_type
            if not units and section_type == "body" and _TITLE_RE.match(text):
                unit_section_type = "title"
                warnings.append(
                    WarningItem(
                        code="inferred_title",
                        field="atomic_units[0].section_type",
                        severity="info",
                        message="Leading AITA/WIBTA framing question inferred as title from source text.",
                    )
                )
            units.append(
                AtomicUnit(
                    unit_id=len(units) + 1,
                    text=raw_source_text[start:end],
                    start_char=start,
                    end_char=end,
                    section_type=unit_section_type,
                )
            )
    return units, warnings


def build_fixed_four_shards(raw_source_text: str, units: Sequence[AtomicUnit]) -> List[Shard]:
    """Merge atomic units into exactly four chronological shards."""
    groups = _partition_units(units, 4)
    shards: List[Shard] = []
    for index, group in enumerate(groups, start=1):
        first = group[0]
        last = group[-1]
        text = raw_source_text[first.start_char : last.end_char].strip()
        shards.append(
            Shard(
                shard_id=index,
                unit_ids=[unit.unit_id for unit in group],
                text=text,
                section_role=_role_for_group(index, group),
            )
        )
    return shards


def _sections(raw: str) -> List[Tuple[int, int, str]]:
    """Return contiguous text sections labeled by visible Reddit-style markers."""
    markers = list(_MARKER_RE.finditer(raw))
    if not markers:
        return [(0, len(raw), "body")]

    sections: List[Tuple[int, int, str]] = []
    if markers[0].start() > 0:
        sections.append((0, markers[0].start(), "body"))
    for index, marker in enumerate(markers):
        start = marker.start()
        end = markers[index + 1].start() if index + 1 < len(markers) else len(raw)
        label = marker.group(1).lower().replace(" ", "")
        if label.startswith("edit"):
            section_type = "edit"
        elif label.startswith("update"):
            section_type = "update"
        elif label.startswith("tl"):
            section_type = "tldr"
        else:
            section_type = "other"
        sections.append((start, end, section_type))
    return sections


def _partition_units(units: Sequence[AtomicUnit], count: int) -> List[List[AtomicUnit]]:
    """Greedily balance source-order units across a fixed number of shards."""
    remaining_units = list(units)
    groups: List[List[AtomicUnit]] = []
    remaining_groups = count
    while remaining_groups:
        if remaining_groups == 1:
            groups.append(remaining_units)
            break
        remaining_tokens = sum(estimate_tokens(unit.text) for unit in remaining_units)
        target = max(1, int(math.ceil(remaining_tokens / float(remaining_groups))))
        group: List[AtomicUnit] = []
        group_tokens = 0
        # Leave at least one unit for every remaining shard; empty shards are invalid.
        while remaining_units and len(remaining_units) > remaining_groups - 1:
            next_unit = remaining_units[0]
            if group and group_tokens + estimate_tokens(next_unit.text) > target:
                break
            group.append(remaining_units.pop(0))
            group_tokens += estimate_tokens(next_unit.text)
        groups.append(group)
        remaining_groups -= 1
    return groups


def _role_for_group(index: int, group: Sequence[AtomicUnit]) -> str:
    """Assign a coarse section role that satisfies the contract enum."""
    section_types = {unit.section_type for unit in group}
    text = " ".join(unit.text for unit in group).lower()
    if index == 1:
        return "setup"
    if "tldr" in section_types:
        return "tldr_summary"
    if section_types.intersection({"edit", "update"}):
        return "clarification"
    if index == 2:
        return "main_event"
    if index == 3:
        return "background_context"
    if "aita" in text or "wibta" in text or "asshole" in text:
        return "final_question"
    return "current_conflict"


def _ineligible(
    example: DatasetExample,
    config: SegmentationConfig,
    created_at: str,
    code: str,
    message: str,
) -> ShardRecord:
    """Create a valid ineligible record that preserves source provenance."""
    return ShardRecord(
        example_id=example.example_id,
        dataset_name=example.dataset_name,
        source_text_field=example.source_text_field,
        run_id=config.run_id,
        segmentation_version=config.segmentation_version,
        segmenter_model=config.segmenter_model,
        created_at=created_at,
        raw_source_text=example.raw_source_text,
        normalized_source_text=None,
        target_turns=4,
        status="ineligible_primary_fixed4",
        atomic_units=[],
        shards=[],
        warnings=[
            WarningItem(
                code=code,
                field="status",
                severity="warning",
                message=message,
            )
        ],
    )
