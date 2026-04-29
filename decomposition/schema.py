from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence
import html
import re
import unicodedata

SECTION_TYPES = {"title", "body", "tldr", "edit", "update", "other"}
SECTION_ROLES = {
    "setup",
    "background_context",
    "main_event",
    "prior_interaction",
    "current_conflict",
    "clarification",
    "motivation",
    "tldr_summary",
    "final_question",
    "other",
}
SUPPORTED_TARGET_TURNS = {4, 6, 8}
STATUSES = {"ok", "ineligible_primary_fixed4", "ineligible_target_shards"}
SEVERITIES = {"info", "warning", "error"}
TARGET_TURNS = 4
_CODE_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class ValidationError(Exception):
    """Raised when a shard record violates the seg_v1 contract."""

    def __init__(self, errors: Sequence[str]):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


@dataclass
class WarningItem:
    """Top-level warning emitted with the canonical code/field/severity shape."""

    code: str
    field: str
    severity: str
    message: str

    def to_dict(self) -> Dict[str, str]:
        """Serialize the warning to JSON-compatible primitives."""
        return {
            "code": self.code,
            "field": self.field,
            "severity": self.severity,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WarningItem":
        """Parse a warning object from JSON-compatible data."""
        return cls(
            code=str(data.get("code", "")),
            field=str(data.get("field", "")),
            severity=str(data.get("severity", "")),
            message=str(data.get("message", "")),
        )


@dataclass
class AtomicUnit:
    """A verbatim source span that can be traced back to raw_source_text."""

    unit_id: int
    text: str
    start_char: int
    end_char: int
    section_type: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the atomic unit to JSON-compatible primitives."""
        return {
            "unit_id": self.unit_id,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "section_type": self.section_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AtomicUnit":
        """Parse an atomic unit object from JSON-compatible data."""
        return cls(
            unit_id=int(data.get("unit_id")),
            text=str(data.get("text", "")),
            start_char=int(data.get("start_char")),
            end_char=int(data.get("end_char")),
            section_type=str(data.get("section_type", "")),
        )


@dataclass
class Shard:
    """A chronological group of atomic units for one target conversation turn."""

    shard_id: int
    unit_ids: List[int]
    text: str
    section_role: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the shard to JSON-compatible primitives."""
        return {
            "shard_id": self.shard_id,
            "unit_ids": list(self.unit_ids),
            "text": self.text,
            "section_role": self.section_role,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Shard":
        """Parse a shard object from JSON-compatible data."""
        return cls(
            shard_id=int(data.get("shard_id")),
            unit_ids=[int(value) for value in data.get("unit_ids", [])],
            text=str(data.get("text", "")),
            section_role=str(data.get("section_role", "")),
        )


@dataclass
class ShardRecord:
    """Canonical decomposition artifact record for one source example."""

    example_id: str
    dataset_name: str
    source_text_field: str
    run_id: str
    segmentation_version: str
    segmenter_model: str
    created_at: str
    raw_source_text: str
    target_turns: int
    status: str
    atomic_units: List[AtomicUnit] = field(default_factory=list)
    shards: List[Shard] = field(default_factory=list)
    warnings: List[WarningItem] = field(default_factory=list)
    normalized_source_text: Optional[str] = None
    gold_label: Optional[str] = None
    request_fingerprint: Optional[str] = None
    content_fingerprint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the record into the JSONL contract shape."""
        data = {
            "example_id": self.example_id,
            "dataset_name": self.dataset_name,
            "source_text_field": self.source_text_field,
            "run_id": self.run_id,
            "segmentation_version": self.segmentation_version,
            "segmenter_model": self.segmenter_model,
            "created_at": self.created_at,
            "raw_source_text": self.raw_source_text,
            "normalized_source_text": self.normalized_source_text,
            "target_turns": self.target_turns,
            "status": self.status,
            "atomic_units": [unit.to_dict() for unit in self.atomic_units],
            "shards": [shard.to_dict() for shard in self.shards],
            "warnings": [warning.to_dict() for warning in self.warnings],
        }
        if self.gold_label is not None:
            data["gold_label"] = self.gold_label
        if self.request_fingerprint is not None:
            data["request_fingerprint"] = self.request_fingerprint
        if self.content_fingerprint is not None:
            data["content_fingerprint"] = self.content_fingerprint
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShardRecord":
        """Parse a shard record from JSON-compatible data before validation."""
        return cls(
            example_id=str(data.get("example_id", "")),
            dataset_name=str(data.get("dataset_name", "")),
            source_text_field=str(data.get("source_text_field", "")),
            run_id=str(data.get("run_id", "")),
            segmentation_version=str(data.get("segmentation_version", "")),
            segmenter_model=str(data.get("segmenter_model", "")),
            created_at=str(data.get("created_at", "")),
            raw_source_text=str(data.get("raw_source_text", "")),
            normalized_source_text=data.get("normalized_source_text"),
            target_turns=int(data.get("target_turns", 0)),
            status=str(data.get("status", "")),
            atomic_units=[AtomicUnit.from_dict(item) for item in data.get("atomic_units", [])],
            shards=[Shard.from_dict(item) for item in data.get("shards", [])],
            warnings=[WarningItem.from_dict(item) for item in data.get("warnings", [])],
            gold_label=data.get("gold_label"),
            request_fingerprint=data.get("request_fingerprint"),
            content_fingerprint=data.get("content_fingerprint"),
        )


def validate_record_dict(data: Dict[str, Any]) -> None:
    """Validate a JSON-compatible record dictionary."""
    try:
        record = ShardRecord.from_dict(data)
    except Exception as exc:  # noqa: BLE001 - validator reports data-shape failures.
        raise ValidationError(["record could not be parsed: %s. Fix: compare the row with ShardRecord fields." % exc])
    validate_record(record)


def validate_record(record: ShardRecord) -> None:
    """Raise ValidationError if a shard record violates the v1 contract."""
    errors: List[str] = []
    _validate_common(record, errors)
    _validate_warnings(record.warnings, errors)

    if record.status == "ok":
        if record.target_turns not in SUPPORTED_TARGET_TURNS:
            errors.append("ok records must have target_turns in %s" % sorted(SUPPORTED_TARGET_TURNS))
        if len(record.shards) != record.target_turns:
            errors.append("ok records must contain exactly target_turns shards")
        if not record.atomic_units:
            errors.append("ok records must contain at least one atomic unit")
        _validate_atomic_units(record, errors)
        _validate_shards(record, errors)
    elif record.status in ("ineligible_primary_fixed4", "ineligible_target_shards"):
        if record.status == "ineligible_target_shards" and record.target_turns not in SUPPORTED_TARGET_TURNS:
            errors.append("ineligible_target_shards records must have target_turns in %s" % sorted(SUPPORTED_TARGET_TURNS))
        if record.shards:
            errors.append("ineligible records must have empty shards")
        if not record.warnings:
            errors.append("ineligible records must include a warning explaining why")
        if record.atomic_units:
            _validate_atomic_units(record, errors)
    else:
        errors.append("invalid status: %s" % record.status)

    if errors:
        raise ValidationError(errors)


def _validate_common(record: ShardRecord, errors: List[str]) -> None:
    """Validate fields required for both ok and ineligible records."""
    for field_name in (
        "example_id",
        "dataset_name",
        "source_text_field",
        "run_id",
        "segmentation_version",
        "segmenter_model",
        "created_at",
        "raw_source_text",
    ):
        if not getattr(record, field_name):
            errors.append("missing required field: %s" % field_name)
    if record.status not in STATUSES:
        errors.append("invalid status: %s" % record.status)


def _validate_warnings(warnings: Iterable[WarningItem], errors: List[str]) -> None:
    """Validate warning objects without deciding whether warnings are required."""
    for index, warning in enumerate(warnings):
        prefix = "warnings[%d]" % index
        if not _CODE_RE.match(warning.code):
            errors.append("%s.code must be lowercase snake_case" % prefix)
        if not warning.field:
            errors.append("%s.field is required" % prefix)
        if warning.severity not in SEVERITIES:
            errors.append("%s.severity is invalid" % prefix)
        if not warning.message:
            errors.append("%s.message is required" % prefix)


def validate_atomic_unit_sequence(
    raw_source_text: str,
    atomic_units: Sequence[AtomicUnit],
    warnings: Optional[Iterable[WarningItem]] = None,
) -> None:
    """Validate source-aligned atomic units without requiring final shards."""
    errors: List[str] = []
    warning_items = list(warnings or [])
    _validate_warnings(warning_items, errors)
    record = ShardRecord(
        example_id="atomic-validation",
        dataset_name="atomic-validation",
        source_text_field="raw_source_text",
        run_id="atomic-validation",
        segmentation_version="atomic-validation",
        segmenter_model="atomic-validation",
        created_at="atomic-validation",
        raw_source_text=raw_source_text,
        target_turns=TARGET_TURNS,
        status="ineligible_target_shards",
        atomic_units=list(atomic_units),
        shards=[],
        warnings=warning_items,
    )
    _validate_atomic_units(record, errors)
    if errors:
        raise ValidationError(errors)


def _validate_atomic_units(record: ShardRecord, errors: List[str]) -> None:
    """Validate source coverage, monotonic offsets, and exact raw-text slices."""
    raw = record.raw_source_text
    previous_end = 0
    expected_ids = list(range(1, len(record.atomic_units) + 1))
    observed_ids = [unit.unit_id for unit in record.atomic_units]
    if observed_ids != expected_ids:
        errors.append("atomic unit ids must be sequential starting at 1")

    for index, unit in enumerate(record.atomic_units):
        prefix = "atomic_units[%d]" % index
        if unit.section_type not in SECTION_TYPES:
            errors.append("%s.section_type is invalid: %s" % (prefix, unit.section_type))
        if unit.start_char < previous_end:
            errors.append("%s overlaps or reorders previous unit" % prefix)
        if unit.start_char < 0 or unit.end_char <= unit.start_char or unit.end_char > len(raw):
            errors.append("%s has invalid start/end offsets" % prefix)
            continue
        # Whitespace-only gaps preserve original formatting without losing story content.
        gap = raw[previous_end : unit.start_char]
        if _semantic_gap_text(gap):
            errors.append("uncovered non-whitespace text before %s: %r" % (prefix, _clip(gap.strip(), 120)))
        if raw[unit.start_char : unit.end_char] != unit.text:
            errors.append("%s text does not match raw_source_text slice" % prefix)
        previous_end = unit.end_char

    trailing = raw[previous_end:]
    if _semantic_gap_text(trailing):
        errors.append("uncovered non-whitespace text after final atomic unit: %r" % _clip(trailing.strip(), 120))

    _validate_other_warnings(record, errors)


def _validate_other_warnings(record: ShardRecord, errors: List[str]) -> None:
    """Require explicit warnings when a controlled enum falls back to other."""
    warned_fields = {warning.field for warning in record.warnings if warning.code == "enum_other"}
    for index, unit in enumerate(record.atomic_units):
        if unit.section_type == "other" and "atomic_units[%d].section_type" % index not in warned_fields:
            errors.append("atomic_units[%d].section_type=other requires enum_other warning" % index)
    for index, shard in enumerate(record.shards):
        if shard.section_role == "other" and "shards[%d].section_role" % index not in warned_fields:
            errors.append("shards[%d].section_role=other requires enum_other warning" % index)


def _validate_shards(record: ShardRecord, errors: List[str]) -> None:
    """Validate shard ids, references, text spans, and complete unit partitioning."""
    unit_by_id = {unit.unit_id: unit for unit in record.atomic_units}
    consumed: List[int] = []
    expected_shard_ids = list(range(1, len(record.shards) + 1))
    observed_shard_ids = [shard.shard_id for shard in record.shards]
    if observed_shard_ids != expected_shard_ids:
        errors.append("shard ids must be sequential starting at 1")

    for index, shard in enumerate(record.shards):
        prefix = "shards[%d]" % index
        if shard.section_role not in SECTION_ROLES:
            errors.append("%s.section_role is invalid: %s" % (prefix, shard.section_role))
        if not shard.unit_ids:
            errors.append("%s.unit_ids must not be empty" % prefix)
            continue
        for unit_id in shard.unit_ids:
            if unit_id not in unit_by_id:
                errors.append("%s references unknown unit_id %s" % (prefix, unit_id))
        if shard.unit_ids != sorted(shard.unit_ids):
            errors.append("%s.unit_ids must be increasing" % prefix)
        consumed.extend(shard.unit_ids)
        if all(unit_id in unit_by_id for unit_id in shard.unit_ids):
            first = unit_by_id[shard.unit_ids[0]]
            last = unit_by_id[shard.unit_ids[-1]]
            expected_text = record.raw_source_text[first.start_char : last.end_char].strip()
            if shard.text != expected_text:
                errors.append("%s.text must match raw text spanning its unit_ids" % prefix)

    expected = [unit.unit_id for unit in record.atomic_units]
    if consumed != expected:
        errors.append("shards must partition atomic units exactly once and in order")


def _clip(value: str, limit: int) -> str:
    """Keep validation messages useful without dumping whole source posts."""
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _semantic_gap_text(value: str) -> str:
    """Return non-format, non-whitespace gap content that carries story text."""
    decoded = html.unescape(value)
    return "".join(char for char in decoded if not char.isspace() and unicodedata.category(char) != "Cf")
