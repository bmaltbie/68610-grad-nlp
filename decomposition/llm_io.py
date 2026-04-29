from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import html
import json
import re
import unicodedata

from .align import AlignmentError, UnitSpec, align_units
from .datasets import DatasetExample
from .deterministic import SEGMENTATION_VERSION, utc_now
from .schema import (
    AtomicUnit,
    Shard,
    ShardRecord,
    ValidationError,
    WarningItem,
    validate_atomic_unit_sequence,
    validate_record,
)


class LLMIngestError(Exception):
    """Raised when provider-neutral model output cannot become a shard record."""

    pass


@dataclass
class AtomicUnitsRecord:
    """Reusable expensive LLM output before deterministic shard planning."""

    example_id: str
    dataset_name: str
    source_text_field: str
    run_id: str
    segmentation_version: str
    segmenter_model: str
    created_at: str
    raw_source_text: str
    atomic_units: List[AtomicUnit] = field(default_factory=list)
    warnings: List[WarningItem] = field(default_factory=list)
    atomic_request_fingerprint: Optional[str] = None
    atomic_content_fingerprint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the atomic cache row into JSONL-safe primitives."""
        data = {
            "example_id": self.example_id,
            "dataset_name": self.dataset_name,
            "source_text_field": self.source_text_field,
            "run_id": self.run_id,
            "segmentation_version": self.segmentation_version,
            "segmenter_model": self.segmenter_model,
            "created_at": self.created_at,
            "raw_source_text": self.raw_source_text,
            "atomic_units": [unit.to_dict() for unit in self.atomic_units],
            "warnings": [warning.to_dict() for warning in self.warnings],
        }
        if self.atomic_request_fingerprint is not None:
            data["atomic_request_fingerprint"] = self.atomic_request_fingerprint
        if self.atomic_content_fingerprint is not None:
            data["atomic_content_fingerprint"] = self.atomic_content_fingerprint
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AtomicUnitsRecord":
        """Parse an atomic cache row before validation."""
        return cls(
            example_id=str(data.get("example_id", "")),
            dataset_name=str(data.get("dataset_name", "")),
            source_text_field=str(data.get("source_text_field", "")),
            run_id=str(data.get("run_id", "")),
            segmentation_version=str(data.get("segmentation_version", "")),
            segmenter_model=str(data.get("segmenter_model", "")),
            created_at=str(data.get("created_at", "")),
            raw_source_text=str(data.get("raw_source_text", "")),
            atomic_units=[AtomicUnit.from_dict(item) for item in data.get("atomic_units", [])],
            warnings=[WarningItem.from_dict(item) for item in data.get("warnings", [])],
            atomic_request_fingerprint=data.get("atomic_request_fingerprint") or data.get("request_fingerprint"),
            atomic_content_fingerprint=data.get("atomic_content_fingerprint") or data.get("content_fingerprint"),
        )


def load_prompt(segmentation_version: str = SEGMENTATION_VERSION) -> str:
    """Load the versioned segmentation prompt for request generation."""
    prompt_path = Path(__file__).with_name("prompts") / (segmentation_version + ".txt")
    if not prompt_path.exists():
        raise FileNotFoundError(
            "prompt version not found: %s. Fix: add decomposition/prompts/%s.txt or pass a supported --segmentation-version."
            % (prompt_path, segmentation_version)
        )
    return prompt_path.read_text(encoding="utf-8")


def load_atomic_prompt(segmentation_version: str = SEGMENTATION_VERSION) -> str:
    """Load the versioned atomic-only prompt for request generation."""
    prompt_path = Path(__file__).with_name("prompts") / (segmentation_version + "_atomic.txt")
    if not prompt_path.exists():
        raise FileNotFoundError(
            "atomic prompt version not found: %s. Fix: add decomposition/prompts/%s_atomic.txt or pass a supported --segmentation-version."
            % (prompt_path, segmentation_version)
        )
    return prompt_path.read_text(encoding="utf-8")


def request_id_for(example: DatasetExample, run_id: str, segmentation_version: str) -> str:
    """Build a deterministic request id tying model output back to one source row."""
    return "%s:%s:%s:%s:%s" % (
        example.dataset_name,
        example.example_id,
        example.source_text_field,
        run_id,
        segmentation_version,
    )


def atomic_request_id_for(example: DatasetExample, run_id: str, segmentation_version: str) -> str:
    """Build a deterministic atomic-only request id for one source row."""
    return request_id_for(example, run_id, segmentation_version) + ":atomic"


def build_request(
    example: DatasetExample,
    run_id: str,
    segmentation_version: str = SEGMENTATION_VERSION,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Create one provider-neutral JSON request for external model execution."""
    prompt = load_prompt(segmentation_version)
    return {
        "request_id": request_id_for(example, run_id, segmentation_version),
        "example_id": example.example_id,
        "dataset_name": example.dataset_name,
        "source_text_field": example.source_text_field,
        "run_id": run_id,
        "segmentation_version": segmentation_version,
        "created_at": created_at or utc_now(),
        "target_turns": 4,
        "raw_source_text": example.raw_source_text,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": example.raw_source_text},
        ],
        "expected_response": {
            "atomic_units": [
                {"unit_id": 1, "text": "verbatim source span", "section_type": "title|body|tldr|edit|update|other"}
            ],
            "shards": [{"unit_ids": [1, 2], "section_role": "setup|main_event|..."}],
            "warnings": [],
        },
    }


def build_atomic_request(
    example: DatasetExample,
    run_id: str,
    segmentation_version: str = SEGMENTATION_VERSION,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Create one provider-neutral atomic-only request for external model execution."""
    prompt = load_atomic_prompt(segmentation_version)
    return {
        "request_id": atomic_request_id_for(example, run_id, segmentation_version),
        "example_id": example.example_id,
        "dataset_name": example.dataset_name,
        "source_text_field": example.source_text_field,
        "run_id": run_id,
        "segmentation_version": segmentation_version,
        "created_at": created_at or utc_now(),
        "task": "atomic_units",
        "raw_source_text": example.raw_source_text,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": example.raw_source_text},
        ],
        "expected_response": {
            "atomic_units": [
                {"unit_id": 1, "text": "verbatim source span", "section_type": "title|body|tldr|edit|update|other"}
            ],
            "warnings": [],
        },
    }


def record_from_response(request: Dict[str, Any], response: Dict[str, Any]) -> ShardRecord:
    """Align, assemble, and validate one model response.

    The trust boundary lives here: LLM output is treated as a proposal. Local
    code computes spans from verbatim text and the schema validator decides
    whether the resulting record can enter shards.jsonl.
    """
    payload = _payload(response)
    unit_specs = _unit_specs_from_payload(_list_payload_field(payload, "atomic_units"))
    if not unit_specs:
        raise LLMIngestError(
            "response contains no atomic_units. Fix: include a non-empty atomic_units list with verbatim source text."
        )
    raw_source_text = str(request["raw_source_text"])
    atomic_units, alignment_warnings = _align_units_for_ingest(raw_source_text, unit_specs)
    shard_items, shard_warnings = _normalize_shard_references(_list_payload_field(payload, "shards"), len(atomic_units))
    shards = _shards_from_payload(raw_source_text, atomic_units, shard_items)
    warnings = [_warning_item(item) for item in _list_payload_field(payload, "warnings")]
    warnings.extend(alignment_warnings)
    warnings.extend(shard_warnings)
    record = ShardRecord(
        example_id=str(request["example_id"]),
        dataset_name=str(request["dataset_name"]),
        source_text_field=str(request["source_text_field"]),
        run_id=str(request["run_id"]),
        segmentation_version=str(request["segmentation_version"]),
        segmenter_model=str(response.get("segmenter_model") or payload.get("segmenter_model") or "external"),
        created_at=str(response.get("created_at") or request.get("created_at") or utc_now()),
        raw_source_text=str(request["raw_source_text"]),
        normalized_source_text=None,
        target_turns=4,
        status="ok",
        atomic_units=atomic_units,
        shards=shards,
        warnings=warnings,
    )
    validate_record(record)
    return record


def atomic_record_from_response(request: Dict[str, Any], response: Dict[str, Any]) -> AtomicUnitsRecord:
    """Align, assemble, and validate one atomic-only model response."""
    payload = _payload(response)
    unit_specs = _unit_specs_from_payload(_list_payload_field(payload, "atomic_units"))
    if not unit_specs:
        raise LLMIngestError(
            "response contains no atomic_units. Fix: include a non-empty atomic_units list with verbatim source text."
        )
    raw_source_text = str(request["raw_source_text"])
    atomic_units, alignment_warnings = _align_units_for_ingest(raw_source_text, unit_specs)
    warnings = [_warning_item(item) for item in _list_payload_field(payload, "warnings")]
    warnings.extend(alignment_warnings)
    record = AtomicUnitsRecord(
        example_id=str(request["example_id"]),
        dataset_name=str(request["dataset_name"]),
        source_text_field=str(request["source_text_field"]),
        run_id=str(request["run_id"]),
        segmentation_version=str(request["segmentation_version"]),
        segmenter_model=str(response.get("segmenter_model") or payload.get("segmenter_model") or "external"),
        created_at=str(response.get("created_at") or request.get("created_at") or utc_now()),
        raw_source_text=raw_source_text,
        atomic_units=atomic_units,
        warnings=warnings,
    )
    validate_atomic_record(record)
    return record


def atomic_record_from_shard_record(record: ShardRecord) -> AtomicUnitsRecord:
    """Reuse the source-aligned atomic units from a validated shard artifact."""
    validate_record(record)
    atomic_record = AtomicUnitsRecord(
        example_id=record.example_id,
        dataset_name=record.dataset_name,
        source_text_field=record.source_text_field,
        run_id=record.run_id,
        segmentation_version=record.segmentation_version,
        segmenter_model=record.segmenter_model,
        created_at=record.created_at,
        raw_source_text=record.raw_source_text,
        atomic_units=list(record.atomic_units),
        warnings=list(record.warnings),
        atomic_request_fingerprint=record.request_fingerprint,
        atomic_content_fingerprint=record.content_fingerprint,
    )
    validate_atomic_record(atomic_record)
    return atomic_record


def validate_atomic_record_dict(data: Dict[str, Any]) -> None:
    """Validate a JSON-compatible atomic cache row."""
    try:
        record = AtomicUnitsRecord.from_dict(data)
    except Exception as exc:  # noqa: BLE001 - validator reports data-shape failures.
        raise ValidationError(["atomic record could not be parsed: %s" % exc])
    validate_atomic_record(record)


def validate_atomic_record(record: AtomicUnitsRecord) -> None:
    """Raise ValidationError if an atomic cache row is not reusable."""
    errors: List[str] = []
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
    if not record.atomic_units:
        errors.append("atomic records must contain at least one atomic unit")
    if errors:
        raise ValidationError(errors)
    validate_atomic_unit_sequence(record.raw_source_text, record.atomic_units, record.warnings)


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield JSON objects from a JSONL file, skipping blank lines."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise LLMIngestError(
                    "%s:%d invalid JSON: %s. Fix: write exactly one JSON object per line."
                    % (path, line_number, exc)
                )


def load_requests(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load request rows keyed by request_id for response correlation."""
    requests: Dict[str, Dict[str, Any]] = {}
    for request in read_jsonl(path):
        request_id = request.get("request_id")
        if not request_id:
            raise LLMIngestError(
                "request missing request_id. Fix: generate requests with `python -m decomposition.cli generate-requests`."
            )
        requests[str(request_id)] = request
    return requests


def _payload(response: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model payloads from common provider-neutral wrapper shapes."""
    for key in ("output", "response"):
        if isinstance(response.get(key), dict):
            return response[key]
    for key in ("model_output", "content"):
        value = response.get(key)
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise LLMIngestError(
                    "%s is not valid JSON: %s. Fix: store model output as a JSON object string."
                    % (key, exc)
                )
            if not isinstance(parsed, dict):
                raise LLMIngestError(
                    "%s must decode to a JSON object. Fix: return keys atomic_units, shards, and warnings."
                    % key
                )
            return parsed
    return response


def _list_payload_field(payload: Dict[str, Any], field_name: str) -> List[Any]:
    """Read a response field that must be a JSON array."""
    value = payload.get(field_name, [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise LLMIngestError("%s must be a list. Fix: return %s as a JSON array." % (field_name, field_name))
    return value


def _unit_specs_from_payload(items: Iterable[Any]) -> List[UnitSpec]:
    """Normalize atomic unit response items and reject contradictory explicit ids."""
    specs: List[UnitSpec] = []
    for index, item in enumerate(items, start=1):
        if isinstance(item, dict) and "unit_id" in item:
            try:
                unit_id = int(item["unit_id"])
            except (TypeError, ValueError):
                raise LLMIngestError(
                    "atomic_units[%d].unit_id must be an integer. Fix: use 1-based source-order ids."
                    % (index - 1)
                )
            if unit_id != index:
                raise LLMIngestError(
                    "atomic_units[%d].unit_id must be %d because unit_ids are 1-based source-order references; got %d. Fix: number atomic_units 1, 2, 3, ... in source order."
                    % (index - 1, index, unit_id)
                )
        specs.append(_unit_spec(item))
    return specs


def _unit_spec(item: Any) -> UnitSpec:
    """Normalize one atomic unit response item into a UnitSpec."""
    if isinstance(item, str):
        return UnitSpec(text=item, section_type="body")
    if not isinstance(item, dict):
        raise LLMIngestError(
            "atomic unit must be object or string. Fix: use {'unit_id': 1, 'text': '<verbatim span>', 'section_type': 'body'}."
        )
    return UnitSpec(text=str(item.get("text", "")), section_type=str(item.get("section_type", "body")))


def _warning_item(item: Any) -> WarningItem:
    """Normalize warning strings or objects into the canonical warning shape."""
    if isinstance(item, str):
        return WarningItem(code="llm_warning", field="response", severity="warning", message=item)
    if not isinstance(item, dict):
        raise LLMIngestError(
            "warning must be object or string. Fix: use {'code', 'field', 'severity', 'message'}."
        )
    return WarningItem.from_dict(item)


_TYPOGRAPHY_TRANSLATION = str.maketrans(
    {
        "’": "'",
        "‘": "'",
        "‛": "'",
        "ʼ": "'",
        "`": "'",
        "´": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "—": "-",
        "–": "-",
    }
)


def _align_units_for_ingest(raw: str, unit_specs: List[UnitSpec]) -> Tuple[List[AtomicUnit], List[WarningItem]]:
    """Align model text, repairing only typography/spacing/control-character drift."""
    try:
        return align_units(raw, unit_specs), []
    except AlignmentError:
        pass

    units: List[AtomicUnit] = []
    cursor = 0
    repair_count = 0
    for index, spec in enumerate(unit_specs, start=1):
        if not spec.text:
            raise AlignmentError(
                "unit %d has empty text. Fix: every atomic_units entry must contain verbatim source text."
                % index
            )
        start = raw.find(spec.text, cursor)
        if start >= 0:
            end = start + len(spec.text)
            text = spec.text
        else:
            repaired = _find_normalized_span(raw, spec.text, cursor)
            if repaired is None:
                raise AlignmentError(_alignment_error_message(raw, spec.text, index, cursor))
            start, end, text = repaired
            repair_count += 1
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

    warnings: List[WarningItem] = []
    if repair_count:
        warnings.append(
            WarningItem(
                code="normalized_alignment_text",
                field="atomic_units.text",
                severity="warning",
                message=(
                    "Local ingest repaired %d atomic unit text span(s) by mapping typography, "
                    "whitespace, or invisible-control-character normalized model text back to exact raw source slices."
                )
                % repair_count,
            )
        )
    return units, warnings


def _find_normalized_span(raw: str, text: str, cursor: int) -> Optional[Tuple[int, int, str]]:
    """Find a nearby raw span that is identical after harmless text normalization."""
    target = _normalize_alignment_text(text)
    if not target:
        return None
    max_start = min(len(raw), cursor + 600)
    length_slack = 120
    for start in range(cursor, max_start):
        if _is_ignorable_alignment_char(raw[start]):
            continue
        min_end = start + max(1, len(text) - length_slack)
        max_end = min(len(raw), start + len(text) + length_slack)
        for end in range(min_end, max_end + 1):
            candidate = raw[start:end]
            normalized = _normalize_alignment_text(candidate)
            if normalized == target:
                return start, end, candidate
            if len(normalized) > len(target) + 6 and not target.startswith(normalized):
                break
    return None


def _normalize_alignment_text(value: str) -> str:
    """Normalize only presentation differences that do not change story content."""
    decoded = html.unescape(value)
    without_controls = "".join(char for char in decoded if unicodedata.category(char) != "Cf")
    normalized_typography = without_controls.translate(_TYPOGRAPHY_TRANSLATION)
    return re.sub(r"\s+", " ", normalized_typography).strip()


def _is_ignorable_alignment_char(value: str) -> bool:
    """Treat separators and invisible format controls as gaps, not unit starts."""
    return value.isspace() or unicodedata.category(value) == "Cf"


def _alignment_error_message(raw: str, text: str, index: int, cursor: int) -> str:
    """Build the same actionable diagnostic shape as strict alignment."""
    return (
        "unit %d text not found after offset %d. Offending unit text: %r. "
        "Raw text near offset %d: %r. Fix: copy exact source text in source order; "
        "do not summarize or rewrite."
    ) % (
        index,
        cursor,
        _clip(text, 180),
        cursor,
        _clip(raw[max(0, cursor - 80) : cursor + 180], 260),
    )


def _normalize_shard_references(shard_items: Iterable[Any], unit_count: int) -> Tuple[List[Any], List[WarningItem]]:
    """Normalize safe shard reference mistakes without changing the source text."""
    items = list(shard_items)
    warnings: List[WarningItem] = []
    all_unit_ids: List[int] = []
    parsed_by_shard: List[List[int]] = []
    for item in items:
        if not isinstance(item, dict):
            return items, warnings
        try:
            unit_ids = [int(value) for value in item.get("unit_ids", [])]
        except (TypeError, ValueError):
            return items, warnings
        parsed_by_shard.append(unit_ids)
        all_unit_ids.extend(unit_ids)

    if 0 in all_unit_ids and sorted(all_unit_ids) == list(range(unit_count)):
        normalized: List[Any] = []
        normalized_by_shard: List[List[int]] = []
        normalized_all_unit_ids: List[int] = []
        for item, unit_ids in zip(items, parsed_by_shard):
            normalized_item = dict(item)
            normalized_ids = [unit_id + 1 for unit_id in unit_ids]
            normalized_item["unit_ids"] = normalized_ids
            normalized.append(normalized_item)
            normalized_by_shard.append(normalized_ids)
            normalized_all_unit_ids.extend(normalized_ids)
        items = normalized
        parsed_by_shard = normalized_by_shard
        all_unit_ids = normalized_all_unit_ids
        warnings.append(
            WarningItem(
                code="normalized_zero_based_unit_ids",
                field="shards.unit_ids",
                severity="warning",
                message=(
                    "Model returned zero-based shard unit_ids covering all atomic units exactly once; "
                    "local ingest normalized them to one-based ids."
                ),
            )
        )
    elif 0 in all_unit_ids:
        return items, warnings

    expected = list(range(1, unit_count + 1))
    if all_unit_ids != expected and sorted(all_unit_ids) == expected and all(parsed_by_shard):
        maxima = [max(unit_ids) for unit_ids in parsed_by_shard]
        if maxima == sorted(maxima) and len(set(maxima)) == len(maxima) and maxima[-1] == unit_count:
            normalized = []
            start_unit_id = 1
            for item, end_unit_id in zip(items, maxima):
                normalized_item = dict(item)
                normalized_item["unit_ids"] = list(range(start_unit_id, end_unit_id + 1))
                normalized.append(normalized_item)
                start_unit_id = end_unit_id + 1
            items = normalized
            warnings.append(
                WarningItem(
                    code="normalized_noncontiguous_shards",
                    field="shards.unit_ids",
                    severity="warning",
                    message=(
                        "Model covered every atomic unit exactly once but used non-contiguous shard groups; "
                        "local ingest projected them to contiguous chronological ranges using shard end boundaries."
                    ),
                )
            )
    return items, warnings


def _shards_from_payload(raw: str, atomic_units: List[Any], shard_items: Iterable[Any]) -> List[Shard]:
    """Build shard objects from model-proposed unit id groupings."""
    by_id = {unit.unit_id: unit for unit in atomic_units}
    shards: List[Shard] = []
    for index, item in enumerate(shard_items, start=1):
        if not isinstance(item, dict):
            raise LLMIngestError(
                "shard must be object. Fix: use {'unit_ids': [1], 'section_role': 'setup'}."
            )
        try:
            unit_ids = [int(value) for value in item.get("unit_ids", [])]
        except (TypeError, ValueError):
            raise LLMIngestError("shard %d unit_ids must be integers. Fix: use source-order unit ids." % index)
        if not unit_ids:
            raise LLMIngestError("shard %d has no unit_ids. Fix: assign one or more atomic unit ids." % index)
        missing = [unit_id for unit_id in unit_ids if unit_id not in by_id]
        if missing:
            if 0 in missing:
                raise LLMIngestError(
                    "shard %d references unit 0, but shard unit_ids must be 1-based. The first atomic unit is 1, not 0. Fix: renumber shard unit_ids to atomic_units.unit_id values."
                    % index
                )
            raise LLMIngestError(
                "shard %d references missing units %s. Fix: shard unit_ids must refer to emitted atomic_units."
                % (index, missing)
            )
        first = by_id[unit_ids[0]]
        last = by_id[unit_ids[-1]]
        shards.append(
            Shard(
                shard_id=index,
                unit_ids=unit_ids,
                text=raw[first.start_char : last.end_char].strip(),
                section_role=str(item.get("section_role", "other")),
            )
        )
    return shards


def _clip(value: str, limit: int) -> str:
    """Keep diagnostics readable for long posts."""
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."
