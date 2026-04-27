from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import json

from .align import UnitSpec, align_units
from .datasets import DatasetExample
from .deterministic import SEGMENTATION_VERSION, utc_now
from .schema import Shard, ShardRecord, WarningItem, validate_record


class LLMIngestError(Exception):
    """Raised when provider-neutral model output cannot become a shard record."""

    pass


def load_prompt(segmentation_version: str = SEGMENTATION_VERSION) -> str:
    """Load the versioned segmentation prompt for request generation."""
    prompt_path = Path(__file__).with_name("prompts") / (segmentation_version + ".txt")
    if not prompt_path.exists():
        raise FileNotFoundError(
            "prompt version not found: %s. Fix: add decomposition/prompts/%s.txt or pass a supported --segmentation-version."
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
            "atomic_units": [{"text": "verbatim source span", "section_type": "title|body|tldr|edit|update|other"}],
            "shards": [{"unit_ids": [1, 2], "section_role": "setup|main_event|..."}],
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
    unit_specs = [_unit_spec(item) for item in payload.get("atomic_units", [])]
    if not unit_specs:
        raise LLMIngestError(
            "response contains no atomic_units. Fix: include a non-empty atomic_units list with verbatim source text."
        )
    atomic_units = align_units(str(request["raw_source_text"]), unit_specs)
    shards = _shards_from_payload(str(request["raw_source_text"]), atomic_units, payload.get("shards", []))
    warnings = [_warning_item(item) for item in payload.get("warnings", [])]
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


def _unit_spec(item: Any) -> UnitSpec:
    """Normalize one atomic unit response item into a UnitSpec."""
    if isinstance(item, str):
        return UnitSpec(text=item, section_type="body")
    if not isinstance(item, dict):
        raise LLMIngestError(
            "atomic unit must be object or string. Fix: use {'text': '<verbatim span>', 'section_type': 'body'}."
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
