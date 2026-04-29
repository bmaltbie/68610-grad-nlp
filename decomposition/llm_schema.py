from __future__ import annotations

from typing import Any, Dict

from .schema import SECTION_ROLES, SECTION_TYPES, SEVERITIES


SEGMENTATION_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "atomic_units": {
            "type": "array",
            "description": (
                "Source-order atomic narrative units that partition the whole source post. "
                "Every text value must be copied exactly from the source, including Unicode typography."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "unit_id": {
                        "type": "integer",
                        "description": "1-based source-order id. The first atomic unit is 1, never 0.",
                    },
                    "text": {
                        "type": "string",
                        "description": (
                            "Exact verbatim source span for one atomic narrative unit. "
                            "Do not normalize quotes, apostrophes, punctuation, emoji, or capitalization."
                        ),
                    },
                    "section_type": {
                        "type": "string",
                        "enum": sorted(SECTION_TYPES),
                    },
                },
                "required": ["unit_id", "text", "section_type"],
                "additionalProperties": False,
            },
        },
        "shards": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "unit_ids": {
                        "type": "array",
                        "description": (
                            "1-based contiguous chronological references to atomic_units.unit_id values. "
                            "Never include 0, skip ids, duplicate ids, or move earlier units into later shards."
                        ),
                        "items": {"type": "integer"},
                    },
                    "section_role": {
                        "type": "string",
                        "enum": sorted(SECTION_ROLES),
                    },
                },
                "required": ["unit_ids", "section_role"],
                "additionalProperties": False,
            },
        },
        "warnings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "field": {"type": "string"},
                    "severity": {"type": "string", "enum": sorted(SEVERITIES)},
                    "message": {"type": "string"},
                },
                "required": ["code", "field", "severity", "message"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["atomic_units", "shards", "warnings"],
    "additionalProperties": False,
}


ATOMIC_UNITS_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "atomic_units": SEGMENTATION_OUTPUT_SCHEMA["properties"]["atomic_units"],
        "warnings": SEGMENTATION_OUTPUT_SCHEMA["properties"]["warnings"],
    },
    "required": ["atomic_units", "warnings"],
    "additionalProperties": False,
}
