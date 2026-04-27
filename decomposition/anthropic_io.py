from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Tuple
import json
import os

from .deterministic import utc_now
from .schema import SECTION_ROLES, SECTION_TYPES, SEVERITIES


DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5"
DEFAULT_MAX_TOKENS = 4096


class AnthropicRunError(Exception):
    """Raised when the native Anthropic runner cannot produce ingestible output."""

    pass


SEGMENTATION_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "atomic_units": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Exact verbatim source span for one atomic narrative unit.",
                    },
                    "section_type": {
                        "type": "string",
                        "enum": sorted(SECTION_TYPES),
                    },
                },
                "required": ["text", "section_type"],
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


def create_anthropic_client(api_key: Optional[str] = None) -> Any:
    """Create the native Anthropic SDK client using ANTHROPIC_API_KEY by default."""
    resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not resolved_key:
        raise AnthropicRunError("ANTHROPIC_API_KEY is not set.")
    try:
        from anthropic import Anthropic
    except ImportError as exc:  # pragma: no cover - dependency is installed in normal project runs.
        raise AnthropicRunError("anthropic package is not installed.") from exc
    return Anthropic(api_key=resolved_key)


def call_anthropic(
    request: Dict[str, Any],
    model: str = DEFAULT_ANTHROPIC_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
    client: Optional[Any] = None,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute one decomposition request against Anthropic and wrap the response for ingest.

    Data flow:

        provider-neutral request
            -> Anthropic Messages API
            -> provider-neutral response wrapper
            -> record_from_response(...)

    Anthropic owns generation. Local ingest still owns source-span authority.
    """
    anthropic_client = client or create_anthropic_client()
    system, messages = request_to_anthropic_messages(request)
    kwargs: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
        "extra_body": {
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": SEGMENTATION_OUTPUT_SCHEMA,
                }
            }
        },
    }
    if system:
        kwargs["system"] = system
    message = anthropic_client.messages.create(**kwargs)
    return response_from_anthropic_message(request, message, model=model, created_at=created_at)


def request_to_anthropic_messages(request: Dict[str, Any]) -> Tuple[str, List[Dict[str, str]]]:
    """Map provider-neutral chat messages to Anthropic's system + messages shape."""
    system_parts: List[str] = []
    api_messages: List[Dict[str, str]] = []
    for message in request.get("messages", []):
        if not isinstance(message, dict):
            raise AnthropicRunError("request message must be an object with role/content")
        role = str(message.get("role", "user"))
        content = str(message.get("content", ""))
        if role in ("system", "developer"):
            system_parts.append(content)
        elif role in ("user", "assistant"):
            api_messages.append({"role": role, "content": content})
        else:
            api_messages.append({"role": "user", "content": content})
    if not api_messages:
        raise AnthropicRunError("request has no user messages to send to Anthropic")
    return "\n\n".join(part for part in system_parts if part), api_messages


def response_from_anthropic_message(
    request: Dict[str, Any],
    message: Any,
    model: str,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Normalize an Anthropic SDK response into the existing ingest wrapper shape."""
    message_dict = message_to_dict(message)
    content_text = content_blocks_to_text(_get(message, "content", []))
    if not content_text.strip():
        raise AnthropicRunError("Anthropic response contained no text content")
    return {
        "request_id": request.get("request_id"),
        "segmenter_model": str(message_dict.get("model") or _get(message, "model", None) or model),
        "created_at": created_at or utc_now(),
        "provider": "anthropic",
        "provider_message_id": message_dict.get("id") or _get(message, "id", None),
        "stop_reason": message_dict.get("stop_reason") or _get(message, "stop_reason", None),
        "usage": message_dict.get("usage"),
        "model_output": content_text,
        "raw_response": message_dict,
    }


def raise_for_incomplete_response(response: Dict[str, Any]) -> None:
    """Reject Anthropic responses that are known to be incomplete before parsing JSON."""
    stop_reason = response.get("stop_reason")
    if stop_reason == "max_tokens":
        raise AnthropicRunError(
            "Anthropic response stopped at max_tokens. Fix: increase --max-tokens and retry this row."
        )
    if stop_reason == "refusal":
        raise AnthropicRunError(
            "Anthropic refused this request. Fix: inspect the row and prompt before retrying."
        )


def message_to_dict(message: Any) -> Dict[str, Any]:
    """Best-effort JSON-compatible SDK response serialization for audit sidecars."""
    if isinstance(message, dict):
        return message
    if hasattr(message, "model_dump"):
        data = message.model_dump(mode="json")
        if isinstance(data, dict):
            return data
    if hasattr(message, "to_dict"):
        data = message.to_dict()
        if isinstance(data, dict):
            return data
    return {"repr": repr(message)}


def content_blocks_to_text(content: Any) -> str:
    """Extract assistant text from Anthropic content blocks or fake-test dictionaries."""
    if isinstance(content, str):
        return content
    if not isinstance(content, Iterable):
        return ""
    parts: List[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        block_type = _get(block, "type", None)
        if block_type == "text" or _get(block, "text", None) is not None:
            parts.append(str(_get(block, "text", "")))
        elif block_type == "tool_use" and isinstance(_get(block, "input", None), dict):
            parts.append(json.dumps(_get(block, "input"), ensure_ascii=False))
    return "".join(parts)


def _get(value: Any, key: str, default: Any = None) -> Any:
    """Read either dict keys or SDK object attributes."""
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)
