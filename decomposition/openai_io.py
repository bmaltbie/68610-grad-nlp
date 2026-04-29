from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Dict, List, Optional
import json
import os

from .deterministic import utc_now
from .llm_schema import SEGMENTATION_OUTPUT_SCHEMA


DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"
QUALITY_OPENAI_MODEL = "gpt-5.5"
DEFAULT_MAX_OUTPUT_TOKENS = 4096


class OpenAIRunError(Exception):
    """Raised when the native OpenAI runner cannot produce ingestible output."""

    pass


def create_openai_client(api_key: Optional[str] = None, max_retries: Optional[int] = None) -> Any:
    """Create the native OpenAI SDK client using OPENAI_API_KEY by default."""
    resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_key:
        raise OpenAIRunError("OPENAI_API_KEY is not set.")
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - dependency is installed in normal project runs.
        raise OpenAIRunError("openai package is not installed.") from exc
    if max_retries is None:
        return OpenAI(api_key=resolved_key)
    return OpenAI(api_key=resolved_key, max_retries=max_retries)


def call_openai(
    request: Dict[str, Any],
    model: str = DEFAULT_OPENAI_MODEL,
    max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    temperature: float = 0.0,
    client: Optional[Any] = None,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute one decomposition request against OpenAI and wrap the response for ingest.

    Data flow:

        provider-neutral request
            -> OpenAI Responses API
            -> provider-neutral response wrapper
            -> record_from_response(...)

    OpenAI owns generation. Local ingest still owns source-span authority.
    """
    openai_client = client or create_openai_client()
    response = openai_client.responses.create(
        **openai_response_kwargs(
            request,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    )
    return response_from_openai_response(request, response, model=model, created_at=created_at)


def openai_response_kwargs(
    request: Dict[str, Any],
    model: str = DEFAULT_OPENAI_MODEL,
    max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Build OpenAI Responses API kwargs for realtime and batch calls."""
    return {
        "model": model,
        "input": request_to_openai_input(request),
        "max_output_tokens": max_tokens,
        "temperature": temperature,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "segmentation",
                "schema": SEGMENTATION_OUTPUT_SCHEMA,
                "strict": True,
            }
        },
    }


def openai_batch_request(
    custom_id: str,
    request: Dict[str, Any],
    model: str = DEFAULT_OPENAI_MODEL,
    max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Build one OpenAI Batch JSONL request row for the Responses endpoint."""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": openai_response_kwargs(
            request,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        ),
    }


def request_to_openai_input(request: Dict[str, Any]) -> List[Dict[str, str]]:
    """Map provider-neutral chat messages to OpenAI Responses input messages."""
    api_messages: List[Dict[str, str]] = []
    for message in request.get("messages", []):
        if not isinstance(message, dict):
            raise OpenAIRunError("request message must be an object with role/content")
        role = str(message.get("role", "user"))
        content = str(message.get("content", ""))
        if role not in ("system", "developer", "user", "assistant"):
            role = "user"
        api_messages.append({"role": role, "content": content})
    if not api_messages:
        raise OpenAIRunError("request has no messages to send to OpenAI")
    return api_messages


def response_from_openai_response(
    request: Dict[str, Any],
    response: Any,
    model: str,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Normalize an OpenAI SDK response into the existing ingest wrapper shape."""
    response_dict = response_to_dict(response)
    return {
        "request_id": request.get("request_id"),
        "segmenter_model": str(response_dict.get("model") or _get(response, "model", None) or model),
        "created_at": created_at or utc_now(),
        "provider": "openai",
        "provider_response_id": response_dict.get("id") or _get(response, "id", None),
        "status": response_dict.get("status") or _get(response, "status", None),
        "incomplete_details": response_dict.get("incomplete_details") or _get(response, "incomplete_details", None),
        "usage": response_dict.get("usage") or _get(response, "usage", None),
        "model_output": response_output_text(response),
        "raw_response": response_dict,
    }


def response_from_openai_batch_result(
    request: Dict[str, Any],
    result: Dict[str, Any],
    model: str,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Normalize one successful OpenAI Batch output row into an ingest wrapper."""
    response = result.get("response")
    body: Any = None
    if isinstance(response, dict):
        body = response.get("body")
    if body is None:
        body = result.get("body") or result
    wrapped = response_from_openai_response(request, body, model=model, created_at=created_at)
    wrapped["batch_custom_id"] = result.get("custom_id")
    if isinstance(response, dict):
        wrapped["provider_request_id"] = response.get("request_id")
        wrapped["provider_status_code"] = response.get("status_code")
    return wrapped


def raise_for_incomplete_response(response: Dict[str, Any]) -> None:
    """Reject OpenAI responses that are known to be incomplete before parsing JSON."""
    if not str(response.get("model_output", "")).strip():
        raise OpenAIRunError(
            "OpenAI response contained no text content. Fix: retry this row or inspect the raw response sidecar."
        )
    status = str(response.get("status") or "")
    if status == "incomplete":
        details = response.get("incomplete_details")
        reason = _get(details, "reason", None)
        raise OpenAIRunError(
            "OpenAI response was incomplete%s. Fix: increase --max-tokens or retry this row."
            % (": %s" % reason if reason else "")
        )


def response_to_dict(response: Any) -> Dict[str, Any]:
    """Best-effort JSON-compatible SDK response serialization for audit sidecars."""
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        data = response.model_dump(mode="json")
        if isinstance(data, dict):
            return data
    if hasattr(response, "to_dict"):
        data = response.to_dict()
        if isinstance(data, dict):
            return data
    return {"repr": repr(response)}


def response_output_text(response: Any) -> str:
    """Extract assistant text from OpenAI Responses objects or fake-test dictionaries."""
    direct = _get(response, "output_text", None)
    if direct is not None:
        return str(direct)
    response_dict = response_to_dict(response)
    direct = response_dict.get("output_text")
    if direct is not None:
        return str(direct)

    parts: List[str] = []
    for item in _iterable(response_dict.get("output", [])):
        for block in _iterable(_get(item, "content", [])):
            block_type = str(_get(block, "type", ""))
            if block_type in ("output_text", "text") or _get(block, "text", None) is not None:
                parts.append(str(_get(block, "text", "")))
    return "".join(parts)


def _iterable(value: Any) -> Iterable[Any]:
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        return value
    return []


def _get(value: Any, key: str, default: Any = None) -> Any:
    """Read either dict keys or SDK object attributes."""
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)
