import json

import pytest

from decomposition.openai_io import (
    DEFAULT_OPENAI_MODEL,
    OpenAIRunError,
    call_openai,
    openai_batch_request,
    openai_response_kwargs,
    raise_for_incomplete_response,
    request_to_openai_input,
)


class FakeResponses:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeClient:
    def __init__(self, response):
        self.responses = FakeResponses(response)


def request():
    return {
        "request_id": "req1",
        "messages": [
            {"role": "system", "content": "segment"},
            {"role": "user", "content": "AITA? First. Second. Third."},
        ],
    }


def test_request_to_openai_input_preserves_system_and_user_messages() -> None:
    assert request_to_openai_input(request()) == [
        {"role": "system", "content": "segment"},
        {"role": "user", "content": "AITA? First. Second. Third."},
    ]


def test_openai_response_kwargs_uses_structured_outputs() -> None:
    kwargs = openai_response_kwargs(request(), model="gpt-test", max_tokens=123, temperature=0.0)

    assert kwargs["model"] == "gpt-test"
    assert kwargs["max_output_tokens"] == 123
    assert kwargs["input"][0]["role"] == "system"
    assert kwargs["text"]["format"]["type"] == "json_schema"
    assert kwargs["text"]["format"]["name"] == "segmentation"
    assert kwargs["text"]["format"]["strict"] is True
    assert kwargs["text"]["format"]["schema"]["required"] == ["atomic_units", "shards", "warnings"]


def test_call_openai_wraps_response_for_ingest() -> None:
    response_payload = {"atomic_units": [], "shards": [], "warnings": []}
    fake_response = {
        "id": "resp_test",
        "model": "gpt-test",
        "status": "completed",
        "usage": {"input_tokens": 1, "output_tokens": 2},
        "output_text": json.dumps(response_payload),
    }
    client = FakeClient(fake_response)

    wrapped = call_openai(
        request(),
        model="gpt-test",
        max_tokens=123,
        temperature=0.0,
        client=client,
        created_at="2026-04-29T00:00:00Z",
    )

    call = client.responses.calls[0]
    assert call["model"] == "gpt-test"
    assert call["text"]["format"]["type"] == "json_schema"
    assert wrapped["request_id"] == "req1"
    assert wrapped["provider"] == "openai"
    assert wrapped["segmenter_model"] == "gpt-test"
    assert wrapped["provider_response_id"] == "resp_test"
    assert wrapped["model_output"] == json.dumps(response_payload)
    assert wrapped["raw_response"]["id"] == "resp_test"


def test_empty_openai_content_is_rejected_as_incomplete() -> None:
    response = {
        "request_id": "req1",
        "provider": "openai",
        "status": "completed",
        "model_output": "",
    }

    with pytest.raises(OpenAIRunError, match="no text content"):
        raise_for_incomplete_response(response)


def test_incomplete_openai_response_is_rejected() -> None:
    response = {
        "request_id": "req1",
        "provider": "openai",
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "model_output": "{}",
    }

    with pytest.raises(OpenAIRunError, match="incomplete"):
        raise_for_incomplete_response(response)


def test_openai_batch_request_targets_responses_endpoint() -> None:
    row = openai_batch_request("custom1", request(), model=DEFAULT_OPENAI_MODEL)

    assert row["custom_id"] == "custom1"
    assert row["method"] == "POST"
    assert row["url"] == "/v1/responses"
    assert row["body"]["model"] == DEFAULT_OPENAI_MODEL
    assert row["body"]["text"]["format"]["type"] == "json_schema"
