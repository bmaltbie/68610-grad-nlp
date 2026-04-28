import json

from decomposition.anthropic_io import SEGMENTATION_OUTPUT_SCHEMA, call_anthropic, request_to_anthropic_messages


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeClient:
    def __init__(self, response):
        self.messages = FakeMessages(response)


def test_request_to_anthropic_messages_hoists_system_prompt() -> None:
    request = {
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "raw post"},
        ]
    }

    system, messages = request_to_anthropic_messages(request)

    assert system == "system prompt"
    assert messages == [{"role": "user", "content": "raw post"}]


def test_call_anthropic_uses_structured_output_extra_body() -> None:
    response_payload = {"atomic_units": [], "shards": [], "warnings": []}
    fake_response = {
        "id": "msg_test",
        "model": "claude-test",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 2},
        "content": [{"type": "text", "text": json.dumps(response_payload)}],
    }
    client = FakeClient(fake_response)
    request = {
        "request_id": "req1",
        "messages": [
            {"role": "system", "content": "segment"},
            {"role": "user", "content": "AITA? First. Second. Third."},
        ],
    }

    wrapped = call_anthropic(
        request,
        model="claude-test",
        max_tokens=123,
        temperature=0.0,
        client=client,
        created_at="2026-04-27T00:00:00Z",
    )

    call = client.messages.calls[0]
    assert call["model"] == "claude-test"
    assert call["max_tokens"] == 123
    assert call["system"] == "segment"
    assert call["messages"] == [{"role": "user", "content": "AITA? First. Second. Third."}]
    assert call["extra_body"]["output_config"]["format"]["type"] == "json_schema"
    assert wrapped["request_id"] == "req1"
    assert wrapped["segmenter_model"] == "claude-test"
    assert wrapped["model_output"] == json.dumps(response_payload)
    assert wrapped["raw_response"]["id"] == "msg_test"


def test_anthropic_schema_requires_one_based_atomic_unit_ids() -> None:
    atomic_item = SEGMENTATION_OUTPUT_SCHEMA["properties"]["atomic_units"]["items"]
    shard_unit_ids = SEGMENTATION_OUTPUT_SCHEMA["properties"]["shards"]["items"]["properties"]["unit_ids"]

    assert "unit_id" in atomic_item["required"]
    assert "1-based" in atomic_item["properties"]["unit_id"]["description"]
    assert "Never include 0" in shard_unit_ids["description"]
