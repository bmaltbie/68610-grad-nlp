from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import threading
import time

from decomposition import cli


REPO_ROOT = Path(__file__).resolve().parents[2]


def run_cli(*args: str, cwd: Path, env=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "decomposition.cli", *args],
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
    )


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_cli_deterministic_happy_path(tmp_path: Path) -> None:
    dataset = tmp_path / "data.csv"
    dataset.write_text(
        "id,original_post\nex1,AITA for refusing? First. Second. Third.\n",
        encoding="utf-8",
    )
    out = tmp_path / "artifacts" / "shards.jsonl"
    errors = tmp_path / "artifacts" / "errors.jsonl"

    result = run_cli(
        "deterministic",
        "--dataset",
        str(dataset),
        "--dataset-name",
        "AITA-NTA-OG",
        "--source-field",
        "original_post",
        "--run-id",
        "run1",
        "--created-at",
        "2026-04-27T00:00:00Z",
        "--out",
        str(out),
        "--errors",
        str(errors),
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, result.stderr
    rows = read_jsonl(out)
    assert rows[0]["status"] == "ok"
    assert rows[0]["example_id"] == "ex1"
    assert errors.read_text(encoding="utf-8") == ""


def test_cli_missing_required_args_returns_error() -> None:
    result = run_cli("deterministic", cwd=REPO_ROOT)

    assert result.returncode != 0
    assert "error" in result.stderr.lower()
    assert "Fix:" in result.stderr
    assert "deterministic --help" in result.stderr


def test_cli_help_explains_pipeline() -> None:
    result = run_cli("--help", cwd=REPO_ROOT)

    assert result.returncode == 0
    assert "Turn AITA CSV rows into validated decomposition artifacts" in result.stdout
    assert "anthropic" in result.stdout
    assert "generate-requests" in result.stdout
    assert "uv run --project decomposition" in result.stdout


def test_cli_dataset_error_includes_fix(tmp_path: Path) -> None:
    dataset = tmp_path / "data.csv"
    dataset.write_text("id,original_post\nex1,AITA for refusing? First. Second. Third.\n", encoding="utf-8")

    result = run_cli(
        "deterministic",
        "--dataset",
        str(dataset),
        "--dataset-name",
        "AITA-NTA-OG",
        "--source-field",
        "missing",
        "--run-id",
        "run1",
        "--out",
        str(tmp_path / "shards.jsonl"),
        "--errors",
        str(tmp_path / "errors.jsonl"),
        cwd=REPO_ROOT,
    )

    assert result.returncode == 1
    assert "source field 'missing' not found" in result.stderr
    assert "Fix:" in result.stderr
    assert "--source-field" in result.stderr


def test_cli_generate_requests_shape(tmp_path: Path) -> None:
    dataset = tmp_path / "data.csv"
    dataset.write_text("id,original_post\nex1,AITA for refusing? First. Second. Third.\n", encoding="utf-8")
    out = tmp_path / "requests.jsonl"

    result = run_cli(
        "generate-requests",
        "--dataset",
        str(dataset),
        "--dataset-name",
        "AITA-NTA-OG",
        "--source-field",
        "original_post",
        "--run-id",
        "run1",
        "--created-at",
        "2026-04-27T00:00:00Z",
        "--out",
        str(out),
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, result.stderr
    request = read_jsonl(out)[0]
    assert request["segmentation_version"] == "seg_v1"
    assert request["messages"][0]["role"] == "system"
    assert request["messages"][1]["content"].startswith("AITA")


def test_cli_ingest_responses_valid_and_invalid(tmp_path: Path) -> None:
    dataset = tmp_path / "data.csv"
    raw = "AITA for refusing? First. Second. Third."
    dataset.write_text("id,original_post\nex1,%s\n" % raw, encoding="utf-8")
    requests = tmp_path / "requests.jsonl"
    out = tmp_path / "shards.jsonl"
    errors = tmp_path / "errors.jsonl"

    make_requests = run_cli(
        "generate-requests",
        "--dataset",
        str(dataset),
        "--dataset-name",
        "AITA-NTA-OG",
        "--source-field",
        "original_post",
        "--run-id",
        "run1",
        "--created-at",
        "2026-04-27T00:00:00Z",
        "--out",
        str(requests),
        cwd=REPO_ROOT,
    )
    assert make_requests.returncode == 0, make_requests.stderr
    request = read_jsonl(requests)[0]
    responses = tmp_path / "responses.jsonl"
    valid_response = {
        "request_id": request["request_id"],
        "segmenter_model": "test-model",
        "response": {
            "atomic_units": [
                {"text": "AITA for refusing?", "section_type": "title"},
                {"text": "First.", "section_type": "body"},
                {"text": "Second.", "section_type": "body"},
                {"text": "Third.", "section_type": "body"},
            ],
            "shards": [
                {"unit_ids": [1], "section_role": "setup"},
                {"unit_ids": [2], "section_role": "main_event"},
                {"unit_ids": [3], "section_role": "background_context"},
                {"unit_ids": [4], "section_role": "current_conflict"},
            ],
            "warnings": [],
        },
    }
    invalid_response = {
        "request_id": request["request_id"],
        "response": {
            "atomic_units": [{"text": "AITA for refusing?"}, {"text": "Changed text."}],
            "shards": [],
            "warnings": [],
        },
    }
    responses.write_text(
        json.dumps(valid_response) + "\n" + json.dumps(invalid_response) + "\n",
        encoding="utf-8",
    )

    result = run_cli(
        "ingest-responses",
        "--requests",
        str(requests),
        "--responses",
        str(responses),
        "--out",
        str(out),
        "--errors",
        str(errors),
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, result.stderr
    assert len(read_jsonl(out)) == 1
    error_rows = read_jsonl(errors)
    assert len(error_rows) == 1
    assert "not found" in error_rows[0]["message"]
    assert "verbatim atomic-unit text" in error_rows[0]["fix"]


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeAnthropicClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)


class SlowFakeMessages(FakeMessages):
    def __init__(self, responses, delay: float):
        super().__init__(responses)
        self.delay = delay
        self.lock = threading.Lock()

    def create(self, **kwargs):
        with self.lock:
            self.calls.append(kwargs)
            response = self.responses.pop(0)
        time.sleep(self.delay)
        if isinstance(response, Exception):
            raise response
        return response


class SlowFakeAnthropicClient:
    def __init__(self, responses, delay: float):
        self.messages = SlowFakeMessages(responses, delay)


class FakeBatches:
    def __init__(self, results=None):
        self.create_calls = []
        self._results = list(results or [])

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return {"id": "msgbatch_test", "processing_status": "in_progress"}

    def retrieve(self, batch_id):
        return {"id": batch_id, "processing_status": "ended"}

    def results(self, batch_id):
        return list(self._results)


class FakeBatchMessages:
    def __init__(self, results=None):
        self.batches = FakeBatches(results)


class FakeBatchClient:
    def __init__(self, results=None):
        self.messages = FakeBatchMessages(results)


def anthropic_message(payload, stop_reason="end_turn"):
    return {
        "id": "msg_test",
        "model": "claude-test",
        "stop_reason": stop_reason,
        "usage": {"input_tokens": 1, "output_tokens": 2},
        "content": [{"type": "text", "text": payload if isinstance(payload, str) else json.dumps(payload)}],
    }


def valid_payload(raw: str):
    parts = raw.split(" ")
    shard_groups = [[1], [2], [3], list(range(4, len(parts) + 1))]
    return {
        "atomic_units": [
            {"text": part, "section_type": "title" if index == 0 else "body"}
            for index, part in enumerate(parts)
        ],
        "shards": [
            {"unit_ids": shard_groups[0], "section_role": "setup"},
            {"unit_ids": shard_groups[1], "section_role": "main_event"},
            {"unit_ids": shard_groups[2], "section_role": "background_context"},
            {"unit_ids": shard_groups[3], "section_role": "current_conflict"},
        ],
        "warnings": [],
    }


def anthropic_args(tmp_path: Path, dataset: Path, client: FakeAnthropicClient) -> argparse.Namespace:
    return argparse.Namespace(
        dataset=dataset,
        dataset_name="AITA-NTA-OG",
        source_field="original_post",
        run_id="run1",
        segmentation_version="seg_v1",
        created_at="2026-04-27T00:00:00Z",
        limit=None,
        out=tmp_path / "shards.jsonl",
        errors=tmp_path / "errors.jsonl",
        raw_responses=tmp_path / "raw_responses.jsonl",
        model="claude-test",
        max_tokens=4096,
        temperature=0.0,
        resume=False,
        resume_include_temp=True,
        raw_responses_mode=None,
        concurrency=1,
        llm_retries=1,
        anthropic_max_retries=2,
        progress="off",
        _anthropic_client=client,
    )


def test_cli_anthropic_happy_path_with_fake_client(tmp_path: Path) -> None:
    raw = "AITA? First. Second. Third."
    dataset = tmp_path / "data.csv"
    dataset.write_text("id,original_post\nex1,%s\n" % raw, encoding="utf-8")
    args = anthropic_args(tmp_path, dataset, FakeAnthropicClient([anthropic_message(valid_payload(raw))]))

    result = cli.anthropic_command(args)

    assert result == 0
    rows = read_jsonl(args.out)
    assert len(rows) == 1
    assert rows[0]["segmenter_model"] == "claude-test"
    assert len(rows[0]["shards"]) == 4
    assert read_jsonl(args.errors) == []
    raw_rows = read_jsonl(args.raw_responses)
    assert raw_rows[0]["provider"] == "anthropic"
    assert raw_rows[0]["raw_response"]["id"] == "msg_test"


def test_cli_anthropic_logs_bad_rows_and_continues(tmp_path: Path) -> None:
    first = "AITA? First. Second. Third."
    second = "AITA? Alpha. Beta. Gamma."
    dataset = tmp_path / "data.csv"
    dataset.write_text("id,original_post\nex1,%s\nex2,%s\n" % (first, second), encoding="utf-8")
    bad_payload = {
        "atomic_units": [{"text": "AITA?", "section_type": "title"}, {"text": "rewritten text", "section_type": "body"}],
        "shards": [
            {"unit_ids": [1], "section_role": "setup"},
            {"unit_ids": [2], "section_role": "main_event"},
            {"unit_ids": [1], "section_role": "background_context"},
            {"unit_ids": [2], "section_role": "current_conflict"},
        ],
        "warnings": [],
    }
    client = FakeAnthropicClient([anthropic_message(valid_payload(first)), anthropic_message(bad_payload)])
    args = anthropic_args(tmp_path, dataset, client)
    args.llm_retries = 0

    result = cli.anthropic_command(args)

    assert result == 0
    assert len(read_jsonl(args.out)) == 1
    error_rows = read_jsonl(args.errors)
    assert len(error_rows) == 1
    assert error_rows[0]["example_id"] == "ex2"
    assert "not found" in error_rows[0]["message"]


def test_cli_anthropic_logs_provider_error_and_continues(tmp_path: Path) -> None:
    first = "AITA? First. Second. Third."
    second = "AITA? Alpha. Beta. Gamma."
    dataset = tmp_path / "data.csv"
    dataset.write_text("id,original_post\nex1,%s\nex2,%s\n" % (first, second), encoding="utf-8")
    client = FakeAnthropicClient([RuntimeError("rate limit"), anthropic_message(valid_payload(second))])
    args = anthropic_args(tmp_path, dataset, client)

    result = cli.anthropic_command(args)

    assert result == 0
    rows = read_jsonl(args.out)
    assert [row["example_id"] for row in rows] == ["ex2"]
    error_rows = read_jsonl(args.errors)
    assert len(error_rows) == 1
    assert error_rows[0]["example_id"] == "ex1"
    assert error_rows[0]["attempts"] == 1
    assert error_rows[0]["retryable"] is False
    assert error_rows[0]["retry_errors"][0]["message"] == "rate limit"


def test_cli_anthropic_logs_malformed_json(tmp_path: Path) -> None:
    raw = "AITA? First. Second. Third."
    dataset = tmp_path / "data.csv"
    dataset.write_text("id,original_post\nex1,%s\n" % raw, encoding="utf-8")
    args = anthropic_args(tmp_path, dataset, FakeAnthropicClient([anthropic_message("{not json}")]))
    args.llm_retries = 0

    result = cli.anthropic_command(args)

    assert result == 0
    assert read_jsonl(args.out) == []
    error_rows = read_jsonl(args.errors)
    assert len(error_rows) == 1
    assert "model_output is not valid JSON" in error_rows[0]["message"]
    assert error_rows[0]["attempts"] == 1
    assert error_rows[0]["retryable"] is True
    assert error_rows[0]["retry_errors"][0]["error_type"] == "LLMIngestError"


def test_cli_anthropic_logs_max_tokens_response(tmp_path: Path) -> None:
    raw = "AITA? First. Second. Third."
    dataset = tmp_path / "data.csv"
    dataset.write_text("id,original_post\nex1,%s\n" % raw, encoding="utf-8")
    client = FakeAnthropicClient(
        [
            anthropic_message(valid_payload(raw), stop_reason="max_tokens"),
            anthropic_message(valid_payload(raw)),
        ]
    )
    args = anthropic_args(
        tmp_path,
        dataset,
        client,
    )
    args.llm_retries = 3

    result = cli.anthropic_command(args)

    assert result == 0
    assert len(client.messages.calls) == 1
    assert read_jsonl(args.out) == []
    error_rows = read_jsonl(args.errors)
    assert len(error_rows) == 1
    assert error_rows[0]["stop_reason"] == "max_tokens"
    assert error_rows[0]["attempts"] == 1
    assert error_rows[0]["retryable"] is False
    assert "max_tokens" in error_rows[0]["message"]
    assert len(read_jsonl(args.raw_responses)) == 1


def test_cli_anthropic_refusal_response_does_not_retry(tmp_path: Path) -> None:
    raw = "AITA? First. Second. Third."
    dataset = tmp_path / "data.csv"
    dataset.write_text("id,original_post\nex1,%s\n" % raw, encoding="utf-8")
    client = FakeAnthropicClient(
        [
            anthropic_message(valid_payload(raw), stop_reason="refusal"),
            anthropic_message(valid_payload(raw)),
        ]
    )
    args = anthropic_args(tmp_path, dataset, client)
    args.llm_retries = 3

    result = cli.anthropic_command(args)

    assert result == 0
    assert len(client.messages.calls) == 1
    error_rows = read_jsonl(args.errors)
    assert len(error_rows) == 1
    assert error_rows[0]["stop_reason"] == "refusal"
    assert error_rows[0]["attempts"] == 1
    assert error_rows[0]["retryable"] is False


def test_cli_anthropic_retries_malformed_json_then_writes_valid_row(tmp_path: Path, capsys) -> None:
    raw = "AITA? First. Second. Third."
    dataset = tmp_path / "data.csv"
    dataset.write_text("id,original_post\nex1,%s\n" % raw, encoding="utf-8")
    client = FakeAnthropicClient([anthropic_message("{not json}"), anthropic_message(valid_payload(raw))])
    args = anthropic_args(tmp_path, dataset, client)
    args.progress = "log"

    result = cli.anthropic_command(args)
    captured = capsys.readouterr()

    assert result == 0
    assert len(client.messages.calls) == 2
    assert len(read_jsonl(args.out)) == 1
    assert read_jsonl(args.errors) == []
    raw_rows = read_jsonl(args.raw_responses)
    assert [row["attempt"] for row in raw_rows] == [1, 2]
    assert [row["max_attempts"] for row in raw_rows] == [2, 2]
    assert "status=retry" in captured.err
    assert "written=1 generated=1 cached=0 errors=0 raw_responses=2 retries=1" in captured.out


def test_cli_anthropic_retries_alignment_error_then_writes_valid_row(tmp_path: Path) -> None:
    raw = "AITA? First. Second. Third."
    dataset = tmp_path / "data.csv"
    dataset.write_text("id,original_post\nex1,%s\n" % raw, encoding="utf-8")
    bad_payload = {
        "atomic_units": [{"text": "rewritten text", "section_type": "title"}],
        "shards": [{"unit_ids": [1], "section_role": "setup"}],
        "warnings": [],
    }
    client = FakeAnthropicClient([anthropic_message(bad_payload), anthropic_message(valid_payload(raw))])
    args = anthropic_args(tmp_path, dataset, client)

    result = cli.anthropic_command(args)

    assert result == 0
    assert len(client.messages.calls) == 2
    assert len(read_jsonl(args.out)) == 1
    assert read_jsonl(args.errors) == []


def test_cli_anthropic_retries_validation_gap_then_writes_valid_row(tmp_path: Path) -> None:
    raw = "AITA? First. Second. Third. Fourth."
    dataset = tmp_path / "data.csv"
    dataset.write_text("id,original_post\nex1,%s\n" % raw, encoding="utf-8")
    bad_payload = {
        "atomic_units": [
            {"text": "AITA?", "section_type": "title"},
            {"text": "First.", "section_type": "body"},
            {"text": "Third.", "section_type": "body"},
            {"text": "Fourth.", "section_type": "body"},
        ],
        "shards": [
            {"unit_ids": [1], "section_role": "setup"},
            {"unit_ids": [2], "section_role": "main_event"},
            {"unit_ids": [3], "section_role": "background_context"},
            {"unit_ids": [4], "section_role": "current_conflict"},
        ],
        "warnings": [],
    }
    client = FakeAnthropicClient([anthropic_message(bad_payload), anthropic_message(valid_payload(raw))])
    args = anthropic_args(tmp_path, dataset, client)

    result = cli.anthropic_command(args)

    assert result == 0
    assert len(client.messages.calls) == 2
    assert len(read_jsonl(args.out)) == 1
    assert read_jsonl(args.errors) == []


def test_cli_anthropic_resume_reuses_matching_cached_row(tmp_path: Path, capsys) -> None:
    raw = "AITA? First. Second. Third."
    dataset = tmp_path / "data.csv"
    dataset.write_text("id,original_post\nex1,%s\n" % raw, encoding="utf-8")
    first_client = FakeAnthropicClient([anthropic_message(valid_payload(raw))])
    first_args = anthropic_args(tmp_path, dataset, first_client)

    assert cli.anthropic_command(first_args) == 0
    cached_rows = read_jsonl(first_args.out)
    assert cached_rows[0]["request_fingerprint"].startswith("sha256:")

    second_client = FakeAnthropicClient([Exception("should not call Anthropic")])
    second_args = anthropic_args(tmp_path, dataset, second_client)
    second_args.resume = True
    second_args.progress = "log"

    result = cli.anthropic_command(second_args)
    captured = capsys.readouterr()

    assert result == 0
    assert len(second_client.messages.calls) == 0
    assert read_jsonl(second_args.out) == cached_rows
    assert read_jsonl(second_args.errors) == []
    assert len(read_jsonl(second_args.raw_responses)) == 1
    assert "status=cached" in captured.err
    assert "written=1 generated=0 cached=1 errors=0 raw_responses=0 retries=0" in captured.out


def test_cli_anthropic_resume_reuses_tmp_when_final_missing(tmp_path: Path) -> None:
    raw = "AITA? First. Second. Third."
    dataset = tmp_path / "data.csv"
    dataset.write_text("id,original_post\nex1,%s\n" % raw, encoding="utf-8")
    first_args = anthropic_args(tmp_path, dataset, FakeAnthropicClient([anthropic_message(valid_payload(raw))]))

    assert cli.anthropic_command(first_args) == 0
    cached_rows = read_jsonl(first_args.out)
    tmp_out = cli._resume_temp_path(first_args.out)
    first_args.out.rename(tmp_out)

    second_client = FakeAnthropicClient([Exception("should not call Anthropic")])
    second_args = anthropic_args(tmp_path, dataset, second_client)
    second_args.resume = True

    assert cli.anthropic_command(second_args) == 0
    assert len(second_client.messages.calls) == 0
    assert read_jsonl(second_args.out) == cached_rows


def test_cli_anthropic_cached_resume_does_not_require_api_key(tmp_path: Path, monkeypatch) -> None:
    raw = "AITA? First. Second. Third."
    dataset = tmp_path / "data.csv"
    dataset.write_text("id,original_post\nex1,%s\n" % raw, encoding="utf-8")
    first_args = anthropic_args(tmp_path, dataset, FakeAnthropicClient([anthropic_message(valid_payload(raw))]))

    assert cli.anthropic_command(first_args) == 0
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    second_args = anthropic_args(tmp_path, dataset, FakeAnthropicClient([]))
    second_args.resume = True
    second_args._anthropic_client = None

    assert cli.anthropic_command(second_args) == 0
    assert read_jsonl(second_args.errors) == []


def test_cli_anthropic_concurrency_preserves_output_order(tmp_path: Path) -> None:
    first = "AITA? First. Second. Third."
    second = "AITA? Alpha. Beta. Gamma."
    dataset = tmp_path / "data.csv"
    dataset.write_text("id,original_post\nex1,%s\nex2,%s\n" % (first, second), encoding="utf-8")
    client = SlowFakeAnthropicClient(
        [anthropic_message(valid_payload(first)), anthropic_message(valid_payload(second))],
        delay=0.2,
    )
    args = anthropic_args(tmp_path, dataset, client)
    args.concurrency = 2

    start = time.monotonic()
    assert cli.anthropic_command(args) == 0
    elapsed = time.monotonic() - start

    assert elapsed < 0.35
    rows = read_jsonl(args.out)
    assert [row["example_id"] for row in rows] == ["ex1", "ex2"]
    assert len(client.messages.calls) == 2


def test_cli_anthropic_content_cache_rewrites_provenance(tmp_path: Path) -> None:
    raw = "AITA? First. Second. Third."
    first_dataset = tmp_path / "first.csv"
    second_dataset = tmp_path / "second.csv"
    first_dataset.write_text("id,original_post\nex1,%s\n" % raw, encoding="utf-8")
    second_dataset.write_text("id,prompt\nex2,%s\n" % raw, encoding="utf-8")
    first_args = anthropic_args(tmp_path, first_dataset, FakeAnthropicClient([anthropic_message(valid_payload(raw))]))

    assert cli.anthropic_command(first_args) == 0

    second_client = FakeAnthropicClient([Exception("should not call Anthropic")])
    second_args = anthropic_args(tmp_path, second_dataset, second_client)
    second_args.dataset_name = "AITA-YTA"
    second_args.source_field = "prompt"
    second_args.out = tmp_path / "second_shards.jsonl"
    second_args.errors = tmp_path / "second_errors.jsonl"
    second_args.raw_responses = tmp_path / "second_raw.jsonl"
    second_args.resume = True
    second_args._resume_cache = cli._load_resume_cache(first_args.out)

    assert cli.anthropic_command(second_args) == 0
    assert len(second_client.messages.calls) == 0
    row = read_jsonl(second_args.out)[0]
    assert row["dataset_name"] == "AITA-YTA"
    assert row["example_id"] == "ex2"
    assert row["source_text_field"] == "prompt"


def test_cli_anthropic_resume_ignores_stale_cached_row(tmp_path: Path) -> None:
    raw = "AITA? First. Second. Third."
    dataset = tmp_path / "data.csv"
    dataset.write_text("id,original_post\nex1,%s\n" % raw, encoding="utf-8")
    first_args = anthropic_args(tmp_path, dataset, FakeAnthropicClient([anthropic_message(valid_payload(raw))]))

    assert cli.anthropic_command(first_args) == 0
    old_fingerprint = read_jsonl(first_args.out)[0]["request_fingerprint"]

    second_client = FakeAnthropicClient([anthropic_message(valid_payload(raw))])
    second_args = anthropic_args(tmp_path, dataset, second_client)
    second_args.resume = True
    second_args.model = "claude-other-test"

    assert cli.anthropic_command(second_args) == 0
    assert len(second_client.messages.calls) == 1
    new_fingerprint = read_jsonl(second_args.out)[0]["request_fingerprint"]
    assert new_fingerprint != old_fingerprint


def test_request_fingerprint_changes_when_semantics_change() -> None:
    request = {
        "dataset_name": "AITA-NTA-OG",
        "example_id": "ex1",
        "source_text_field": "original_post",
        "run_id": "run1",
        "segmentation_version": "seg_v1",
        "raw_source_text": "AITA? First. Second. Third.",
        "messages": [{"role": "system", "content": "prompt v1"}, {"role": "user", "content": "raw"}],
    }
    base = cli._request_fingerprint(request, "claude-test", 4096, 0.0, output_schema={"type": "object"})

    assert cli._request_fingerprint(dict(request), "claude-test", 4096, 0.0, output_schema={"type": "object"}) == base

    changed_raw = dict(request, raw_source_text="AITA? Different.")
    changed_prompt = dict(request, messages=[{"role": "system", "content": "prompt v2"}])

    assert cli._request_fingerprint(changed_raw, "claude-test", 4096, 0.0, output_schema={"type": "object"}) != base
    assert cli._request_fingerprint(changed_prompt, "claude-test", 4096, 0.0, output_schema={"type": "object"}) != base
    assert cli._request_fingerprint(request, "claude-other", 4096, 0.0, output_schema={"type": "object"}) != base
    assert cli._request_fingerprint(request, "claude-test", 8192, 0.0, output_schema={"type": "object"}) != base
    assert cli._request_fingerprint(request, "claude-test", 4096, 0.2, output_schema={"type": "object"}) != base
    assert cli._request_fingerprint(request, "claude-test", 4096, 0.0, output_schema={"type": "array"}) != base


def test_cli_anthropic_bulk_manifest_uses_default_source_field(tmp_path: Path) -> None:
    raw = "AITA? Flipped. Story. Here."
    dataset = tmp_path / "flip.csv"
    dataset.write_text("id,original_post,flipped_story\nex1,wrong,%s\n" % raw, encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "dataset": str(dataset),
                "dataset_name": "AITA-NTA-FLIP",
                "out": str(tmp_path / "flip_shards.jsonl"),
                "errors": str(tmp_path / "flip_errors.jsonl"),
                "raw_responses": str(tmp_path / "flip_raw.jsonl"),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = FakeAnthropicClient([anthropic_message(valid_payload(raw))])
    args = argparse.Namespace(
        manifest=manifest,
        run_id="run1",
        segmentation_version="seg_v1",
        created_at="2026-04-27T00:00:00Z",
        limit=None,
        model="claude-test",
        max_tokens=4096,
        temperature=0.0,
        llm_retries=1,
        anthropic_max_retries=2,
        concurrency=1,
        progress="off",
        resume=True,
        resume_include_temp=True,
        raw_responses_mode=None,
        _anthropic_client=client,
    )

    assert cli.anthropic_bulk_command(args) == 0
    assert client.messages.calls[0]["messages"][0]["content"] == raw
    rows = read_jsonl(tmp_path / "flip_shards.jsonl")
    assert rows[0]["source_text_field"] == "flipped_story"


def test_cli_anthropic_batch_submit_sends_only_cache_misses(tmp_path: Path) -> None:
    first = "AITA? First. Second. Third."
    second = "AITA? Alpha. Beta. Gamma."
    dataset = tmp_path / "data.csv"
    dataset.write_text("id,original_post\nex1,%s\nex2,%s\n" % (first, second), encoding="utf-8")
    first_args = anthropic_args(tmp_path, dataset, FakeAnthropicClient([anthropic_message(valid_payload(first))]))
    first_args.limit = 1
    assert cli.anthropic_command(first_args) == 0
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "dataset": str(dataset),
                "dataset_name": "AITA-NTA-OG",
                "source_field": "original_post",
                "out": str(first_args.out),
                "errors": str(first_args.errors),
                "raw_responses": str(first_args.raw_responses),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = FakeBatchClient()
    state = tmp_path / "batch_state.json"
    args = argparse.Namespace(
        manifest=manifest,
        batch_state=state,
        run_id="run1",
        segmentation_version="seg_v1",
        created_at="2026-04-27T00:00:00Z",
        limit=None,
        model="claude-test",
        max_tokens=4096,
        temperature=0.0,
        llm_retries=1,
        anthropic_max_retries=2,
        concurrency=1,
        progress="off",
        resume=True,
        resume_include_temp=True,
        raw_responses_mode=None,
        _anthropic_client=client,
    )

    assert cli.anthropic_batch_submit_command(args) == 0
    assert len(client.messages.batches.create_calls[0]["requests"]) == 1
    state_data = json.loads(state.read_text(encoding="utf-8"))
    assert state_data["request_count"] == 1
    assert state_data["requests"][0]["request"]["example_id"] == "ex2"


def test_cli_anthropic_batch_submit_all_cached_does_not_require_api_key(tmp_path: Path, monkeypatch) -> None:
    raw = "AITA? First. Second. Third."
    dataset = tmp_path / "data.csv"
    dataset.write_text("id,original_post\nex1,%s\n" % raw, encoding="utf-8")
    first_args = anthropic_args(tmp_path, dataset, FakeAnthropicClient([anthropic_message(valid_payload(raw))]))
    assert cli.anthropic_command(first_args) == 0
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "dataset": str(dataset),
                "dataset_name": "AITA-NTA-OG",
                "source_field": "original_post",
                "out": str(first_args.out),
                "errors": str(first_args.errors),
                "raw_responses": str(first_args.raw_responses),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    state = tmp_path / "batch_state.json"
    args = argparse.Namespace(
        manifest=manifest,
        batch_state=state,
        run_id="run1",
        segmentation_version="seg_v1",
        created_at="2026-04-27T00:00:00Z",
        limit=None,
        model="claude-test",
        max_tokens=4096,
        temperature=0.0,
        llm_retries=1,
        anthropic_max_retries=2,
        concurrency=1,
        progress="off",
        resume=True,
        resume_include_temp=True,
        raw_responses_mode=None,
        _anthropic_client=None,
    )

    assert cli.anthropic_batch_submit_command(args) == 0
    state_data = json.loads(state.read_text(encoding="utf-8"))
    assert state_data["batch_id"] is None
    assert state_data["request_count"] == 0


def test_cli_anthropic_batch_collect_handles_unordered_success_and_failures(tmp_path: Path) -> None:
    raws = [
        "AITA? One. Two. Three.",
        "AITA? Four. Five. Six.",
        "AITA? Seven. Eight. Nine.",
        "AITA? Ten. Eleven. Twelve.",
    ]
    dataset = tmp_path / "data.csv"
    dataset.write_text(
        "id,original_post\n" + "\n".join("ex%d,%s" % (idx, raw) for idx, raw in enumerate(raws, start=1)) + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "dataset": str(dataset),
                "dataset_name": "AITA-NTA-OG",
                "source_field": "original_post",
                "out": str(tmp_path / "batch_shards.jsonl"),
                "errors": str(tmp_path / "batch_errors.jsonl"),
                "raw_responses": str(tmp_path / "batch_raw.jsonl"),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    state = tmp_path / "batch_state.json"
    submit_client = FakeBatchClient()
    submit_args = argparse.Namespace(
        manifest=manifest,
        batch_state=state,
        run_id="run1",
        segmentation_version="seg_v1",
        created_at="2026-04-27T00:00:00Z",
        limit=None,
        model="claude-test",
        max_tokens=4096,
        temperature=0.0,
        llm_retries=1,
        anthropic_max_retries=2,
        concurrency=1,
        progress="off",
        resume=True,
        resume_include_temp=True,
        raw_responses_mode=None,
        _anthropic_client=submit_client,
    )
    assert cli.anthropic_batch_submit_command(submit_args) == 0
    state_data = json.loads(state.read_text(encoding="utf-8"))
    requests = state_data["requests"]
    results = [
        {
            "custom_id": requests[2]["custom_id"],
            "result": {"type": "succeeded", "message": anthropic_message(valid_payload(raws[2]), stop_reason="max_tokens")},
        },
        {
            "custom_id": requests[0]["custom_id"],
            "result": {"type": "succeeded", "message": anthropic_message(valid_payload(raws[0]))},
        },
        {
            "custom_id": requests[1]["custom_id"],
            "result": {"type": "succeeded", "message": anthropic_message("{not json}")},
        },
        {
            "custom_id": requests[3]["custom_id"],
            "result": {"type": "errored", "error": {"type": "server_error", "message": "boom"}},
        },
    ]
    collect_args = argparse.Namespace(
        batch_state=state,
        raw_responses_mode="overwrite",
        resume_include_temp=True,
        anthropic_max_retries=2,
        _anthropic_client=FakeBatchClient(results),
    )

    assert cli.anthropic_batch_collect_command(collect_args) == 0
    assert [row["example_id"] for row in read_jsonl(tmp_path / "batch_shards.jsonl")] == ["ex1"]
    error_rows = read_jsonl(tmp_path / "batch_errors.jsonl")
    assert len(error_rows) == 3
    assert {row["error_type"] for row in error_rows} == {"LLMIngestError", "AnthropicRunError"}
    assert len(read_jsonl(tmp_path / "batch_raw.jsonl")) == 3


def test_cli_anthropic_log_progress_reports_counts(tmp_path: Path, capsys) -> None:
    first = "AITA? First. Second. Third."
    second = "AITA? Alpha. Beta. Gamma."
    dataset = tmp_path / "data.csv"
    dataset.write_text("id,original_post\nex1,%s\nex2,%s\n" % (first, second), encoding="utf-8")
    bad_payload = {
        "atomic_units": [{"text": "AITA?", "section_type": "title"}, {"text": "rewritten text", "section_type": "body"}],
        "shards": [
            {"unit_ids": [1], "section_role": "setup"},
            {"unit_ids": [2], "section_role": "main_event"},
            {"unit_ids": [1], "section_role": "background_context"},
            {"unit_ids": [2], "section_role": "current_conflict"},
        ],
        "warnings": [],
    }
    client = FakeAnthropicClient([anthropic_message(valid_payload(first)), anthropic_message(bad_payload)])
    args = anthropic_args(tmp_path, dataset, client)
    args.progress = "log"
    args.llm_retries = 0

    result = cli.anthropic_command(args)
    captured = capsys.readouterr()

    assert result == 0
    assert "sharding 1/2 example_id=ex1 status=ok ok=1 errors=0" in captured.err
    assert "sharding 2/2 example_id=ex2 status=error ok=1 errors=1" in captured.err
    assert "written=1 generated=1 cached=0 errors=1 raw_responses=2 retries=0" in captured.out
    assert "top_errors:" in captured.out
    assert "AlignmentError: 1" in captured.out


def test_cli_anthropic_missing_key_returns_error(tmp_path: Path) -> None:
    dataset = tmp_path / "data.csv"
    dataset.write_text("id,original_post\nex1,AITA? First. Second. Third.\n", encoding="utf-8")
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)
    out = tmp_path / "shards.jsonl"
    errors = tmp_path / "errors.jsonl"

    result = run_cli(
        "anthropic",
        "--dataset",
        str(dataset),
        "--dataset-name",
        "AITA-NTA-OG",
        "--source-field",
        "original_post",
        "--run-id",
        "run1",
        "--out",
        str(out),
        "--errors",
        str(errors),
        cwd=REPO_ROOT,
        env=env,
    )

    assert result.returncode == 1
    assert "ANTHROPIC_API_KEY" in result.stderr
    assert not out.exists()
    assert not errors.exists()


def test_cli_validate_invalid_file_returns_error(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text('{"status":"ok"}\n', encoding="utf-8")

    result = run_cli("validate", "--input", str(invalid), cwd=REPO_ROOT)

    assert result.returncode == 1
    assert "missing required field" in result.stderr
