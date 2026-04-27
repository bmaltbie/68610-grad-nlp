from pathlib import Path
import json
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]


def run_cli(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "decomposition.cli", *args],
        cwd=str(cwd),
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


def test_cli_validate_invalid_file_returns_error(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text('{"status":"ok"}\n', encoding="utf-8")

    result = run_cli("validate", "--input", str(invalid), cwd=REPO_ROOT)

    assert result.returncode == 1
    assert "missing required field" in result.stderr
