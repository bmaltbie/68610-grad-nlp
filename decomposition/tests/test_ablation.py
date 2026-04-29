from pathlib import Path
import argparse
import json

import pytest

from decomposition import ablation
from decomposition.datasets import DatasetError
from decomposition.datasets import DatasetExample
from decomposition.llm_io import build_atomic_request
from decomposition.schema import AtomicUnit, ShardRecord, validate_record
from decomposition.shard_planner import plan_shards


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class FakeResponses:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("unexpected OpenAI call")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeFiles:
    def __init__(self, output_text=""):
        self.create_calls = []
        self.output_text = output_text

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return {"id": "file_test"}

    def content(self, file_id):
        assert file_id == "file_output"
        return self.output_text


class FakeBatches:
    def __init__(self):
        self.create_calls = []

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return {"id": "batch_test", "status": "validating", "input_file_id": kwargs["input_file_id"]}

    def retrieve(self, batch_id):
        return {"id": batch_id, "status": "completed", "output_file_id": "file_output"}


class FakeOpenAIClient:
    def __init__(self, responses=None, batch_output=""):
        self.responses = FakeResponses(responses)
        self.files = FakeFiles(batch_output)
        self.batches = FakeBatches()


def atomic_payload(raw: str):
    return {
        "atomic_units": [
            {"unit_id": index, "text": unit.text, "section_type": unit.section_type}
            for index, unit in enumerate(units_for(raw), start=1)
        ],
        "warnings": [],
    }


def openai_response(payload, model="gpt-test"):
    return {
        "id": "resp_test",
        "model": model,
        "status": "completed",
        "usage": {"input_tokens": 1, "output_tokens": 2},
        "output_text": payload if isinstance(payload, str) else json.dumps(payload),
    }


def dataset(tmp_path: Path, raws):
    path = tmp_path / "data.csv"
    path.write_text(
        "id,prompt\n" + "\n".join("ex%d,%s" % (index, raw) for index, raw in enumerate(raws, start=1)) + "\n",
        encoding="utf-8",
    )
    return path


def args_for(tmp_path: Path, data_path: Path, client: FakeOpenAIClient):
    return argparse.Namespace(
        dataset=data_path,
        dataset_name="AITA-YTA",
        source_field="prompt",
        run_id="run1",
        segmentation_version="seg_v1",
        created_at="2026-04-29T00:00:00Z",
        limit=None,
        provider="openai",
        model="gpt-test",
        target_turns="4,6,8",
        out_dir=tmp_path / "artifacts",
        seed_shards=[],
        resume=False,
        resume_include_temp=True,
        concurrency=1,
        llm_retries=0,
        provider_max_retries=2,
        openai_max_retries=2,
        max_tokens=4096,
        temperature=0.0,
        progress="off",
        shard_policy="natural_dp_v1",
        _openai_client=client,
    )


def units_for(raw: str):
    units = []
    cursor = 0
    for token_index, token in enumerate(raw.split(" "), start=1):
        start = raw.find(token, cursor)
        end = start + len(token)
        units.append(AtomicUnit(token_index, raw[start:end], start, end, "title" if token_index == 1 else "body"))
        cursor = end
    return units


def write_seed(path: Path, raw: str) -> None:
    units = units_for(raw)
    record = ShardRecord(
        example_id="seed-ex",
        dataset_name="AITA-YTA",
        source_text_field="prompt",
        run_id="seed-run",
        segmentation_version="seg_v1",
        segmenter_model="gpt-old",
        created_at="2026-04-29T00:00:00Z",
        raw_source_text=raw,
        target_turns=4,
        status="ok",
        atomic_units=units,
        shards=plan_shards(raw, units, 4),
        warnings=[],
    )
    validate_record(record)
    path.write_text(json.dumps(record.to_dict()) + "\n", encoding="utf-8")


def test_shard_ablation_seed_shards_imports_atomic_units_without_openai_calls(tmp_path: Path) -> None:
    raw = "AITA? One. Two. Three. Four. Five. Six. Seven."
    data_path = dataset(tmp_path, [raw])
    seed = tmp_path / "seed.jsonl"
    write_seed(seed, raw)
    client = FakeOpenAIClient()
    args = args_for(tmp_path, data_path, client)
    args.seed_shards = [seed]

    assert ablation.run_shard_ablation(args) == 0

    assert client.responses.calls == []
    assert len(read_jsonl(tmp_path / "artifacts" / "atomic_units.AITA-YTA.openai.jsonl")) == 1
    summary = json.loads((tmp_path / "artifacts" / "shard_ablation.AITA-YTA.openai.summary.json").read_text(encoding="utf-8"))
    assert summary["atomic"]["generated"] == 0
    assert summary["atomic"]["cached"] == 1
    for target in (4, 6, 8):
        rows = read_jsonl(tmp_path / "artifacts" / ("shards.AITA-YTA.openai.k%d.jsonl" % target))
        assert len(rows[0]["shards"]) == target


def test_shard_ablation_replays_raw_response_sidecar_before_openai_calls(tmp_path: Path) -> None:
    raw = "AITA? One. Two. Three. Four. Five. Six. Seven."
    data_path = dataset(tmp_path, [raw])
    client = FakeOpenAIClient()
    args = args_for(tmp_path, data_path, client)
    args.resume = True
    paths = ablation.ablation_paths(args.out_dir, args.dataset_name, args.provider, args.segmentation_version)
    paths.raw_responses.parent.mkdir(parents=True)
    example = DatasetExample("ex1", "AITA-YTA", "prompt", raw, 1)
    request = build_atomic_request(example, args.run_id, args.segmentation_version, args.created_at)
    raw_response = {
        "request_id": request["request_id"],
        "provider": "openai",
        "segmenter_model": "gpt-test",
        "status": "completed",
        "created_at": "2026-04-29T00:00:00Z",
        "model_output": json.dumps(atomic_payload(raw)),
    }
    paths.raw_responses.write_text(json.dumps(raw_response) + "\n", encoding="utf-8")

    assert ablation.run_shard_ablation(args) == 0

    assert client.responses.calls == []
    assert len(read_jsonl(paths.atomic)) == 1


def test_shard_ablation_reuses_final_atomic_cache_on_rerun(tmp_path: Path) -> None:
    raw = "AITA? One. Two. Three. Four. Five. Six. Seven."
    data_path = dataset(tmp_path, [raw])
    first_client = FakeOpenAIClient([openai_response(atomic_payload(raw))])
    first_args = args_for(tmp_path, data_path, first_client)
    assert ablation.run_shard_ablation(first_args) == 0

    second_client = FakeOpenAIClient()
    second_args = args_for(tmp_path, data_path, second_client)
    second_args.resume = True

    assert ablation.run_shard_ablation(second_args) == 0
    assert second_client.responses.calls == []


def test_shard_ablation_short_rows_emit_ineligible_target_rows(tmp_path: Path) -> None:
    raw = "AITA? One. Two."
    data_path = dataset(tmp_path, [raw])
    client = FakeOpenAIClient([openai_response(atomic_payload(raw))])
    args = args_for(tmp_path, data_path, client)

    assert ablation.run_shard_ablation(args) == 0

    for target in (4, 6, 8):
        row = read_jsonl(tmp_path / "artifacts" / ("shards.AITA-YTA.openai.k%d.jsonl" % target))[0]
        assert row["status"] == "ineligible_target_shards"
        assert row["atomic_units"]
        assert row["shards"] == []


def test_shard_ablation_rejects_unknown_shard_policy_before_writing_empty_artifacts(tmp_path: Path) -> None:
    raw = "AITA? One. Two. Three. Four. Five. Six. Seven."
    data_path = dataset(tmp_path, [raw])
    seed = tmp_path / "seed.jsonl"
    write_seed(seed, raw)
    args = args_for(tmp_path, data_path, FakeOpenAIClient())
    args.seed_shards = [seed]
    args.shard_policy = "not_a_policy"

    with pytest.raises(DatasetError, match="unsupported --shard-policy"):
        ablation.run_shard_ablation(args)

    assert not (tmp_path / "artifacts" / "shards.AITA-YTA.openai.k4.jsonl").exists()


def test_shard_ablation_batch_submit_sends_only_atomic_cache_misses(tmp_path: Path) -> None:
    raws = [
        "AITA? One. Two. Three. Four. Five. Six. Seven.",
        "AITA? Alpha. Beta. Gamma. Delta. Epsilon. Zeta. Eta.",
    ]
    data_path = dataset(tmp_path, raws)
    seed = tmp_path / "seed.jsonl"
    write_seed(seed, raws[0])
    client = FakeOpenAIClient()
    args = args_for(tmp_path, data_path, client)
    args.seed_shards = [seed]
    args.batch_state = tmp_path / "batch_state.json"
    args.batch_input = tmp_path / "batch_input.jsonl"

    assert ablation.submit_shard_ablation_batch(args) == 0

    assert len(read_jsonl(args.batch_input)) == 1
    state = json.loads(args.batch_state.read_text(encoding="utf-8"))
    assert state["request_count"] == 1
    assert state["requests"][0]["request"]["example_id"] == "ex2"


def test_shard_ablation_batch_collect_handles_unordered_success_and_failure_rows(tmp_path: Path) -> None:
    raws = [
        "AITA? One. Two. Three. Four. Five. Six. Seven.",
        "AITA? Alpha. Beta. Gamma. Delta. Epsilon. Zeta. Eta.",
        "AITA? Red. Blue. Green. Yellow. Purple. White. Black.",
    ]
    data_path = dataset(tmp_path, raws)
    submit_args = args_for(tmp_path, data_path, FakeOpenAIClient())
    submit_args.batch_state = tmp_path / "batch_state.json"
    submit_args.batch_input = tmp_path / "batch_input.jsonl"
    assert ablation.submit_shard_ablation_batch(submit_args) == 0
    state = json.loads(submit_args.batch_state.read_text(encoding="utf-8"))
    requests = state["requests"]
    output_rows = [
        {
            "custom_id": requests[2]["custom_id"],
            "response": {"status_code": 200, "body": openai_response(atomic_payload(raws[2]))},
        },
        {
            "custom_id": requests[0]["custom_id"],
            "response": {"status_code": 500, "body": {"error": {"message": "server boom"}}},
        },
        {
            "custom_id": requests[1]["custom_id"],
            "response": {"status_code": 200, "body": openai_response(atomic_payload(raws[1]))},
        },
    ]
    collect_args = argparse.Namespace(
        batch_state=submit_args.batch_state,
        raw_responses_mode="overwrite",
        resume_include_temp=True,
        openai_max_retries=2,
        _openai_client=FakeOpenAIClient(batch_output="\n".join(json.dumps(row) for row in output_rows) + "\n"),
    )

    assert ablation.collect_shard_ablation_batch(collect_args) == 0

    k4_rows = read_jsonl(tmp_path / "artifacts" / "shards.AITA-YTA.openai.k4.jsonl")
    assert [row["example_id"] for row in k4_rows] == ["ex2", "ex3"]
    assert len(read_jsonl(tmp_path / "artifacts" / "run_errors.AITA-YTA.openai.atomic.jsonl")) == 1
    raw_path = tmp_path / "artifacts" / "seg_v1_openai_atomic_responses.AITA-YTA.jsonl"
    raw_before = raw_path.read_text(encoding="utf-8")

    rerun_args = argparse.Namespace(
        batch_state=submit_args.batch_state,
        raw_responses_mode="overwrite",
        resume_include_temp=True,
        openai_max_retries=2,
        _openai_client=FakeOpenAIClient(batch_output=""),
    )

    assert ablation.collect_shard_ablation_batch(rerun_args) == 0
    assert raw_path.read_text(encoding="utf-8") == raw_before
