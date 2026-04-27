from pathlib import Path

import pytest

from decomposition.datasets import DatasetError, load_dataset


def test_load_dataset_uses_id_column(tmp_path: Path) -> None:
    path = tmp_path / "aita.csv"
    path.write_text("id,original_post\nabc123,Text here.\n", encoding="utf-8")

    rows = list(load_dataset(path, "AITA-NTA-OG", "original_post"))

    assert rows[0].example_id == "abc123"
    assert rows[0].dataset_name == "AITA-NTA-OG"
    assert rows[0].raw_source_text == "Text here."


def test_load_dataset_uses_unnamed_index_when_id_missing(tmp_path: Path) -> None:
    path = tmp_path / "yta.csv"
    path.write_text(",prompt\n7,Prompt text.\n", encoding="utf-8")

    rows = list(load_dataset(path, "AITA-YTA", "prompt"))

    assert rows[0].example_id == "7"
    assert rows[0].source_text_field == "prompt"


def test_load_dataset_errors_on_missing_source_field(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("id,body\nabc,Text.\n", encoding="utf-8")

    with pytest.raises(DatasetError):
        list(load_dataset(path, "AITA-NTA-OG", "original_post"))
