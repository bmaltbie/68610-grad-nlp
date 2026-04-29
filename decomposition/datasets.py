from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional
import csv


class DatasetError(Exception):
    """Raised when a CSV cannot be interpreted as a decomposition dataset."""

    pass


@dataclass
class DatasetExample:
    """One raw post plus the provenance needed for shard artifacts."""

    example_id: str
    dataset_name: str
    source_text_field: str
    raw_source_text: str
    row_number: int


def load_dataset(
    dataset_path: Path,
    dataset_name: str,
    source_text_field: str,
    limit: Optional[int] = None,
) -> Iterator[DatasetExample]:
    """Yield dataset rows with stable example ids and raw source text.

    The source CSVs are not perfectly uniform: some have an `id` column, while
    AITA-YTA relies on an unnamed index column. This loader centralizes that
    choice so downstream code can treat `example_id` as stable provenance.
    """
    with dataset_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise DatasetError(
                "dataset has no header: %s. Fix: provide a CSV with a header row containing the source text column."
                % dataset_path
            )
        if source_text_field not in reader.fieldnames:
            raise DatasetError(
                "source field %r not found in %s; fields=%s. Fix: pass one of these names to --source-field."
                % (source_text_field, dataset_path, reader.fieldnames)
            )
        count = 0
        for row_number, row in enumerate(reader, start=1):
            if limit is not None and count >= limit:
                break
            example_id = _example_id(row, row_number)
            raw_source_text = row.get(source_text_field) or ""
            yield DatasetExample(
                example_id=example_id,
                dataset_name=dataset_name,
                source_text_field=source_text_field,
                raw_source_text=raw_source_text,
                row_number=row_number,
            )
            count += 1


def _example_id(row: dict, row_number: int) -> str:
    """Choose the most stable id available for the current dataset row."""
    if row.get("id"):
        return str(row["id"])
    if row.get(""):
        return str(row[""])
    return str(row_number - 1)
