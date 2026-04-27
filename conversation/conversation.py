#!/usr/bin/env python
"""Experiment driver.

Walks ``seed.json`` × ``shards.json`` (both colocated next to this file)
and runs one Petri-style audit per (seed, shard-object) pair.  For each
pair the seed's prompt is prepended with the shard strings as a bulleted
context block, fed to the auditor as its seed instruction, and the
resulting auditor/target transcript is written to
``outputs/<seed_identifier>/`` (relative to the current working
directory) in Inspect AI's native ``.eval`` format.

No judge is run — this script only produces conversation transcripts.

The target model is **not** given a pre-set system prompt — the auditor's
built-in ``set_system_message`` guidance asks it to default to a
general-purpose HHH assistant when the seed does not imply a specialised
role.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# ---------- configuration ----------

_HERE = Path(__file__).resolve().parent
SEED_PATH = _HERE / "seed.json"
SHARDS_PATH = _HERE / "shards.json"
OUTPUT_ROOT = Path("outputs")

AUDITOR_MODEL = "openai/gpt-5.4-nano"
TARGET_MODEL = "openai/gpt-5.4-nano"

MAX_TURNS = 20


# ---------- loaders ----------


def load_seeds() -> list[dict[str, Any]]:
    raw = json.loads(SEED_PATH.read_text())
    _validate_seeds(raw)
    return raw


def load_shards() -> list[dict[str, Any]]:
    raw = json.loads(SHARDS_PATH.read_text())
    _validate_shards(raw)
    return raw


def _validate_seeds(raw: Any) -> None:
    if not isinstance(raw, list):
        raise ValueError(f"{SEED_PATH}: top-level JSON must be a list")
    seen: set[str] = set()
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(
                f"seed[{i}]: each entry must be an object with "
                "'identifier', 'description', and 'prompt'"
            )
        for field in ("identifier", "description", "prompt"):
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"seed[{i}]: missing or empty required string field {field!r}"
                )
        identifier = entry["identifier"]
        if identifier in seen:
            raise ValueError(
                f"seed[{i}]: duplicate identifier {identifier!r} "
                "(must be unique across seed.json)"
            )
        seen.add(identifier)


def _validate_shards(raw: Any) -> None:
    if not isinstance(raw, list):
        raise ValueError(f"{SHARDS_PATH}: top-level JSON must be a list")
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"shard[{i}]: each entry must be an object")
        shards = entry.get("shards")
        if not isinstance(shards, list) or not all(isinstance(s, str) for s in shards):
            raise ValueError(
                f"shard[{i}]: 'shards' must be a list of strings"
            )


def build_modified_instruction(seed_prompt: str, shards: list[str]) -> str:
    """Prepend shard strings as a bulleted context block."""
    bullets = "\n".join(f"- {s}" for s in shards)
    return f"## Context:\n{bullets}\n\n## Instructions:\n{seed_prompt}"


# ---------- main ----------


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv()

    # imports after load_dotenv so provider keys are set for inspect-ai
    from inspect_ai import eval as inspect_eval
    from inspect_ai.dataset import MemoryDataset, Sample

    from auditor._task.audit import audit

    seeds = load_seeds()
    shards = load_shards()

    all_ok = True
    for seed in seeds:
        sid = seed["identifier"]
        out_dir = OUTPUT_ROOT / sid
        out_dir.mkdir(parents=True, exist_ok=True)

        seed_max_turns = int(seed.get("max_turns", MAX_TURNS))

        for j, shard_obj in enumerate(shards):
            modified = build_modified_instruction(
                seed["prompt"], shard_obj["shards"]
            )
            sample = Sample(
                id=f"{sid}-shard{j}",
                input=modified,
                metadata={
                    "seed_identifier": sid,
                    "seed_description": seed["description"],
                    "shard_index": j,
                },
            )
            task = audit(
                dataset=MemoryDataset([sample], name=f"{sid}-shard{j}"),
                max_turns=seed_max_turns,
            )
            print(f"[{sid} / shard {j}] starting…", file=sys.stderr)
            logs = inspect_eval(
                tasks=task,
                model_roles={
                    "auditor": AUDITOR_MODEL,
                    "target": TARGET_MODEL,
                },
                log_dir=str(out_dir),
                log_format="eval",
            )
            for log in logs:
                print(
                    f"[{sid} / shard {j}] status={log.status} "
                    f"location={log.location}",
                    file=sys.stderr,
                )
                if log.status != "success":
                    all_ok = False

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
