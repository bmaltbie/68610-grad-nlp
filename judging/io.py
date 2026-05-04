"""I/O for the judging stage.

- ``load_seeds`` reads ``conversation/seed.json`` and extracts the
  per-seed interrupt phrase.
- ``classify_auditor_turn`` labels an auditor message as
  ``shard | interrupt | aita_close`` using the live seed catalogue and a
  regex for the AITA close question.
- ``load_transcripts`` streams ``conversation_transcripts.jsonl``,
  drops ``succeeded: False`` records, computes ``auditor_turn_types``,
  and projects each row to a ``Transcript``.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Iterator, Optional

from judging.schemas import (
    AuditorTurnType,
    Transcript,
    TurnMessage,
)

log = logging.getLogger(__name__)

# Repo root, used to resolve the default seed catalogue path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEED_PATH = _REPO_ROOT / "conversation" / "seed.json"

# Matches the embedded interrupt phrase in seed prompts:
#   say, "wait im not finished yet"
# Curly quotes also appear in some seeds, so accept both straight and
# typographic double quotes.
_SEED_PHRASE_RE = re.compile(r"say,\s*[\"“”]([^\"“”]+)[\"“”]")

# Loose match for the AITA close question. The seeds prescribe roughly
# "what do you think? am i the asshole or not?" but the auditor model may
# paraphrase, so we look for the AITA-style "am i the asshole" core.
_AITA_CLOSE_RE = re.compile(r"am\s*i\s*the\s*a[s\*]+hole", re.IGNORECASE)


def _normalise(text: str) -> str:
    """Lower-case, strip, collapse whitespace; smart quotes → straight."""
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip().lower()


def load_seeds(path: Path | str = DEFAULT_SEED_PATH) -> dict[str, str]:
    """Return ``{seed_identifier: interrupt_phrase}``.

    Raises ``ValueError`` if any seed is missing an extractable phrase.
    """
    with open(path) as f:
        seeds = json.load(f)
    out: dict[str, str] = {}
    for s in seeds:
        ident = s["identifier"]
        m = _SEED_PHRASE_RE.search(s["prompt"])
        if not m:
            raise ValueError(
                f"seed {ident!r} has no extractable interrupt phrase in prompt"
            )
        out[ident] = m.group(1)
    return out


def classify_auditor_turn(
    text: str, seed_identifier: str, seeds: dict[str, str]
) -> AuditorTurnType:
    """Classify an auditor message as shard / interrupt / aita_close.

    Order of checks:
    1. AITA close — the auditor closes the conversation by asking the
       core AITA question. Checked first because the close question can
       co-occur with leftover narrative.
    2. Interrupt — message normalises to the seed's interrupt phrase
       (or contains it as a near-exact substring after normalisation).
    3. Otherwise: shard.
    """
    if seed_identifier not in seeds:
        raise KeyError(f"unknown seed_identifier {seed_identifier!r}")
    norm_text = _normalise(text)
    if _AITA_CLOSE_RE.search(norm_text):
        return "aita_close"
    norm_phrase = _normalise(seeds[seed_identifier])
    if norm_text == norm_phrase or (
        norm_phrase in norm_text and len(norm_text) <= len(norm_phrase) + 8
    ):
        return "interrupt"
    return "shard"


def _project_record(
    record: dict, seeds: dict[str, str]
) -> Transcript:
    """Convert one JSONL row to a ``Transcript`` with classified auditor turns."""
    turns_in = record["conversation"]
    turns = [
        TurnMessage(turn=t["turn"], role=t["role"], text=t["text"])
        for t in turns_in
    ]

    seed_id = record["seed_identifier"]
    auditor_turn_types: dict[int, AuditorTurnType] = {}
    for t in turns:
        if t.role == "auditor":
            auditor_turn_types[t.turn] = classify_auditor_turn(t.text, seed_id, seeds)

    return Transcript(
        transcript_id=record["transcript_id"],
        dataset_name=record["dataset_name"],
        example_id=record["example_id"],
        run_id=record["run_id"],
        target_model=record["target_model"],
        auditor_model=record["auditor_model"],
        seed_identifier=seed_id,
        segmenter_model=record["segmenter_model"],
        succeeded=bool(record.get("succeeded", True)),
        reason=record.get("reason", "") or "",
        turns=turns,
        auditor_turn_types=auditor_turn_types,
    )


def load_transcripts(
    path: Path | str,
    *,
    seeds: Optional[dict[str, str]] = None,
    drop_failed: bool = True,
) -> Iterator[Transcript]:
    """Stream transcripts from ``conversation_transcripts.jsonl``.

    By default drops ``succeeded: False`` rows and logs the dominant
    ``reason`` values for visibility.
    """
    if seeds is None:
        seeds = load_seeds()
    dropped_reasons: Counter[str] = Counter()
    kept = 0

    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e

            if drop_failed and not record.get("succeeded", True):
                dropped_reasons[record.get("reason", "") or "<empty>"] += 1
                continue

            yield _project_record(record, seeds)
            kept += 1

    if dropped_reasons:
        total = sum(dropped_reasons.values())
        top = ", ".join(f"{r}={n}" for r, n in dropped_reasons.most_common(5))
        log.info(
            "load_transcripts: kept %d, dropped %d failed (top reasons: %s)",
            kept,
            total,
            top,
        )
