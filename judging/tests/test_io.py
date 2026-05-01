"""Tests for judging.io.

These run against the real ``conversation/seed.json`` and the real
``datasets/conversation_transcripts.jsonl`` — no mock fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from judging.io import (
    DEFAULT_SEED_PATH,
    classify_auditor_turn,
    load_seeds,
    load_transcripts,
)
from judging.schemas import Transcript

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRANSCRIPTS = REPO_ROOT / "datasets" / "conversation_transcripts.jsonl"


# ---------------------------------------------------------------------------
# Seed loading
# ---------------------------------------------------------------------------


def test_load_seeds_extracts_phrase_for_every_seed():
    seeds = load_seeds(DEFAULT_SEED_PATH)
    assert len(seeds) >= 20
    # Every seed_identifier should map to a non-empty phrase.
    for ident, phrase in seeds.items():
        assert ident.startswith("seed")
        assert phrase, f"empty phrase for {ident}"


def test_load_seeds_known_phrase():
    seeds = load_seeds(DEFAULT_SEED_PATH)
    assert seeds["seed1-emotion1"] == "wait im not finished yet"


def test_load_seeds_missing_phrase_raises(tmp_path):
    bad = tmp_path / "seed.json"
    bad.write_text(json.dumps([{"identifier": "seed-x", "prompt": "no quote here"}]))
    with pytest.raises(ValueError, match="no extractable interrupt phrase"):
        load_seeds(bad)


# ---------------------------------------------------------------------------
# Auditor turn classification
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def seeds() -> dict[str, str]:
    return load_seeds(DEFAULT_SEED_PATH)


def test_classify_interrupt_exact_match(seeds):
    assert (
        classify_auditor_turn("wait im not finished yet", "seed1-emotion1", seeds)
        == "interrupt"
    )


def test_classify_interrupt_with_smart_quotes(seeds):
    # seed3 uses a curly apostrophe in "i'm trying to word this right"
    assert (
        classify_auditor_turn("gimme a sec i'm trying to word this right", "seed3-emotion1", seeds)
        == "interrupt"
    )


def test_classify_aita_close(seeds):
    txt = "what do you think? am i the asshole or not?"
    assert classify_auditor_turn(txt, "seed1-emotion1", seeds) == "aita_close"


def test_classify_aita_close_takes_precedence_over_interrupt(seeds):
    # If the close question is asked, classify as aita_close even if the
    # interrupt phrase is buried in the same message.
    txt = "wait im not finished yet, but anyway, am I the asshole?"
    assert classify_auditor_turn(txt, "seed1-emotion1", seeds) == "aita_close"


def test_classify_shard_default(seeds):
    txt = "However, I told him that I had reconnected with another friend..."
    assert classify_auditor_turn(txt, "seed1-emotion1", seeds) == "shard"


def test_classify_unknown_seed_raises(seeds):
    with pytest.raises(KeyError):
        classify_auditor_turn("hi", "seed-not-real", seeds)


# ---------------------------------------------------------------------------
# Transcript loading against the real JSONL
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TRANSCRIPTS.exists(), reason="conversation_transcripts.jsonl missing")
def test_load_transcripts_yields_valid_transcripts():
    records = list(load_transcripts(TRANSCRIPTS))
    assert records, "expected at least one transcript"
    for r in records:
        assert isinstance(r, Transcript)
        assert r.transcript_id
        assert r.dataset_name in {"AITA-YTA", "AITA-NTA-OG", "AITA-NTA-FLIP"}
        # Every auditor turn has a classification.
        for t in r.auditor_turns():
            assert t.turn in r.auditor_turn_types
            assert r.auditor_turn_types[t.turn] in {"shard", "interrupt", "aita_close"}


@pytest.mark.skipif(not TRANSCRIPTS.exists(), reason="conversation_transcripts.jsonl missing")
def test_load_transcripts_classifies_known_interrupt():
    records = list(load_transcripts(TRANSCRIPTS))
    # The pilot transcript starts with seed1-emotion1; turn 3 is the
    # exact interrupt phrase. Find any record using that seed and verify.
    seed1 = [r for r in records if r.seed_identifier == "seed1-emotion1"]
    if not seed1:
        pytest.skip("no seed1-emotion1 transcripts in current dataset")
    r = seed1[0]
    interrupt_turns = [
        t for t in r.auditor_turns() if r.auditor_turn_types[t.turn] == "interrupt"
    ]
    assert interrupt_turns, "expected at least one interrupt turn for seed1-emotion1"


@pytest.mark.skipif(not TRANSCRIPTS.exists(), reason="conversation_transcripts.jsonl missing")
def test_load_transcripts_drops_failed(tmp_path):
    # Read real records, append a synthetic failed one, verify it's dropped.
    real = TRANSCRIPTS.read_text().strip().splitlines()
    failed = json.loads(real[0])
    failed["transcript_id"] = "synthetic-failed"
    failed["succeeded"] = False
    failed["reason"] = "max_turns_exceeded"

    fp = tmp_path / "mixed.jsonl"
    fp.write_text("\n".join(real + [json.dumps(failed)]) + "\n")

    out = list(load_transcripts(fp))
    ids = {r.transcript_id for r in out}
    assert "synthetic-failed" not in ids
    assert len(out) == len(real)


@pytest.mark.skipif(not TRANSCRIPTS.exists(), reason="conversation_transcripts.jsonl missing")
def test_load_transcripts_keep_failed_when_disabled(tmp_path):
    real = TRANSCRIPTS.read_text().strip().splitlines()
    failed = json.loads(real[0])
    failed["transcript_id"] = "synthetic-failed-2"
    failed["succeeded"] = False

    fp = tmp_path / "mixed.jsonl"
    fp.write_text("\n".join(real + [json.dumps(failed)]) + "\n")

    out = list(load_transcripts(fp, drop_failed=False))
    ids = {r.transcript_id for r in out}
    assert "synthetic-failed-2" in ids


def test_load_transcripts_invalid_json_raises(tmp_path):
    fp = tmp_path / "bad.jsonl"
    fp.write_text("not json\n")
    with pytest.raises(ValueError, match="invalid JSON"):
        list(load_transcripts(fp))
