#!/usr/bin/env python
"""Export Inspect AI ``.eval`` transcripts to a flat JSONL.

Walks a directory of ``.eval`` logs (default ``outputs/``), extracts the
auditor↔target conversation from each sample, and appends one JSON
record per sample to ``conversation_transcripts.jsonl`` (default).

Each record has::

    {
      "transcript_id":    <UUID4 unique to this record>,
      "conversation": [
        {"turn": 1, "role": "auditor", "text": "<first message sent to target>"},
        {"turn": 2, "role": "target",  "text": "<first target reply>"},
        ...
      ],
      "auditor_model":    <e.g. "openai/gpt-5.4-nano", may be null>,
      "target_model":     <e.g. "openai/gpt-5.4-nano", may be null>,
      "succeeded":        <bool: True iff log status was "success",
                           auditor called end_conversation, AND the
                           target's final reply contained an AITA
                           verdict>,
      "reason":           <"" on success, else a high-level category:
                           "provider_error" | "hit_max_turns" |
                           "missing_verdict">,
      "dataset_name":     <from sample metadata, may be null>,
      "example_id":       <from sample metadata, may be null>,
      "run_id":           <from sample metadata, may be null>,
      "segmenter_model":  <from sample metadata, may be null>,
      "seed_identifier":  <from sample metadata, may be null>
    }

The conversation array is reconstructed from the auditor agent's message
log: each ``send_message`` tool call from the auditor produces one
``{"role": "auditor", ...}`` entry, and each tool response wrapped in
``<target_response>...</target_response>`` produces one
``{"role": "target", ...}`` entry. ``turn`` is 1-indexed in message
order. Other auditor tool calls (``set_system_message``,
``end_conversation``, ``resume``, etc.) are not part of the transcript.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable

DEFAULT_OUTPUTS = Path("outputs")
DEFAULT_OUT = Path("conversation_transcripts.jsonl")

_TARGET_BLOCK_RE = re.compile(
    r"<target_response>\s*(?:\[message_id:[^\]]*\]\s*)?(.*?)\s*</target_response>",
    re.DOTALL,
)

# AITA verdict markers — Reddit acronyms (YTA/NTA/ESH/NAH/YWBTA/YWNBTA/INFO)
# plus the common phrasal forms a model might use ("you're the asshole",
# "not the asshole", "everyone sucks here", "no assholes here"). Matched
# case-insensitively with word boundaries to keep false positives low.
_VERDICT_RE = re.compile(
    r"\b(?:YTA|NTA|ESH|NAH|YWBTA|YWNBTA|INFO)\b"
    r"|\byou'?re\s+(?:not\s+)?the\s+asshole\b"
    r"|\byou\s+are\s+(?:not\s+)?the\s+asshole\b"
    r"|\bnot\s+the\s+asshole\b"
    r"|\beveryone\s+sucks(?:\s+here)?\b"
    r"|\bno\s+assholes?(?:\s+here)?\b",
    re.IGNORECASE,
)


def _has_verdict(text: str) -> bool:
    return bool(text and _VERDICT_RE.search(text))


def _extract_target_text(tool_text: str) -> str | None:
    """Return the inner text of a <target_response> block, or None.

    The auditor's `send_message` tool returns target replies wrapped as
    ``<target_response>\\n[message_id: Mxx]\\n<actual text>\\n</target_response>``.
    We strip the wrapper and message_id header.
    """
    if not tool_text or "<target_response>" not in tool_text:
        return None
    m = _TARGET_BLOCK_RE.search(tool_text)
    if not m:
        return None
    return m.group(1).strip()


def evaluate_outcome(
    log_status: str | None, messages: Iterable[Any]
) -> tuple[bool, str]:
    """Return (succeeded, reason) for a sample.

    High-level reason categories (precedence order, first match wins):
      ``"provider_error"`` — Inspect AI log status was not ``"success"``
        (provider/API/runtime failure)
      ``"hit_max_turns"`` — log succeeded but the auditor never called
        ``end_conversation`` (the run was cut off by ``max_turns``)
      ``"missing_verdict"`` — the target's final reply contained no
        recognizable AITA verdict (YTA/NTA/ESH/NAH/INFO or phrasal
        equivalent). The dataset is Reddit AITA, so the target is
        expected to render a judgment in its closing message.
      ``""`` — succeeded
    """
    if log_status != "success":
        return False, "provider_error"
    ended_cleanly = False
    last_target_text: str | None = None
    for m in messages or []:
        role = getattr(m, "role", None)
        if role == "assistant":
            for tc in getattr(m, "tool_calls", None) or []:
                if getattr(tc, "function", None) == "end_conversation":
                    ended_cleanly = True
        elif role == "tool":
            target = _extract_target_text(getattr(m, "text", "") or "")
            if target:
                last_target_text = target
    if not ended_cleanly:
        return False, "hit_max_turns"
    if not _has_verdict(last_target_text or ""):
        return False, "missing_verdict"
    return True, ""


def build_conversation(messages: Iterable[Any]) -> list[dict[str, Any]]:
    convo: list[dict[str, Any]] = []
    for msg in messages or []:
        role = getattr(msg, "role", None)
        if role == "assistant":
            for tc in getattr(msg, "tool_calls", None) or []:
                if getattr(tc, "function", None) != "send_message":
                    continue
                args = getattr(tc, "arguments", None) or {}
                text = args.get("message")
                if isinstance(text, str) and text.strip():
                    convo.append({
                        "turn": len(convo) + 1,
                        "role": "auditor",
                        "text": text.strip(),
                    })
        elif role == "tool":
            text = getattr(msg, "text", "") or ""
            target = _extract_target_text(text)
            if target:
                convo.append({
                    "turn": len(convo) + 1,
                    "role": "target",
                    "text": target,
                })
    return convo


def build_record(
    sample: Any,
    *,
    auditor_model: str | None = None,
    target_model: str | None = None,
    log_status: str | None = None,
) -> dict[str, Any]:
    meta = sample.metadata or {}
    messages = sample.messages or []
    succeeded, reason = evaluate_outcome(log_status, messages)
    return {
        "transcript_id": str(uuid.uuid4()),
        "conversation": build_conversation(messages),
        "auditor_model": auditor_model,
        "target_model": target_model,
        "succeeded": succeeded,
        "reason": reason,
        "dataset_name": meta.get("dataset_name"),
        "example_id": meta.get("example_id"),
        "run_id": meta.get("run_id"),
        "segmenter_model": meta.get("segmenter_model"),
        "seed_identifier": meta.get("seed_identifier"),
    }


def export(eval_paths: list[Path], out_path: Path, *, truncate: bool) -> int:
    from inspect_ai.log import read_eval_log

    if not eval_paths:
        print("no .eval files found", file=sys.stderr)
        return 1

    mode = "w" if truncate else "a"
    n_samples = 0
    with out_path.open(mode) as out:
        for ef in eval_paths:
            log = read_eval_log(str(ef))
            samples = log.samples or []
            if not samples:
                print(f"  {ef}: no samples", file=sys.stderr)
                continue
            roles = getattr(log.eval, "model_roles", None) or {}
            auditor_model = getattr(roles.get("auditor"), "model", None)
            target_model = getattr(roles.get("target"), "model", None)
            log_status = getattr(log, "status", None)
            for sample in samples:
                record = build_record(
                    sample,
                    auditor_model=auditor_model,
                    target_model=target_model,
                    log_status=log_status,
                )
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_samples += 1
            print(
                f"  {ef.name}: appended {len(samples)} sample(s)",
                file=sys.stderr,
            )

    print(
        f"wrote {n_samples} record(s) to {out_path} "
        f"({'truncated' if truncate else 'appended'})",
        file=sys.stderr,
    )
    return 0


def _collect_eval_files(inputs: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in inputs:
        if p.is_file() and p.suffix == ".eval":
            out.append(p)
        elif p.is_dir():
            out.extend(sorted(p.rglob("*.eval")))
        else:
            print(f"skipping (not a .eval file or directory): {p}", file=sys.stderr)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Export Petri auditor↔target transcripts from .eval logs "
            "to a flat JSONL file."
        )
    )
    p.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help=(
            "One or more .eval files or directories to walk for .eval files "
            f"(default: {DEFAULT_OUTPUTS}/)."
        ),
    )
    p.add_argument(
        "-o", "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output JSONL path (default: {DEFAULT_OUT}).",
    )
    p.add_argument(
        "--truncate",
        action="store_true",
        help="Overwrite the output file instead of appending.",
    )
    args = p.parse_args(argv)

    inputs = args.inputs or [DEFAULT_OUTPUTS]
    eval_files = _collect_eval_files(inputs)
    return export(eval_files, args.out, truncate=args.truncate)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
