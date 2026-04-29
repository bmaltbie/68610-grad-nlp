# conversation/

Self-contained experiment driver. Walks `seed.json` × `shards.json`,
runs one Petri-style audit per (seed, shard) pair, and writes Inspect
AI `.eval` transcripts to `outputs/<seed_identifier>/` (relative to the
current working directory).

No scoring is run — this directory only produces conversation
transcripts.

## Files

- `conversation.py` — driver script. Model IDs and `MAX_TURNS` are
  module-level constants; edit inline to change.
- `seed.json` — seed prompts (schema below).
- `shards.json` — list of `{ "shards": ["context bullet", ...] }`
  objects prepended to each seed.

## Run

From the repo root, with the project installed (`pip install -e .` in
a Python 3.12+ venv) and `.env` configured with provider API keys
(`ANTHROPIC_API_KEY`, etc.):

```
python conversation/conversation.py
```

Open a transcript:

```
inspect view start --log-dir outputs/<seed_identifier>/
```

## seed.json schema

A list of objects. Each entry must have non-empty string fields
`identifier` (unique across the file), `description`, `prompt`, and
optional integer `max_turns`. Bare-string seeds are rejected.
`target_system` fields, if present, are silently ignored — the target
receives no preset system prompt.

## shards.json schema

A list of `{ "shards": list[str] }` objects. Each shard list is
prepended to the seed prompt as a bulleted `## Context:` block before
the seed's `## Instructions:`.
