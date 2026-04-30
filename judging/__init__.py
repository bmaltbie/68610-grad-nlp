"""Judging stage of the multi-turn ELEPHANT pipeline.

Reads ``conversation_transcripts.jsonl`` from the conversation stage and
emits per-turn ELEPHANT sycophancy scores plus paired moral-sycophancy
verdicts.
"""

__version__ = "0.1.0"
