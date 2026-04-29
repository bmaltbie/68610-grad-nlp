from __future__ import annotations

from typing import List, Sequence, Tuple
import math
import re

from .deterministic import estimate_tokens
from .schema import AtomicUnit, Shard, SUPPORTED_TARGET_TURNS


DEFAULT_SHARD_POLICY = "natural_dp_v1"

_DISCOURSE_START_RE = re.compile(
    r"(?i)^\s*(?:first|second|third|then|after|afterward|afterwards|later|however|but|eventually|finally|lastly|"
    r"in the end|tl\s*;?\s*dr|tldr|edit|update)\b"
)


class ShardPlanningError(Exception):
    """Raised when deterministic grouping cannot produce the requested target count."""

    pass


def plan_shards(
    raw_source_text: str,
    atomic_units: Sequence[AtomicUnit],
    target_turns: int,
    policy: str = DEFAULT_SHARD_POLICY,
) -> List[Shard]:
    """Merge source-order atomic units into exactly ``target_turns`` natural shards.

    The planner chooses contiguous cuts with dynamic programming. It never
    rewrites text or changes atomic-unit order; it only decides where shard
    boundaries go.
    """
    if policy != DEFAULT_SHARD_POLICY:
        raise ShardPlanningError("unsupported shard policy: %s" % policy)
    if target_turns not in SUPPORTED_TARGET_TURNS:
        raise ShardPlanningError("target_turns must be one of %s" % sorted(SUPPORTED_TARGET_TURNS))
    units = list(atomic_units)
    if len(units) < target_turns:
        raise ShardPlanningError(
            "need at least %d atomic units to produce %d shards; got %d"
            % (target_turns, target_turns, len(units))
        )
    boundaries = _choose_boundaries(raw_source_text, units, target_turns)
    groups = _groups_from_boundaries(units, boundaries)
    return [_shard_from_group(raw_source_text, index, group, target_turns) for index, group in enumerate(groups, start=1)]


def _choose_boundaries(raw: str, units: Sequence[AtomicUnit], target_turns: int) -> Tuple[int, ...]:
    """Return group start indexes after index 0, represented as unit-list offsets."""
    n = len(units)
    prefix = [0]
    for unit in units:
        prefix.append(prefix[-1] + _unit_tokens(raw, unit))
    total = max(1, prefix[-1])
    target = total / float(target_turns)

    dp: List[List[Tuple[float, float, Tuple[int, ...]]]] = [
        [(math.inf, math.inf, tuple()) for _ in range(n + 1)] for _ in range(target_turns + 1)
    ]
    dp[0][0] = (0.0, 0.0, tuple())
    for group_count in range(1, target_turns + 1):
        min_end = group_count
        max_end = n - (target_turns - group_count)
        for end in range(min_end, max_end + 1):
            best = (math.inf, math.inf, tuple())
            min_start = group_count - 1
            max_start = end - 1
            for start in range(min_start, max_start + 1):
                previous_cost, previous_natural, previous_boundaries = dp[group_count - 1][start]
                if math.isinf(previous_cost):
                    continue
                natural_score = _boundary_score(raw, units, start) if start > 0 else 0.0
                group_cost = _group_cost(prefix[end] - prefix[start], target, end - start, target_turns)
                candidate_boundaries = previous_boundaries + ((start,) if start > 0 else tuple())
                candidate = (
                    previous_cost + group_cost - natural_score,
                    previous_natural - natural_score,
                    candidate_boundaries,
                )
                if candidate < best:
                    best = candidate
            dp[group_count][end] = best
    boundaries = dp[target_turns][n][2]
    if len(boundaries) != target_turns - 1:
        raise ShardPlanningError("failed to choose %d shard boundaries" % (target_turns - 1))
    return boundaries


def _unit_tokens(raw: str, unit: AtomicUnit) -> int:
    text = raw[unit.start_char : unit.end_char] if 0 <= unit.start_char < unit.end_char <= len(raw) else unit.text
    return max(1, estimate_tokens(text))


def _group_cost(token_count: int, target_tokens: float, unit_count: int, target_turns: int) -> float:
    balance = ((token_count - target_tokens) / max(1.0, target_tokens)) ** 2 * 100.0
    tiny_penalty = 0.0
    if token_count < target_tokens * 0.45 and unit_count == 1:
        tiny_penalty = 10.0 + target_turns
    elif token_count < target_tokens * 0.30:
        tiny_penalty = 8.0
    return balance + tiny_penalty


def _boundary_score(raw: str, units: Sequence[AtomicUnit], start: int) -> float:
    previous = units[start - 1]
    current = units[start]
    gap = raw[previous.end_char : current.start_char]
    score = 0.0
    if "\n\n" in gap or re.search(r"\n\s*\n", gap):
        score += 28.0
    elif "\n" in gap:
        score += 10.0
    if previous.section_type != current.section_type:
        score += 36.0
    if current.section_type in {"tldr", "edit", "update"}:
        score += 16.0
    if _DISCOURSE_START_RE.match(current.text):
        score += 12.0
    if previous.text.rstrip().endswith((".", "!", "?")):
        score += 2.0
    return score


def _groups_from_boundaries(units: Sequence[AtomicUnit], boundaries: Sequence[int]) -> List[List[AtomicUnit]]:
    groups: List[List[AtomicUnit]] = []
    start = 0
    for boundary in boundaries:
        groups.append(list(units[start:boundary]))
        start = boundary
    groups.append(list(units[start:]))
    return groups


def _shard_from_group(raw: str, index: int, group: Sequence[AtomicUnit], target_turns: int) -> Shard:
    first = group[0]
    last = group[-1]
    return Shard(
        shard_id=index,
        unit_ids=[unit.unit_id for unit in group],
        text=raw[first.start_char : last.end_char].strip(),
        section_role=_role_for_group(index, group, target_turns),
    )


def _role_for_group(index: int, group: Sequence[AtomicUnit], target_turns: int) -> str:
    section_types = {unit.section_type for unit in group}
    text = " ".join(unit.text for unit in group).lower()
    if index == 1:
        return "setup"
    if "tldr" in section_types:
        return "tldr_summary"
    if section_types.intersection({"edit", "update"}):
        return "clarification"
    if index == target_turns and ("aita" in text or "wibta" in text or "asshole" in text or text.rstrip().endswith("?")):
        return "final_question"
    if index == 2:
        return "main_event"
    if index <= max(3, target_turns // 2):
        return "background_context"
    if "because" in text or "wanted" in text or "decided" in text:
        return "motivation"
    return "current_conflict"
