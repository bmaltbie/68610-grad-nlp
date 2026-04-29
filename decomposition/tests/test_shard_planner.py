import pytest

from decomposition.schema import AtomicUnit
from decomposition.shard_planner import ShardPlanningError, plan_shards


def units_for(raw: str):
    units = []
    cursor = 0
    for index, token in enumerate(raw.replace("\n\n", " ").split(" "), start=1):
        if not token:
            continue
        start = raw.find(token, cursor)
        end = start + len(token)
        units.append(AtomicUnit(len(units) + 1, raw[start:end], start, end, "body"))
        cursor = end
    return units


def test_plan_shards_produces_requested_supported_counts() -> None:
    raw = "AITA? One. Two. Three. Four. Five. Six. Seven."
    units = units_for(raw)

    for target in (4, 6, 8):
        shards = plan_shards(raw, units, target)

        assert len(shards) == target
        assert [unit_id for shard in shards for unit_id in shard.unit_ids] == list(range(1, len(units) + 1))


def test_plan_shards_rejects_too_few_units() -> None:
    raw = "AITA? One. Two."

    with pytest.raises(ShardPlanningError, match="need at least 4"):
        plan_shards(raw, units_for(raw), 4)


def test_plan_shards_prefers_paragraph_boundary_when_close_to_balanced() -> None:
    raw = "U1. U2. U3.\n\nU4. U5. U6. U7. U8. U9."
    units = units_for(raw)

    shards = plan_shards(raw, units, 4)

    assert shards[0].unit_ids == [1, 2, 3]
