"""Decomposition utilities for chronology-preserving AITA sharding."""

from .schema import (
    AtomicUnit,
    Shard,
    ShardRecord,
    ValidationError,
    WarningItem,
    validate_record,
    validate_record_dict,
)

__all__ = [
    "AtomicUnit",
    "Shard",
    "ShardRecord",
    "ValidationError",
    "WarningItem",
    "validate_record",
    "validate_record_dict",
]
