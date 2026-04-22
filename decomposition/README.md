# Decomposition Module

This directory contains decomposition-specific notes and research material.

For the project-level architecture, sharding contract, and cross-module handoff rules, start with the framework design:

- [Multi-Turn ELEPHANT Sharding Framework and Handoff Contract](../docs/multi-turn-elephant-framework-design.md)

## Module Ownership

The decomposition module owns the first stage of the benchmark pipeline:

```text
single AITA post
    |
    v
decomposition module
    |
    v
shards.jsonl
```

The module is responsible for:

- ingesting raw AITA post text;
- extracting traceable atomic units;
- validating raw-text source spans;
- merging units into neutral chronological shards;
- writing `shards.jsonl`.

The module does not own:

- conversational rephrasing;
- user simulator prompting;
- target model prompting;
- judge prompting or scoring.

## Local References

- [Deep research report](./deep-research-report.md)
