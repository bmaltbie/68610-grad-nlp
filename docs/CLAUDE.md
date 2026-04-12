# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Graduate NLP course (MIT 6.8610) research repository containing the **ELEPHANT dataset** — a collection of datasets for studying language understanding, perspective-taking, and subjective reasoning in NLP. Currently data-only; no build system, tests, or application code exist yet.

## Repository Structure

- `datasets/` — CSV datasets from the ELEPHANT project (sourced from [OSF](https://osf.io/r3dmj/?view_only=37ee66a8020a45c29a38bd704ca61067))
  - `OEQ.csv` — 3,027 advice-seeking queries with human responses
  - `AITA-YTA.csv` — 2,000 "You're The Asshole" Reddit posts + top comments
  - `AITA-NTA-FLIP.csv` — 1,591 "Not The Asshole" posts with flipped-perspective versions
  - `AITA-NTA-OG.csv` — Original NTA posts
  - `SS.csv` — 3,777 assumption-laden subjective statements
  - `Manifest.md` — Dataset documentation

## Dataset Schemas

- **OEQ**: `prompt, human, source, validation_human, indirectness_human, framing_human`
- **AITA-YTA**: `prompt, top_comment, is_asshole, ytanta, validation_human, indirectness_human, framing_human`
- **AITA-NTA-FLIP**: `id, original_post, flipped_story`
- **AITA-NTA-OG**: `id, original_post`
- **SS**: `sentence, self_attitude, other_attitude`

## Git Branches

- `main` — stable branch
- `evaluation` — evaluation/experimentation work
