# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-02-25

### Added

- **Q1 — Whisper Fine-Tuning**: Baseline evaluation and fine-tuning pipeline on FLEURS Hindi
- **Q2 — Disfluency Detection**: Rule-based detection of fillers, repetitions, elongations, and hesitations
- **Q3 — Spelling Classification**: Classify ~1,77,000 unique Hindi words as correct/incorrect
- **Q4 — Lattice WER**: Fair WER computation using majority consensus from 5 ASR models
- Data pipeline: URL fixing, audio download, preprocessing, and text normalization
- Shared utilities: config loader, structured logger, WER/CER metrics
- YAML config files for preprocessing, training, and lattice evaluation
- Integration test suite (`test_all.py`)
- Unit tests for all modules (`tests/`)
- GitHub Actions CI/CD pipeline
- Project metadata: LICENSE (MIT), CONTRIBUTING.md, CODE_OF_CONDUCT.md
