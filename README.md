# HiASR-FairEval — Hindi ASR Research & Fair Evaluation

![Tests](https://github.com/Aryan4766/HiASR-FairEval/actions/workflows/test.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A research-grade ASR pipeline implementing Whisper fine-tuning, disfluency detection,
spelling error classification, and lattice-based fair WER evaluation for Hindi speech data.

## Project Overview

| Component | Description |
|-----------|-------------|
| **Q1** — Whisper Fine-Tuning | Fine-tune `whisper-small` on Josh Talks Hindi data, evaluate on FLEURS |
| **Q2** — Disfluency Detection | Rule-based detection of fillers, repetitions, elongations from audio |
| **Q3** — Spelling Classification | Classify ~1,77,000 unique Hindi words as correct/incorrect |
| **Q4** — Lattice WER | Fair WER using majority consensus from 5 ASR models |

## Project Structure

```
HiASR-FairEval/
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── CHANGELOG.md
├── requirements.txt              # Pinned production dependencies
├── requirements-dev.txt           # Dev/test dependencies
├── .gitignore
├── test_all.py                    # End-to-end pipeline test
├── .github/
│   └── workflows/
│       └── test.yml               # CI/CD pipeline
├── configs/                       # YAML configuration files
│   ├── preprocessing.yaml
│   ├── training.yaml
│   └── lattice.yaml
├── src/
│   ├── data_pipeline/             # Data download & preprocessing
│   │   ├── fix_urls.py
│   │   ├── download_subset.py
│   │   ├── audio_preprocessing.py
│   │   └── text_normalization.py
│   ├── q1_whisper/                # Whisper fine-tuning & evaluation
│   │   ├── baseline_eval_colab.ipynb
│   │   ├── finetune_subset_colab.ipynb
│   │   ├── compute_wer.py
│   │   └── error_analysis.py
│   ├── q2_disfluency/             # Disfluency detection
│   │   ├── detect_rules.py
│   │   ├── segment_audio.py
│   │   └── stats_analysis.py
│   ├── q3_spelling/               # Spelling classification
│   │   ├── extract_unique_words.py
│   │   ├── spell_classifier.py
│   │   └── error_pattern_analysis.py
│   ├── q4_lattice/                # Lattice-based WER
│   │   ├── align_dp.py
│   │   ├── majority_consensus.py
│   │   └── lattice_wer.py
│   └── utils/                     # Shared utilities
│       ├── logger.py
│       ├── config_loader.py
│       └── metrics.py
├── notebooks/                     # Jupyter notebooks (exploration)
│   ├── README.md
│   ├── 01_whisper_baseline_eval.ipynb
│   └── 02_whisper_finetune.ipynb
├── tests/                         # Unit tests
│   ├── conftest.py
│   ├── test_q1_whisper.py
│   ├── test_q2_disfluency.py
│   ├── test_q3_spelling.py
│   ├── test_q4_lattice.py
│   ├── test_data_pipeline.py
│   └── test_utils.py
└── outputs/                       # Generated outputs
    ├── q1/
    ├── q2/
    ├── q3/
    └── q4/
```

## Setup

### Prerequisites
- Python 3.9+ (tested on 3.9, 3.10, 3.11)
- 8GB RAM (local)
- Google Colab with T4 GPU (for Whisper training)

### Installation

```bash
# Clone the repository
git clone https://github.com/Aryan4766/HiASR-FairEval.git
cd HiASR-FairEval

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# For development (includes test/lint tools)
pip install -r requirements-dev.txt
```

### Data Setup

The dataset files are **not** tracked in git (they are in `.gitignore`). Place the following files in the project root:

| File | Size | Description |
|------|------|-------------|
| `FT Data.xlsx` | ~15 KB | Fine-tuning corpus index |
| `FT Result.xlsx` | ~52 KB | Fine-tuning result template |
| `Unique Words Data.xlsx` | ~2.4 MB | Hindi word list (~1,77,000 words) |
| `Speech Disfluencies List.xlsx` | ~22 KB | Disfluency examples by type |
| `Speech Disfluencies Result.xlsx` | ~51 KB | Disfluency results |
| `Question 4.xlsx` | ~14 KB | Multi-model ASR transcriptions |

### Hardware Design

| Task | Environment | Reason |
|------|------------|--------|
| Data preprocessing | **Local** | CPU-only, streaming |
| Whisper fine-tuning | **Colab GPU** | Requires T4 |
| Baseline evaluation | **Colab GPU** | Faster inference |
| Q2-Q4 analysis | **Local** | CPU-sufficient |

## Usage

### 1. Data Pipeline

```bash
# Place dataset index in data/raw/dataset_index.csv
# Fix URLs, download subset, preprocess
python -m src.data_pipeline.download_subset
python -m src.data_pipeline.audio_preprocessing
```

### 2. Q1 — Whisper Fine-Tuning

1. Open `notebooks/01_whisper_baseline_eval.ipynb` in Google Colab
2. Run all cells for baseline WER on FLEURS Hindi
3. Open `notebooks/02_whisper_finetune.ipynb` in Colab
4. Upload processed subset and run fine-tuning
5. Run error analysis locally:

```bash
python -m src.q1_whisper.error_analysis --predictions outputs/q1/predictions.csv
```

### 3. Q2 — Disfluency Detection

```bash
# Detect disfluencies from transcriptions
python -m src.q2_disfluency.detect_rules

# Segment audio clips
python -m src.q2_disfluency.segment_audio --report outputs/q2/disfluency_report.csv --audio-dir data/processed

# Generate stats and plots
python -m src.q2_disfluency.stats_analysis --report outputs/q2/disfluency_report.csv
```

### 4. Q3 — Spelling Classification

```bash
# Classify word list
python -m src.q3_spelling.spell_classifier --input data/raw/unique_words.csv --output outputs/q3/word_classification.csv

# Analyze error patterns
python -m src.q3_spelling.error_pattern_analysis --classification outputs/q3/word_classification.csv
```

### 5. Q4 — Lattice WER

```bash
# Run with provided model transcriptions
python -m src.q4_lattice.lattice_wer --data data/raw/model_transcriptions.json

# Or run demo with synthetic data
python -m src.q4_lattice.lattice_wer
```

### Running Tests

```bash
# Run all unit tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=src --cov-report=term-missing

# Run integration test (requires data files)
python test_all.py
```

## Key Outputs

| Output | Path |
|--------|------|
| Baseline WER metrics | `outputs/q1/baseline_metrics.json` |
| Fine-tuned WER metrics | `outputs/q1/finetuned_metrics.json` |
| Error analysis | `outputs/q1/error_analysis.csv` |
| Disfluency report | `outputs/q2/disfluency_report.csv` |
| Spelling classification | `outputs/q3/word_classification.csv` |
| Lattice WER comparison | `outputs/q4/wer_comparison.csv` |

## Configuration

All parameters are config-driven via YAML files in `configs/`:

- `preprocessing.yaml` — Audio/text normalization, dataset URLs, subset sizes
- `training.yaml` — Whisper hyperparameters, evaluation settings, ablation configs
- `lattice.yaml` — Alignment costs, consensus thresholds

## Engineering Practices

- No hardcoded paths — config-driven via YAML
- Type hints and docstrings throughout
- Memory-efficient (streaming, per-file processing)
- GPU-optional design (CPU fallback everywhere)
- Structured logging with hardware info
- Comprehensive test suite (`test_all.py` + `tests/`)
- CI/CD pipeline with lint, format, and test checks
- Reproducible (seed-controlled randomization, pinned dependencies)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

This project is part of the JoshTalks AI Researcher Intern assessment. Built using:

- [OpenAI Whisper](https://github.com/openai/whisper) for ASR
- [Google FLEURS](https://huggingface.co/datasets/google/fleurs) for evaluation
- [jiwer](https://github.com/jitsi/jiwer) for WER computation
