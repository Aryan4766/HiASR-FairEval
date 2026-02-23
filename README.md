# HiASR-FairEval — Hindi ASR Research & Fair Evaluation

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
├── requirements.txt
├── .gitignore
├── test_all.py                 # End-to-end pipeline test
├── configs/                    # YAML configuration files
│   ├── preprocessing.yaml
│   ├── training.yaml
│   └── lattice.yaml
├── src/
│   ├── data_pipeline/          # Data download & preprocessing
│   │   ├── fix_urls.py
│   │   ├── download_subset.py
│   │   ├── audio_preprocessing.py
│   │   └── text_normalization.py
│   ├── q1_whisper/             # Whisper fine-tuning & evaluation
│   │   ├── baseline_eval_colab.ipynb
│   │   ├── finetune_subset_colab.ipynb
│   │   ├── compute_wer.py
│   │   └── error_analysis.py
│   ├── q2_disfluency/          # Disfluency detection
│   │   ├── detect_rules.py
│   │   ├── segment_audio.py
│   │   └── stats_analysis.py
│   ├── q3_spelling/            # Spelling classification
│   │   ├── extract_unique_words.py
│   │   ├── spell_classifier.py
│   │   └── error_pattern_analysis.py
│   ├── q4_lattice/             # Lattice-based WER
│   │   ├── align_dp.py
│   │   ├── majority_consensus.py
│   │   └── lattice_wer.py
│   └── utils/                  # Shared utilities
│       ├── logger.py
│       ├── config_loader.py
│       └── metrics.py
└── outputs/                    # Generated outputs
    ├── q1/
    ├── q2/
    ├── q3/
    └── q4/
```

## Setup

### Prerequisites
- Python 3.8+
- 8GB RAM (local)
- Google Colab with T4 GPU (for Whisper training)

### Installation

```bash
pip install -r requirements.txt
```

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

1. Open `src/q1_whisper/baseline_eval_colab.ipynb` in Google Colab
2. Run all cells for baseline WER on FLEURS Hindi
3. Open `src/q1_whisper/finetune_subset_colab.ipynb` in Colab
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

- No hardcoded paths — config-driven
- Type hints and docstrings throughout
- Memory-efficient (streaming, per-file processing)
- GPU-optional design (CPU fallback everywhere)
- Structured logging with hardware info
- Comprehensive test suite (`test_all.py`)
- Reproducible (seed-controlled randomization)

## License

This project is part of the JoshTalks AI Researcher Intern assessment.
