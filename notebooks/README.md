# Notebooks

These Jupyter notebooks are provided for **exploration and visualization** only.  
For production use, see the CLI-enabled Python modules in `src/`.

## Available Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_whisper_baseline_eval.ipynb` | Baseline Whisper evaluation on FLEURS Hindi (runs on Google Colab with T4 GPU) |
| `02_whisper_finetune.ipynb` | Fine-tune Whisper-small on Hindi data subset (Colab GPU required) |

## How to Run

### Google Colab (Recommended)
1. Upload the notebook to [Google Colab](https://colab.research.google.com/)
2. Set Runtime → GPU (T4)
3. Run all cells

### Local
```bash
pip install jupyter
jupyter notebook notebooks/01_whisper_baseline_eval.ipynb
```

> ⚠️ **Note:** Local execution requires a CUDA-capable GPU for reasonable performance.
