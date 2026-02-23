"""
Lattice WER — JoshTalks ASR Research
=======================================
End-to-end lattice-based WER computation.

Pipeline:
1. Align all 5 model outputs to the human reference
2. Build majority consensus at each word position
3. Create adjusted reference (lattice-corrected)
4. Recompute WER for each model against adjusted reference
5. Validate: adjusted WER ≤ original WER where reference was corrected

This reduces WER for models that were unfairly penalized due to
reference errors, while keeping WER unchanged for others.
"""

import json
import os
import sys
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.q4_lattice.align_dp import align, align_strings
from src.q4_lattice.majority_consensus import (
    align_all_to_reference,
    build_position_matrix,
    compute_consensus,
    get_consensus_reference,
    summarize_consensus,
)

logger = setup_logger("lattice_wer", log_to_file=True)


def compute_wer(reference: List[str], hypothesis: List[str]) -> float:
    """Compute WER from word lists."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    _, distance, _ = align(reference, hypothesis)
    return distance / len(reference)


def lattice_wer_pipeline(
    reference_str: str,
    model_outputs_str: List[str],
    model_names: Optional[List[str]] = None,
    min_agreement: int = 3,
) -> Dict:
    """
    Full lattice-based WER pipeline.

    Args:
        reference_str: Human reference transcription string.
        model_outputs_str: List of model output strings (5 models).
        model_names: Optional names for each model.
        min_agreement: Minimum agreement for consensus.

    Returns:
        Dictionary with original WER, adjusted WER, and comparisons.
    """
    num_models = len(model_outputs_str)
    if model_names is None:
        model_names = [f"Model_{i+1}" for i in range(num_models)]

    # Parse into word lists
    reference = reference_str.strip().split()
    model_outputs = [out.strip().split() for out in model_outputs_str]

    logger.info(f"Reference: {reference_str[:80]}...")
    logger.info(f"Number of models: {num_models}")

    # Step 1: Compute original WER for each model
    original_wers = {}
    for name, output in zip(model_names, model_outputs):
        wer = compute_wer(reference, output)
        original_wers[name] = round(wer, 4)
        logger.info(f"  Original WER [{name}]: {wer:.4f}")

    # Step 2: Build lattice and compute consensus
    alignments = align_all_to_reference(reference, model_outputs)
    position_matrix = build_position_matrix(reference, alignments)
    position_matrix = compute_consensus(position_matrix, min_agreement, num_models)

    # Step 3: Get adjusted reference
    adjusted_reference = get_consensus_reference(position_matrix)
    consensus_summary = summarize_consensus(position_matrix)

    logger.info(f"\nConsensus Summary:")
    logger.info(f"  Positions: {consensus_summary['total_positions']}")
    logger.info(f"  Reference kept: {consensus_summary['reference_kept']}")
    logger.info(f"  Reference overridden: {consensus_summary['reference_overridden']}")

    # Step 4: Recompute WER with adjusted reference
    adjusted_wers = {}
    for name, output in zip(model_names, model_outputs):
        wer = compute_wer(adjusted_reference, output)
        adjusted_wers[name] = round(wer, 4)
        logger.info(f"  Adjusted WER [{name}]: {wer:.4f}")

    # Step 5: Compute deltas and validate
    results = []
    all_valid = True

    for name in model_names:
        orig = original_wers[name]
        adj = adjusted_wers[name]
        delta = round(adj - orig, 4)

        # Validate: adjusted should be ≤ original (or equal if no override affects this model)
        valid = adj <= orig + 0.0001  # Small epsilon for floating point

        if not valid:
            logger.warning(
                f"VALIDATION FAILED for {name}: adjusted WER ({adj}) > original WER ({orig})"
            )
            all_valid = False

        results.append({
            "model": name,
            "original_wer": orig,
            "adjusted_wer": adj,
            "delta": delta,
            "valid": valid,
        })

    return {
        "results": results,
        "consensus_summary": consensus_summary,
        "original_reference": " ".join(reference),
        "adjusted_reference": " ".join(adjusted_reference),
        "all_valid": all_valid,
    }


def format_comparison_table(results: List[Dict]) -> str:
    """Format results as a readable comparison table."""
    header = f"{'Model':<15} {'Original WER':>14} {'Adjusted WER':>14} {'Delta':>10} {'Valid':>8}"
    separator = "=" * 65
    lines = [separator, header, separator]

    for r in results:
        delta_str = f"{r['delta']:+.4f}"
        valid_str = "✓" if r["valid"] else "✗"
        lines.append(
            f"{r['model']:<15} {r['original_wer']:>14.4f} "
            f"{r['adjusted_wer']:>14.4f} {delta_str:>10} {valid_str:>8}"
        )

    lines.append(separator)
    return "\n".join(lines)


def plot_wer_comparison(
    results: List[Dict],
    save_path: str = "outputs/q4/lattice_wer_comparison.png",
) -> None:
    """Plot original vs adjusted WER for all models."""
    fig, ax = plt.subplots(figsize=(12, 6))

    models = [r["model"] for r in results]
    original = [r["original_wer"] for r in results]
    adjusted = [r["adjusted_wer"] for r in results]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, original, width, label="Original WER",
                   color="#E8636F", edgecolor="white")
    bars2 = ax.bar(x + width/2, adjusted, width, label="Adjusted WER (Lattice)",
                   color="#4A90D9", edgecolor="white")

    ax.set_ylabel("WER", fontsize=12)
    ax.set_title("Lattice-Based WER Comparison: Original vs Adjusted",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"WER comparison plot saved to {save_path}")


def run_lattice_evaluation(
    data_path: Optional[str] = None,
    output_dir: str = "outputs/q4",
) -> None:
    """
    Run lattice WER evaluation.

    If data_path is provided, loads data from file.
    Otherwise, uses synthetic demo data.

    Args:
        data_path: Path to data file with reference + model outputs.
        output_dir: Output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    if data_path and os.path.exists(data_path):
        # Load from file
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        reference = data["reference"]
        model_outputs = data["model_outputs"]
        model_names = data.get("model_names", None)
    else:
        # Synthetic demo data
        logger.info("Using synthetic demo data (no data file provided)")
        reference = "यह एक अच्छा दिन है और मौसम बहुत सुंदर है"
        model_outputs = [
            "यह एक अच्छा दिन है और मौसम बहुत सुंदर है",    # Model 1: perfect
            "ये एक अच्छा दिन है और मौसम बहुत सुंदर है",     # Model 2: ये vs यह
            "यह अच्छा दिन है और मौसम बहुत सुंदर है",         # Model 3: deletion
            "यह एक अच्छा दिन है और मौसम बहुत सुन्दर है",    # Model 4: spelling variant
            "यह एक अच्छा दिन है और मौसम बहुत सुंदर है",     # Model 5: perfect
        ]
        model_names = ["Whisper-S", "Wav2Vec", "Conformer", "Kaldi", "NeMo"]

    # Run pipeline
    output = lattice_wer_pipeline(reference, model_outputs, model_names)

    # Print comparison table
    table = format_comparison_table(output["results"])
    print(f"\n{table}")

    # Print consensus info
    print(f"\nOriginal Reference:  {output['original_reference']}")
    print(f"Adjusted Reference:  {output['adjusted_reference']}")
    print(f"Validation: {'ALL PASSED ✓' if output['all_valid'] else 'SOME FAILED ✗'}")

    # Save results
    results_df = pd.DataFrame(output["results"])
    results_df.to_csv(os.path.join(output_dir, "wer_comparison.csv"), index=False)

    with open(os.path.join(output_dir, "lattice_results.json"), "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Plot
    plot_wer_comparison(output["results"], os.path.join(output_dir, "lattice_wer_comparison.png"))

    logger.info(f"Lattice WER evaluation complete. Results saved to {output_dir}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lattice-Based WER Evaluation")
    parser.add_argument("--data", default=None, help="Path to data JSON file")
    parser.add_argument("--output-dir", default="outputs/q4", help="Output directory")
    args = parser.parse_args()

    run_lattice_evaluation(args.data, args.output_dir)
