"""
Majority Consensus — JoshTalks ASR Research
==============================================
Construct a lattice from multiple ASR model outputs and
determine consensus transcription.

Logic:
- Align all model outputs to reference at word level
- At each position, if ≥3 out of 5 models agree → consensus word
- If consensus differs from reference → reference may be wrong
- Use consensus to create a "fair" reference for WER
"""

import os
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.logger import setup_logger
from src.q4_lattice.align_dp import align, MATCH, SUBSTITUTE, INSERT, DELETE

logger = setup_logger("majority_consensus", log_to_file=True)


def align_all_to_reference(
    reference: List[str],
    model_outputs: List[List[str]],
) -> List[List[Tuple]]:
    """
    Align all model outputs to the reference.

    Args:
        reference: Reference word list.
        model_outputs: List of model output word lists.

    Returns:
        List of alignments (one per model).
    """
    alignments = []
    for i, output in enumerate(model_outputs):
        alignment, distance, stats = align(reference, output)
        alignments.append(alignment)
        logger.debug(f"Model {i+1} alignment: distance={distance}, stats={stats}")
    return alignments


def build_position_matrix(
    reference: List[str],
    alignments: List[List[Tuple]],
) -> List[Dict]:
    """
    Build a position-by-position matrix of model outputs.

    For each reference position, collect what each model produced
    (the word it substituted, or whether it deleted/inserted).

    Args:
        reference: Reference word list.
        alignments: List of model alignments to reference.

    Returns:
        List of position dictionaries, each containing:
        - ref_word: Reference word at this position
        - model_words: List of what each model produced
        - operations: List of operations per model
    """
    positions = []

    for alignment in alignments:
        pos_idx = 0
        model_positions = {}

        for ref_w, hyp_w, op in alignment:
            if op in (MATCH, SUBSTITUTE):
                if pos_idx not in model_positions:
                    model_positions[pos_idx] = []
                model_positions[pos_idx].append((hyp_w, op))
                pos_idx += 1
            elif op == DELETE:
                if pos_idx not in model_positions:
                    model_positions[pos_idx] = []
                model_positions[pos_idx].append(("<DEL>", DELETE))
                pos_idx += 1
            elif op == INSERT:
                # Insertion doesn't consume a reference position
                pass

        positions.append(model_positions)

    # Build matrix
    matrix = []
    for ref_idx, ref_word in enumerate(reference):
        pos = {
            "ref_idx": ref_idx,
            "ref_word": ref_word,
            "model_words": [],
            "operations": [],
        }

        for model_positions in positions:
            if ref_idx in model_positions and model_positions[ref_idx]:
                word, op = model_positions[ref_idx][0]
                pos["model_words"].append(word)
                pos["operations"].append(op)
            else:
                pos["model_words"].append("<MISSING>")
                pos["operations"].append("missing")

        matrix.append(pos)

    return matrix


def compute_consensus(
    position_matrix: List[Dict],
    min_agreement: int = 3,
    num_models: int = 5,
) -> List[Dict]:
    """
    Apply majority voting at each position to build consensus reference.

    Rules:
    - If ≥min_agreement models agree on a word → use that word as consensus
    - If consensus differs from reference → reference was likely wrong
    - If no consensus → keep reference word

    Args:
        position_matrix: Position matrix from build_position_matrix.
        min_agreement: Minimum models that must agree.
        num_models: Total number of models.

    Returns:
        Enhanced position matrix with consensus information.
    """
    for pos in position_matrix:
        model_words = pos["model_words"]
        ref_word = pos["ref_word"]

        # Count word frequencies across models (excluding special tokens)
        valid_words = [w for w in model_words if w not in ("<DEL>", "<MISSING>")]
        word_counts = Counter(valid_words)

        if word_counts:
            most_common_word, most_common_count = word_counts.most_common(1)[0]

            if most_common_count >= min_agreement:
                pos["consensus_word"] = most_common_word
                pos["consensus_count"] = most_common_count
                pos["consensus_matches_ref"] = (most_common_word == ref_word)

                if not pos["consensus_matches_ref"]:
                    pos["ref_override"] = True
                    pos["reason"] = (
                        f"{most_common_count}/{num_models} models agree on "
                        f"'{most_common_word}' vs reference '{ref_word}'"
                    )
                else:
                    pos["ref_override"] = False
                    pos["reason"] = "consensus matches reference"
            else:
                pos["consensus_word"] = ref_word
                pos["consensus_count"] = 0
                pos["consensus_matches_ref"] = True
                pos["ref_override"] = False
                pos["reason"] = "no majority consensus — keeping reference"
        else:
            pos["consensus_word"] = ref_word
            pos["consensus_count"] = 0
            pos["consensus_matches_ref"] = True
            pos["ref_override"] = False
            pos["reason"] = "all models deleted or missing"

    return position_matrix


def get_consensus_reference(position_matrix: List[Dict]) -> List[str]:
    """
    Extract the consensus reference word list.

    Args:
        position_matrix: Position matrix with consensus computed.

    Returns:
        List of consensus reference words.
    """
    return [pos["consensus_word"] for pos in position_matrix]


def summarize_consensus(position_matrix: List[Dict]) -> Dict:
    """
    Summarize consensus decisions.

    Returns:
        Summary dictionary.
    """
    total_positions = len(position_matrix)
    overrides = sum(1 for pos in position_matrix if pos.get("ref_override", False))
    matches = total_positions - overrides

    return {
        "total_positions": total_positions,
        "reference_kept": matches,
        "reference_overridden": overrides,
        "override_rate": round(overrides / max(total_positions, 1) * 100, 2),
    }


if __name__ == "__main__":
    # Demo with synthetic data
    reference = ["यह", "एक", "अच्छा", "दिन", "है"]

    model_outputs = [
        ["यह", "एक", "अच्छा", "दिन", "है"],     # Model 1: perfect
        ["यह", "बहुत", "अच्छा", "दिन", "है"],     # Model 2: substitution
        ["ये", "एक", "अच्छा", "दिन", "है"],       # Model 3: substitution
        ["यह", "एक", "अच्छा", "दिन", "है"],       # Model 4: perfect
        ["यह", "एक", "बढ़िया", "दिन", "है"],       # Model 5: substitution
    ]

    print("Reference:", " ".join(reference))
    for i, out in enumerate(model_outputs):
        print(f"Model {i+1}:  ", " ".join(out))

    alignments = align_all_to_reference(reference, model_outputs)
    matrix = build_position_matrix(reference, alignments)
    matrix = compute_consensus(matrix, min_agreement=3)
    consensus = get_consensus_reference(matrix)

    print(f"\nConsensus:  {' '.join(consensus)}")
    summary = summarize_consensus(matrix)
    print(f"Summary: {summary}")

    for pos in matrix:
        if pos.get("ref_override"):
            print(f"\n  Override at position {pos['ref_idx']}: {pos['reason']}")
