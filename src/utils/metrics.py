"""
Metrics Module — JoshTalks ASR Research
========================================
WER/CER computation with detailed error breakdown.
Uses jiwer for standard metrics.
"""

from typing import Any, Dict, List, Tuple

import jiwer


def compute_wer_detailed(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Compute detailed WER breakdown for a single utterance pair.

    Args:
        reference: Ground-truth transcription.
        hypothesis: Model prediction.

    Returns:
        Dictionary with WER, CER, substitution/insertion/deletion rates.
    """
    # Handle empty strings
    if not reference.strip():
        if not hypothesis.strip():
            return {
                "wer": 0.0, "cer": 0.0,
                "substitutions": 0, "insertions": 0, "deletions": 0,
                "hits": 0, "ref_length": 0,
            }
        else:
            return {
                "wer": 1.0, "cer": 1.0,
                "substitutions": 0, "insertions": len(hypothesis.split()), "deletions": 0,
                "hits": 0, "ref_length": 0,
            }

    # Word-level metrics
    wer_output = jiwer.process_words(reference, hypothesis)
    ref_len = wer_output.substitutions + wer_output.deletions + wer_output.hits

    wer = (wer_output.substitutions + wer_output.deletions + wer_output.insertions) / max(ref_len, 1)

    # Character-level metrics
    cer_output = jiwer.process_characters(reference, hypothesis)
    cer_ref_len = cer_output.substitutions + cer_output.deletions + cer_output.hits
    cer = (cer_output.substitutions + cer_output.deletions + cer_output.insertions) / max(cer_ref_len, 1)

    return {
        "wer": round(wer, 4),
        "cer": round(cer, 4),
        "substitutions": wer_output.substitutions,
        "insertions": wer_output.insertions,
        "deletions": wer_output.deletions,
        "hits": wer_output.hits,
        "ref_length": ref_len,
    }


def compute_corpus_metrics(
    references: List[str],
    hypotheses: List[str],
) -> Dict[str, Any]:
    """
    Compute corpus-level WER/CER with aggregated breakdown.

    Args:
        references: List of reference transcriptions.
        hypotheses: List of hypothesis transcriptions.

    Returns:
        Dictionary with corpus-level and per-utterance metrics.
    """
    assert len(references) == len(hypotheses), (
        f"Length mismatch: {len(references)} references vs {len(hypotheses)} hypotheses"
    )

    # Corpus-level WER
    corpus_wer = jiwer.wer(references, hypotheses)
    corpus_cer = jiwer.cer(references, hypotheses)

    # Detailed word-level breakdown
    wer_output = jiwer.process_words(references, hypotheses)
    total_ref = wer_output.substitutions + wer_output.deletions + wer_output.hits

    # Per-utterance metrics
    per_utterance = []
    for ref, hyp in zip(references, hypotheses):
        per_utterance.append(compute_wer_detailed(ref, hyp))

    return {
        "corpus_wer": round(corpus_wer, 4),
        "corpus_cer": round(corpus_cer, 4),
        "total_substitutions": wer_output.substitutions,
        "total_insertions": wer_output.insertions,
        "total_deletions": wer_output.deletions,
        "total_hits": wer_output.hits,
        "total_ref_words": total_ref,
        "substitution_rate": round(wer_output.substitutions / max(total_ref, 1), 4),
        "insertion_rate": round(wer_output.insertions / max(total_ref, 1), 4),
        "deletion_rate": round(wer_output.deletions / max(total_ref, 1), 4),
        "num_utterances": len(references),
        "per_utterance": per_utterance,
    }


def get_error_pairs(
    references: List[str],
    hypotheses: List[str],
) -> Tuple[List[Tuple[str, str]], List[str], List[str]]:
    """
    Extract substitution pairs, deleted words, and inserted words.

    Args:
        references: List of reference transcriptions.
        hypotheses: List of hypothesis transcriptions.

    Returns:
        Tuple of (substitution_pairs, deleted_words, inserted_words).
    """
    substitutions = []
    deleted_words = []
    inserted_words = []

    for ref, hyp in zip(references, hypotheses):
        output = jiwer.process_words(ref, hyp)
        ref_words = ref.split()
        hyp_words = hyp.split()

        for chunk in output.alignments[0]:
            if chunk.type == "substitute":
                for i, j in zip(
                    range(chunk.ref_start_idx, chunk.ref_end_idx),
                    range(chunk.hyp_start_idx, chunk.hyp_end_idx),
                ):
                    substitutions.append((ref_words[i], hyp_words[j]))
            elif chunk.type == "delete":
                for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                    deleted_words.append(ref_words[i])
            elif chunk.type == "insert":
                for j in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                    inserted_words.append(hyp_words[j])

    return substitutions, deleted_words, inserted_words
