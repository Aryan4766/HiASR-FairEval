"""
Word-Level DP Alignment — JoshTalks ASR Research
===================================================
Minimum edit distance alignment at the word level.
Used for lattice-based WER computation.

Alignment unit: WORD
Justification: Word-level alignment is chosen because:
1. Standard WER operates at word level — direct comparison
2. Hindi words are well-delimited by whitespace
3. Subword alignment would introduce tokenizer-dependent artifacts
4. Phrase-level would miss fine-grained substitution patterns
"""

import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.logger import setup_logger

logger = setup_logger("align_dp", log_to_file=False)


# Operation types
MATCH = "match"
SUBSTITUTE = "substitute"
INSERT = "insert"
DELETE = "delete"


def align(
    reference: List[str],
    hypothesis: List[str],
    sub_cost: float = 1.0,
    ins_cost: float = 1.0,
    del_cost: float = 1.0,
) -> Tuple[List[Tuple], float, Dict]:
    """
    Word-level dynamic programming alignment (minimum edit distance).

    Args:
        reference: List of reference words.
        hypothesis: List of hypothesis words.
        sub_cost: Cost of substitution.
        ins_cost: Cost of insertion.
        del_cost: Cost of deletion.

    Returns:
        Tuple of:
        - alignment: List of (ref_word, hyp_word, operation) tuples
        - distance: Total edit distance
        - stats: Dictionary with operation counts
    """
    m = len(reference)
    n = len(hypothesis)

    # DP table
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    backtrack = [[""] * (n + 1) for _ in range(m + 1)]

    # Initialize
    for i in range(1, m + 1):
        dp[i][0] = i * del_cost
        backtrack[i][0] = DELETE

    for j in range(1, n + 1):
        dp[0][j] = j * ins_cost
        backtrack[0][j] = INSERT

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                # Match
                dp[i][j] = dp[i - 1][j - 1]
                backtrack[i][j] = MATCH
            else:
                # Substitution
                sub = dp[i - 1][j - 1] + sub_cost
                # Deletion (ref word not in hyp)
                dele = dp[i - 1][j] + del_cost
                # Insertion (extra word in hyp)
                ins = dp[i][j - 1] + ins_cost

                min_cost = min(sub, dele, ins)
                dp[i][j] = min_cost

                if min_cost == sub:
                    backtrack[i][j] = SUBSTITUTE
                elif min_cost == dele:
                    backtrack[i][j] = DELETE
                else:
                    backtrack[i][j] = INSERT

    # Backtrace to get alignment
    alignment = []
    i, j = m, n

    while i > 0 or j > 0:
        if i > 0 and j > 0 and backtrack[i][j] == MATCH:
            alignment.append((reference[i - 1], hypothesis[j - 1], MATCH))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and backtrack[i][j] == SUBSTITUTE:
            alignment.append((reference[i - 1], hypothesis[j - 1], SUBSTITUTE))
            i -= 1
            j -= 1
        elif i > 0 and backtrack[i][j] == DELETE:
            alignment.append((reference[i - 1], "<DEL>", DELETE))
            i -= 1
        elif j > 0 and backtrack[i][j] == INSERT:
            alignment.append(("<INS>", hypothesis[j - 1], INSERT))
            j -= 1
        else:
            break

    alignment.reverse()

    # Compute stats
    stats = {
        MATCH: sum(1 for _, _, op in alignment if op == MATCH),
        SUBSTITUTE: sum(1 for _, _, op in alignment if op == SUBSTITUTE),
        INSERT: sum(1 for _, _, op in alignment if op == INSERT),
        DELETE: sum(1 for _, _, op in alignment if op == DELETE),
    }

    return alignment, dp[m][n], stats


def align_strings(
    reference_str: str,
    hypothesis_str: str,
    sub_cost: float = 1.0,
    ins_cost: float = 1.0,
    del_cost: float = 1.0,
) -> Tuple[List[Tuple], float, Dict]:
    """
    Convenience wrapper: align two strings by splitting on whitespace.

    Args:
        reference_str: Reference transcription string.
        hypothesis_str: Hypothesis transcription string.

    Returns:
        Same as align().
    """
    ref_words = reference_str.strip().split()
    hyp_words = hypothesis_str.strip().split()
    return align(ref_words, hyp_words, sub_cost, ins_cost, del_cost)


def print_alignment(alignment: List[Tuple]) -> None:
    """Pretty-print an alignment."""
    print(f"\n{'Ref':<20} {'Hyp':<20} {'Op':<12}")
    print("-" * 52)
    for ref_w, hyp_w, op in alignment:
        marker = "" if op == MATCH else " ←"
        print(f"{ref_w:<20} {hyp_w:<20} {op:<12}{marker}")


if __name__ == "__main__":
    # Demo
    ref = ["यह", "एक", "अच्छा", "दिन", "है"]
    hyp = ["यह", "बहुत", "अच्छा", "है"]

    alignment, distance, stats = align(ref, hyp)

    print(f"Edit Distance: {distance}")
    print(f"Stats: {stats}")
    print_alignment(alignment)
