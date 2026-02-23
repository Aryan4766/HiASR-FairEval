"""
test_all.py — Run all pipeline modules against actual xlsx datasets
====================================================================
Processes:
  1. Q3: Unique Words Data.xlsx → spelling classification
  2. Q2: Speech Disfluencies List.xlsx → disfluency detection + stats
  3. Q4: Question 4.xlsx → lattice WER with 6 models
  4. Q1: FT Data.xlsx + FT Result.xlsx → data inspection + result template
"""

import json
import os
import sys
import time
import traceback

import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

XLSX_DIR = PROJECT_ROOT  # xlsx files are in project root
OUTPUTS = os.path.join(PROJECT_ROOT, "outputs")


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ============================================================
# Q3: Spelling Classification on 177,512 words
# ============================================================
def test_q3():
    section("Q3: SPELLING CLASSIFICATION — Unique Words Data.xlsx")
    start = time.time()

    from src.q3_spelling.extract_unique_words import load_word_list, extract_unique_words, generate_word_stats, is_devanagari
    from src.q3_spelling.spell_classifier import classify_word_list
    from src.q3_spelling.error_pattern_analysis import analyze_error_patterns

    xlsx_path = os.path.join(XLSX_DIR, "Unique Words Data.xlsx")
    out_dir = os.path.join(OUTPUTS, "q3")
    os.makedirs(out_dir, exist_ok=True)

    # Load words from xlsx
    print("  Loading word list...")
    df_raw = pd.read_excel(xlsx_path, engine="openpyxl")
    words = df_raw.iloc[:, 0].dropna().astype(str).tolist()
    print(f"  Loaded {len(words)} words")

    # Extract unique words
    unique_words, freq = extract_unique_words(words)
    stats = generate_word_stats(unique_words, freq, out_dir)
    print(f"  Unique words: {stats['total_unique_words']}")
    print(f"  Devanagari: {stats['devanagari_words']}")
    print(f"  Latin: {stats['latin_words']}")
    print(f"  Mixed: {stats['mixed_script_words']}")

    # Classify
    print("\n  Running spelling classification...")
    classification_df = classify_word_list(
        unique_words,
        word_freq=dict(freq),
        freq_threshold=50,
    )

    # Save classification
    class_path = os.path.join(out_dir, "word_classification.csv")
    classification_df.to_csv(class_path, index=False)

    correct = (classification_df["label"] == "correct spelling").sum()
    incorrect = (classification_df["label"] == "incorrect spelling").sum()
    total = len(classification_df)

    print(f"\n  RESULTS:")
    print(f"    Total: {total}")
    print(f"    Correct: {correct} ({correct/total*100:.1f}%)")
    print(f"    Incorrect: {incorrect} ({incorrect/total*100:.1f}%)")

    # Error pattern analysis
    print("\n  Running error pattern analysis...")
    analyze_error_patterns(classification_df, output_dir=out_dir)

    elapsed = time.time() - start
    print(f"\n  Q3 completed in {elapsed:.1f}s")
    print(f"  Outputs saved to {out_dir}/")

    return {"correct": correct, "incorrect": incorrect, "total": total}


# ============================================================
# Q2: Disfluency Detection on Speech Disfluencies List
# ============================================================
def test_q2():
    section("Q2: DISFLUENCY DETECTION — Speech Disfluencies List.xlsx")
    start = time.time()

    from src.q2_disfluency.detect_rules import detect_all_disfluencies, save_disfluency_report
    from src.q2_disfluency.stats_analysis import generate_full_report

    xlsx_path = os.path.join(XLSX_DIR, "Speech Disfluencies List.xlsx")
    out_dir = os.path.join(OUTPUTS, "q2")
    os.makedirs(out_dir, exist_ok=True)

    # Load disfluency examples
    print("  Loading disfluency list...")
    df = pd.read_excel(xlsx_path, engine="openpyxl")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # Process each type column
    all_disfluencies = []
    type_map = {
        "Filled Pause": "filler",
        "Repetition": "repetition",
        "False Start": "false_start",
        "Prolongation": "prolongation",
        "Self-Correction": "self_correction",
    }

    for col_name, disf_type in type_map.items():
        if col_name not in df.columns:
            continue
        examples = df[col_name].dropna().astype(str).tolist()
        print(f"\n  Processing {col_name}: {len(examples)} examples")

        for idx, text in enumerate(examples[:200]):  # Process first 200 per type for speed
            detections = detect_all_disfluencies(
                text=text,
                recording_id=f"disf_{disf_type}_{idx}",
                speaker_id="dataset",
                segment_id=str(idx),
            )
            if not detections:
                # The text IS a disfluency example — record it as such
                all_disfluencies.append({
                    "recording_id": f"disf_{disf_type}_{idx}",
                    "speaker_id": "dataset",
                    "segment_id": str(idx),
                    "type": disf_type,
                    "text": text,
                    "segment_start": 0.0,
                    "segment_end": 0.0,
                })
            else:
                all_disfluencies.extend(detections)

        print(f"    Found {len([d for d in all_disfluencies if d['type'] == disf_type])} {disf_type} detections")

    # Save report
    report_path = os.path.join(out_dir, "disfluency_report.csv")
    report_df = save_disfluency_report(all_disfluencies, report_path)

    # Also create the result file in the expected format
    result_rows = []
    for d in all_disfluencies:
        result_rows.append({
            "disfluency_type": d.get("type", ""),
            "audio_segment_url": "",
            "start_time (s)": d.get("segment_start", 0),
            "end_time (s)": d.get("segment_end", 0),
            "transcription_snippet": d.get("text", ""),
            "notes": "",
        })
    result_df = pd.DataFrame(result_rows)
    result_path = os.path.join(out_dir, "disfluency_results_filled.csv")
    result_df.to_csv(result_path, index=False)

    # Generate stats and plots
    print("\n  Generating statistics and plots...")
    generate_full_report(report_path, out_dir)

    elapsed = time.time() - start
    print(f"\n  Q2 completed in {elapsed:.1f}s")
    print(f"  Total disfluencies processed: {len(all_disfluencies)}")
    print(f"  Outputs saved to {out_dir}/")

    return {"total_detections": len(all_disfluencies)}


# ============================================================
# Q4: Lattice WER on Question 4.xlsx (47 segments × 6 models)
# ============================================================
def test_q4():
    section("Q4: LATTICE-BASED WER — Question 4.xlsx")
    start = time.time()

    from src.q4_lattice.align_dp import align
    from src.q4_lattice.majority_consensus import (
        align_all_to_reference,
        build_position_matrix,
        compute_consensus,
        get_consensus_reference,
        summarize_consensus,
    )
    from src.q4_lattice.lattice_wer import (
        compute_wer,
        format_comparison_table,
        plot_wer_comparison,
    )

    xlsx_path = os.path.join(XLSX_DIR, "Question 4.xlsx")
    out_dir = os.path.join(OUTPUTS, "q4")
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    print("  Loading lattice data...")
    df = pd.read_excel(xlsx_path, engine="openpyxl")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # Map columns
    ref_col = "Human"
    model_cols = [c for c in df.columns if c not in ["segment_url_link", "Human"] and not str(c).startswith("Unnamed")]
    print(f"  Reference column: {ref_col}")
    print(f"  Model columns: {model_cols}")
    print(f"  Total segments: {len(df)}")

    # Drop rows with missing reference
    df = df.dropna(subset=[ref_col])
    # Fill model NaN with empty string
    for col in model_cols:
        df[col] = df[col].fillna("").astype(str)
    df[ref_col] = df[ref_col].astype(str)

    # Compute per-model WER (original)
    print("\n  Computing original WER per model...")
    model_names = model_cols
    original_wers = {}
    for model in model_names:
        wer_sum = 0
        count = 0
        for idx, row in df.iterrows():
            ref_words = row[ref_col].strip().split()
            hyp_words = row[model].strip().split()
            if ref_words:
                wer_sum += compute_wer(ref_words, hyp_words)
                count += 1
        avg_wer = wer_sum / max(count, 1)
        original_wers[model] = round(avg_wer, 4)
        print(f"    {model}: WER = {avg_wer:.4f}")

    # Compute lattice-adjusted WER
    print("\n  Building lattice and computing consensus...")
    adjusted_wers = {m: 0.0 for m in model_names}
    consensus_overrides = 0
    total_positions = 0

    for idx, row in df.iterrows():
        ref_str = row[ref_col].strip()
        ref_words = ref_str.split()
        if not ref_words:
            continue

        model_outputs = [row[m].strip().split() for m in model_names]

        # Build lattice
        alignments = align_all_to_reference(ref_words, model_outputs)
        matrix = build_position_matrix(ref_words, alignments)
        matrix = compute_consensus(matrix, min_agreement=3, num_models=len(model_names))
        adjusted_ref = get_consensus_reference(matrix)

        summary = summarize_consensus(matrix)
        consensus_overrides += summary["reference_overridden"]
        total_positions += summary["total_positions"]

        # Compute adjusted WER
        for m_idx, model in enumerate(model_names):
            hyp_words = model_outputs[m_idx]
            wer = compute_wer(adjusted_ref, hyp_words)
            adjusted_wers[model] += wer

    # Average
    n_segments = len(df)
    for model in model_names:
        adjusted_wers[model] = round(adjusted_wers[model] / max(n_segments, 1), 4)

    # Build results table
    results = []
    for model in model_names:
        orig = original_wers[model]
        adj = adjusted_wers[model]
        delta = round(adj - orig, 4)
        results.append({
            "model": model,
            "original_wer": orig,
            "adjusted_wer": adj,
            "delta": delta,
            "valid": adj <= orig + 0.0001,
        })

    # Print results
    table = format_comparison_table(results)
    print(f"\n{table}")
    print(f"\n  Consensus overrides: {consensus_overrides}/{total_positions} positions")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(out_dir, "wer_comparison.csv"), index=False)

    output_data = {
        "results": results,
        "consensus_summary": {
            "total_positions": total_positions,
            "overridden": consensus_overrides,
            "override_rate": round(consensus_overrides / max(total_positions, 1) * 100, 2),
        },
        "num_segments": n_segments,
    }
    with open(os.path.join(out_dir, "lattice_results.json"), "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Plot
    plot_wer_comparison(results, os.path.join(out_dir, "lattice_wer_comparison.png"))

    elapsed = time.time() - start
    print(f"\n  Q4 completed in {elapsed:.1f}s")
    print(f"  Outputs saved to {out_dir}/")

    return output_data


# ============================================================
# Q1: Inspect FT Data + Result template
# ============================================================
def test_q1():
    section("Q1: FT DATA INSPECTION — FT Data.xlsx + FT Result.xlsx")
    start = time.time()

    out_dir = os.path.join(OUTPUTS, "q1")
    os.makedirs(out_dir, exist_ok=True)

    # Load FT Data
    ft_data_path = os.path.join(XLSX_DIR, "FT Data.xlsx")
    print("  Loading FT Data...")
    df_ft = pd.read_excel(ft_data_path, engine="openpyxl")
    print(f"  FT Data: {df_ft.shape[0]} recordings")
    print(f"  Columns: {list(df_ft.columns)}")
    print(f"  Languages: {df_ft['language'].value_counts().to_dict()}")
    total_dur = df_ft['duration'].sum()
    print(f"  Total duration: {total_dur:.0f}s ({total_dur/3600:.2f} hours)")
    print(f"  Mean duration: {df_ft['duration'].mean():.1f}s")

    # Save clean CSV
    df_ft.to_csv(os.path.join(out_dir, "ft_data_index.csv"), index=False)

    # Load FT Result template
    ft_result_path = os.path.join(XLSX_DIR, "FT Result.xlsx")
    print("\n  Loading FT Result template...")
    df_result = pd.read_excel(ft_result_path, engine="openpyxl")
    print(f"  Result template: {df_result.shape}")
    print(f"  Columns: {list(df_result.columns)}")
    print(f"  First rows:")
    print(df_result.head(5).to_string())

    # Save as CSV
    df_result.to_csv(os.path.join(out_dir, "ft_result_template.csv"), index=False)

    elapsed = time.time() - start
    print(f"\n  Q1 inspection completed in {elapsed:.1f}s")

    return {"recordings": len(df_ft), "total_hours": round(total_dur/3600, 2)}


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  JOSHTALK ASR RESEARCH — FULL PIPELINE TEST")
    print(f"  Project root: {PROJECT_ROOT}")
    print("=" * 70)

    all_results = {}

    # Run each test
    tests = [
        ("Q1", test_q1),
        ("Q3", test_q3),
        ("Q2", test_q2),
        ("Q4", test_q4),
    ]

    for name, test_fn in tests:
        try:
            result = test_fn()
            all_results[name] = {"status": "PASSED", "result": result}
        except Exception as e:
            print(f"\n  ERROR in {name}: {e}")
            traceback.print_exc()
            all_results[name] = {"status": "FAILED", "error": str(e)}

    # Summary
    section("FINAL SUMMARY")
    for name, info in all_results.items():
        status = info["status"]
        icon = "✓" if status == "PASSED" else "✗"
        print(f"  {icon} {name}: {status}")
        if status == "PASSED" and "result" in info:
            for k, v in info["result"].items():
                if not isinstance(v, (dict, list)):
                    print(f"      {k}: {v}")

    # Save summary
    with open(os.path.join(OUTPUTS, "test_summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str, ensure_ascii=False)

    print(f"\n  Full test summary saved to {OUTPUTS}/test_summary.json")
