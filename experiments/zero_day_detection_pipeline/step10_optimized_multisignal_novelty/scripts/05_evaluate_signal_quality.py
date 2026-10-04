"""
05_evaluate_signal_quality.py: Information quality, AUROC/AUPRC & Monotonicity of Novelty Signals
Step 10: Optimized Multi-Signal Novelty Fusion
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, roc_curve
from common_utils import load_step10_config, load_frozen_artifacts, ensure_output_dirs, STEP10_DIR
from importlib import import_module

compute_signals_mod = import_module("02_compute_raw_signals")
compute_all_novelty_signals = compute_signals_mod.compute_all_novelty_signals

norm_mod = import_module("03_calibrate_signals")
EmpiricalCDFNormalizer = norm_mod.EmpiricalCDFNormalizer

def run_evaluate_signal_quality():
    print("=" * 80)
    print(">>> STEP 10: 05 - EVALUATE NOVELTY SIGNAL QUALITY & MONOTONICITY <<<")
    print("=" * 80)

    ensure_output_dirs()
    artifacts = load_frozen_artifacts()

    # 1. Load Calibrated Validation Known Signals
    val_calib_path = os.path.join(STEP10_DIR, "outputs", "calibration", "calibrated_validation_signals.parquet")
    df_val_z = pd.read_parquet(val_calib_path)
    n_val = len(df_val_z)
    print(f"Loaded {n_val:,} calibrated validation known flows.")

    # 2. Load Pseudo-Unknowns & Compute Raw Signals
    pseudo_path = os.path.join(STEP10_DIR, "outputs", "pseudo_unknown", "pseudo_unknown_samples.parquet")
    df_pseudo = pd.read_parquet(pseudo_path)
    print(f"Computing raw novelty signals for {len(df_pseudo):,} pseudo-unknown flows...")
    df_pseudo_raw = compute_all_novelty_signals(df_pseudo, artifacts)
    df_pseudo_raw["generator_type"] = df_pseudo["generator_type"].values

    # 3. Load Empirical CDF Normalizer & Transform to Z_i
    norm_path = os.path.join(STEP10_DIR, "outputs", "calibration", "per_signal_calibration.json")
    normalizer = EmpiricalCDFNormalizer()
    normalizer.load(norm_path)

    df_pseudo_z = normalizer.transform(df_pseudo_raw)
    df_pseudo_z["generator_type"] = df_pseudo["generator_type"].values
    df_pseudo_z["pred_idx"] = df_pseudo_raw["pred_idx"].values
    df_pseudo_z["pred_class"] = df_pseudo_raw["pred_class"].values

    # Cache calibrated pseudo-unknowns
    pseudo_calib_path = os.path.join(STEP10_DIR, "outputs", "pseudo_unknown", "pseudo_calibrated_signals.parquet")
    df_pseudo_z.to_parquet(pseudo_calib_path, index=False)
    print(f"Saved calibrated pseudo-unknown signals to: {pseudo_calib_path}")

    signal_keys = [
        ("Z_confidence", "Confidence Novelty (1 - Pmax)"),
        ("Z_mahalanobis", "Mahalanobis Novelty (Dm)"),
        ("Z_leaf", "Leaf-Space Novelty (1 - LeafSim)"),
        ("Z_relative", "Relative Class-Separation (Dm/Dother)")
    ]

    metric_records = []
    roc_curve_records = []

    eval_cohorts = [
        ("All Generators (Aggregate)", df_pseudo_z),
        ("Cross-Class Interpolation", df_pseudo_z[df_pseudo_z["generator_type"] == "cross_class_interpolation"]),
        ("Controlled Perturbation", df_pseudo_z[df_pseudo_z["generator_type"] == "controlled_perturbation"]),
        ("Boundary Low-Density", df_pseudo_z[df_pseudo_z["generator_type"] == "boundary_low_density"]),
        ("Hard Pseudo-Unknowns", df_pseudo_z[df_pseudo_z["generator_type"] == "hard_pseudo_unknown"])
    ]

    print("\n--- Empirical Novelty Signal Discrimination (Validation Known vs Pseudo-Unknown) ---")
    print(f"{'Cohort':<28} | {'Signal':<35} | {'AUROC':<7} | {'AUPRC':<7} | {'Recall@95':<10} | {'F1@95':<7}")
    print("-" * 105)

    for cohort_name, sub_pseudo in eval_cohorts:
        n_p = len(sub_pseudo)
        if n_p == 0:
            continue
        y_true = np.concatenate([np.zeros(n_val, dtype=np.int32), np.ones(n_p, dtype=np.int32)])

        for sig_col, sig_desc in signal_keys:
            scores = np.concatenate([df_val_z[sig_col].values, sub_pseudo[sig_col].values])

            auroc = roc_auc_score(y_true, scores)
            auprc = average_precision_score(y_true, scores)

            # Threshold at 95th percentile of validation score (approx 0.95 for empirical CDF Z_i)
            th_95 = np.percentile(df_val_z[sig_col].values, 95.0)
            y_pred = (scores > th_95).astype(np.int32)
            prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
            fpr = (y_pred[:n_val].sum()) / float(n_val)

            metric_records.append({
                "cohort": cohort_name,
                "signal_key": sig_col,
                "signal_name": sig_desc,
                "pseudo_samples": n_p,
                "auroc": round(float(auroc), 4),
                "auprc": round(float(auprc), 4),
                "threshold_p95": round(float(th_95), 5),
                "recall_at_p95": round(float(rec) * 100.0, 2),
                "precision_at_p95": round(float(prec) * 100.0, 2),
                "f1_at_p95": round(float(f1) * 100.0, 2),
                "fpr_at_p95": round(float(fpr) * 100.0, 2)
            })

            print(f"{cohort_name:<28} | {sig_desc:<35} | {auroc:.4f} | {auprc:.4f} | {rec*100:>8.2f}% | {f1*100:>6.2f}%")

            if cohort_name == "All Generators (Aggregate)":
                fpr_curve, tpr_curve, thresholds = roc_curve(y_true, scores)
                step = max(1, len(fpr_curve) // 100)
                for pt in range(0, len(fpr_curve), step):
                    roc_curve_records.append({
                        "signal_key": sig_col,
                        "signal_name": sig_desc,
                        "fpr": round(float(fpr_curve[pt]), 5),
                        "tpr": round(float(tpr_curve[pt]), 5),
                        "threshold": round(float(thresholds[pt]), 5)
                    })

    df_metrics = pd.DataFrame(metric_records)
    out_metrics_path = os.path.join(STEP10_DIR, "outputs", "signal_analysis", "signal_metrics.csv")
    df_metrics.to_csv(out_metrics_path, index=False)

    df_roc = pd.DataFrame(roc_curve_records)
    out_roc_path = os.path.join(STEP10_DIR, "outputs", "signal_analysis", "signal_roc_data.csv")
    df_roc.to_csv(out_roc_path, index=False)

    # 4. Correlation and Redundancy Analysis
    sig_cols_list = ["Z_confidence", "Z_mahalanobis", "Z_leaf", "Z_relative"]
    # Calibration set: Validation Known + Pseudo-Unknowns
    df_calib_combined = pd.concat([df_val_z[sig_cols_list], df_pseudo_z[sig_cols_list]], ignore_index=True)
    corr_spearman = df_calib_combined.corr(method="spearman")
    out_corr_path = os.path.join(STEP10_DIR, "outputs", "signal_analysis", "signal_correlation_matrix.csv")
    corr_spearman.to_csv(out_corr_path)

    print("\n--- Spearman Rank Correlation Matrix on Calibration Data (Validation + Pseudo-Unknowns) ---")
    print(corr_spearman.round(4).to_string())

    # Standalone Signal AUROC and AUPRC Summary
    agg_metrics = df_metrics[df_metrics["cohort"] == "All Generators (Aggregate)"]
    print("\n--- Individual Novelty Signal Standalone Performance (Aggregate Pseudo-Unknowns) ---")
    for _, row in agg_metrics.iterrows():
        print(f"  {row['signal_key']:<16} ({row['signal_name']}): AUROC = {row['auroc']:.4f} | AUPRC = {row['auprc']:.4f} | Recall@95 = {row['recall_at_p95']:.2f}% | F1@95 = {row['f1_at_p95']:.2f}%")

    # 4. Monotonicity and Calibration Verification
    # Check that for deciles of Z_i, pseudo-unknown density increases monotonically
    monotonicity_report = {}
    deciles = np.linspace(0.0, 1.0, 11)
    for sig_col, sig_desc in signal_keys:
        val_s = df_val_z[sig_col].values
        pseudo_s = df_pseudo_z[sig_col].values
        bin_counts_val, _ = np.histogram(val_s, bins=deciles)
        bin_counts_pseudo, _ = np.histogram(pseudo_s, bins=deciles)

        pseudo_ratio = bin_counts_pseudo / np.maximum(1, (bin_counts_val + bin_counts_pseudo))
        is_monotonic_trend = bool(pseudo_ratio[-1] > pseudo_ratio[0])

        monotonicity_report[sig_col] = {
            "signal_name": sig_desc,
            "decile_bins": [round(float(d), 2) for d in deciles],
            "known_bin_counts": [int(c) for c in bin_counts_val],
            "pseudo_bin_counts": [int(c) for c in bin_counts_pseudo],
            "pseudo_proportion_by_bin": [round(float(r), 4) for r in pseudo_ratio],
            "monotonic_separation_verified": is_monotonic_trend
        }

    out_mono_path = os.path.join(STEP10_DIR, "outputs", "signal_analysis", "monotonicity_check.json")
    with open(out_mono_path, "w", encoding="utf-8") as f:
        json.dump(monotonicity_report, f, indent=2)

    print(f"\nSaved signal metrics to: {out_metrics_path}")
    print(f"Saved ROC curve points to: {out_roc_path}")
    print(f"Saved monotonicity check to: {out_mono_path}")
    return df_metrics

if __name__ == "__main__":
    run_evaluate_signal_quality()
