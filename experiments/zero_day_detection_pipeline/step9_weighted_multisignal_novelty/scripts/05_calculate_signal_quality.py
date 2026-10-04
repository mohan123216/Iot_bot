"""
05_calculate_signal_quality.py: Measure information quality of each novelty signal
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, roc_curve
from common_utils import load_step9_config, load_frozen_artifacts
from importlib import import_module

compute_signals_mod = import_module("02_compute_novelty_signals")
compute_all_novelty_signals = compute_signals_mod.compute_all_novelty_signals

norm_mod = import_module("03_percentile_normalization")
PercentileNormalizer = norm_mod.PercentileNormalizer

def run_calculate_signal_quality():
    print("=" * 80)
    print(">>> STEP 9: 05 - CALCULATE NOVELTY SIGNAL QUALITY (AUROC, AUPRC, F1) <<<")
    print("=" * 80)

    step9_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts = load_frozen_artifacts()

    # 1. Load Validation Known Signals
    val_signals_path = os.path.join(step9_dir, "outputs", "calibration", "calibration_signal_scores.parquet")
    df_val_signals = pd.read_parquet(val_signals_path)
    n_val = len(df_val_signals)

    # 2. Load Pseudo-Unknowns & Compute Signals
    pseudo_path = os.path.join(step9_dir, "outputs", "pseudo_unknown", "pseudo_unknown_samples.parquet")
    df_pseudo = pd.read_parquet(pseudo_path)
    print(f"Computing novelty signals for {len(df_pseudo):,} pseudo-unknown flows...")
    df_pseudo_signals = compute_all_novelty_signals(df_pseudo, artifacts)
    df_pseudo_signals["generator_type"] = df_pseudo["generator_type"].values

    # 3. Load Percentile Normalizer & Transform Signals to Z_i
    norm_path = os.path.join(step9_dir, "outputs", "calibration", "percentile_reference.json")
    normalizer = PercentileNormalizer()
    normalizer.load(norm_path)

    Z_val = normalizer.transform(df_val_signals)
    Z_pseudo = normalizer.transform(df_pseudo_signals)

    # Add generator type to pseudo Z
    Z_pseudo["generator_type"] = df_pseudo_signals["generator_type"].values

    signal_keys = [
        ("Z_confidence", "Confidence Novelty (1 - Pmax)"),
        ("Z_mahalanobis", "Mahalanobis Novelty (Dm)"),
        ("Z_leaf", "Leaf-Space Novelty (1 - LeafSim)"),
        ("Z_relative", "Relative Class-Separation (Dm/Dother)")
    ]

    metric_records = []
    roc_curve_records = []

    # Generators to evaluate
    eval_cohorts = [
        ("All Generators (Aggregate)", Z_pseudo),
        ("Cross-Class Interpolation", Z_pseudo[Z_pseudo["generator_type"] == "cross_class_interpolation"]),
        ("Controlled Perturbation", Z_pseudo[Z_pseudo["generator_type"] == "controlled_perturbation"]),
        ("Boundary Low-Density", Z_pseudo[Z_pseudo["generator_type"] == "boundary_low_density"])
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
            scores = np.concatenate([Z_val[sig_col].values, sub_pseudo[sig_col].values])

            auroc = roc_auc_score(y_true, scores)
            auprc = average_precision_score(y_true, scores)

            # Performance at 95th percentile threshold (which is 0.95 for Z_i)
            th_95 = 0.95
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
                "threshold_p95": round(float(th_95), 4),
                "recall_at_p95": round(float(rec) * 100.0, 2),
                "precision_at_p95": round(float(prec) * 100.0, 2),
                "f1_at_p95": round(float(f1) * 100.0, 2),
                "fpr_at_p95": round(float(fpr) * 100.0, 2)
            })

            print(f"{cohort_name:<28} | {sig_desc:<35} | {auroc:.4f} | {auprc:.4f} | {rec*100:>8.2f}% | {f1*100:>6.2f}%")

            # Store ROC curve points for aggregate cohort
            if cohort_name == "All Generators (Aggregate)":
                fpr_curve, tpr_curve, thresholds = roc_curve(y_true, scores)
                # Sample 100 points for compact CSV
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
    out_metrics_path = os.path.join(step9_dir, "outputs", "signal_analysis", "signal_metrics.csv")
    df_metrics.to_csv(out_metrics_path, index=False)

    df_roc = pd.DataFrame(roc_curve_records)
    out_roc_path = os.path.join(step9_dir, "outputs", "signal_analysis", "signal_roc_data.csv")
    df_roc.to_csv(out_roc_path, index=False)

    print(f"\nSaved signal metrics to: {out_metrics_path}")
    print(f"Saved ROC curve points to: {out_roc_path}")
    return df_metrics

if __name__ == "__main__":
    run_calculate_signal_quality()
