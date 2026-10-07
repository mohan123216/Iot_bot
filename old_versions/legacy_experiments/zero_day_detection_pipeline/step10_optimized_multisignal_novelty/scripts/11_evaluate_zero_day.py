"""
11_evaluate_zero_day.py: Unbiased evaluation on held-out zero-day attack (Service_Scan)
Step 10: Optimized Multi-Signal Novelty Fusion
Strict Rule: Evaluated ONLY ONCE after all models, weights, and thresholds are strictly frozen.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from common_utils import load_step10_config, get_abs_path, load_frozen_artifacts, ensure_output_dirs, STEP10_DIR
from importlib import import_module

compute_signals_mod = import_module("02_compute_raw_signals")
compute_all_novelty_signals = compute_signals_mod.compute_all_novelty_signals

norm_mod = import_module("03_calibrate_signals")
EmpiricalCDFNormalizer = norm_mod.EmpiricalCDFNormalizer

score_mod = import_module("08_build_unified_score")
UnifiedScoreBuilder = score_mod.UnifiedScoreBuilder

def run_evaluate_zero_day():
    print("=" * 80)
    print(">>> STEP 10: 11 - EVALUATE HELD-OUT ZERO-DAY ATTACK (Service_Scan, N=7,302) <<<")
    print("=" * 80)

    ensure_output_dirs()
    cfg = load_step10_config()
    artifacts = load_frozen_artifacts()
    mapping = artifacts["mapping"]
    zero_day_class = mapping["zero_day_class"]
    idx_to_class = {int(k): v for k, v in mapping["idx_to_class"].items()}

    # 1. Load Zero-Day Test Data
    zd_path = get_abs_path(cfg["paths"]["zeroday_test_path"])
    df_zd = pd.read_parquet(zd_path)
    n_zd = len(df_zd)
    print(f"Loaded Zero-Day Test Set: {n_zd:,} flows ({zero_day_class})")

    # Assert exactly 7,302 flows and all Service_Scan
    assert n_zd == 7302, f"Expected 7,302 flows, got {n_zd}"
    assert (df_zd["subcategory"] == zero_day_class).all(), f"Found non-{zero_day_class} traffic in zero-day test!"

    # 2. Compute Signals & Normalization
    print("Computing 4 raw novelty signals for Zero-Day Test...")
    zd_signals = compute_all_novelty_signals(df_zd, artifacts)

    norm_path = os.path.join(STEP10_DIR, "outputs", "calibration", "per_signal_calibration.json")
    normalizer = EmpiricalCDFNormalizer()
    normalizer.load(norm_path)
    Z_zd = normalizer.transform(zd_signals)

    # 3. Compute Unified Score
    opt_weights_path = os.path.join(STEP10_DIR, "outputs", "weights", "optimized_weights.json")
    with open(opt_weights_path, "r", encoding="utf-8") as f:
        weights_info = json.load(f)
    builder = UnifiedScoreBuilder(weights_info["selected_optimal_weights"])
    s_unified_zd = builder.compute(Z_zd)
    zd_signals["S_unified"] = s_unified_zd

    # 4. Load Class-Conditional Thresholds & Known Test Metrics
    thresh_csv_path = os.path.join(STEP10_DIR, "outputs", "thresholds", "class_conditional_thresholds.csv")
    df_thresholds = pd.read_csv(thresh_csv_path).set_index("class_idx")

    kt_metrics_path = os.path.join(STEP10_DIR, "outputs", "known_test", "metrics.json")
    with open(kt_metrics_path, "r", encoding="utf-8") as f:
        known_test_metrics = json.load(f)

    op_path = os.path.join(STEP10_DIR, "outputs", "thresholds", "operating_point_selection.json")
    with open(op_path, "r", encoding="utf-8") as f:
        op_info = json.load(f)
    primary_pct = float(op_info["selected_percentile"])
    primary_label = str(op_info["selected_label"])

    percentiles = cfg["parameters"]["thresholds"]["candidate_percentiles"]
    pred_indices = zd_signals["pred_idx"].values
    results_by_percentile = {}

    print("\n--- Zero-Day Detection Performance Across Percentile Thresholds ---")
    print(f"{'Percentile':<12} | {'Detected':<10} | {'Missed':<10} | {'Recall':<10} | {'Precision':<11} | {'F1 Score':<10} | {'Known Acc':<10}")
    print("-" * 88)

    for p in percentiles:
        col_p = f"P{p:.1f}"
        tau_array = np.array([df_thresholds.loc[c, col_p] for c in pred_indices])
        is_detected = (s_unified_zd > tau_array)

        tp = int(is_detected.sum())
        fn = n_zd - tp
        recall = (tp / float(n_zd)) * 100.0

        # Retrieve FP from known test at this percentile
        fp = known_test_metrics[col_p]["false_unknowns"]
        known_acc = known_test_metrics[col_p]["known_acceptance_rate_pct"]
        benign_rej = known_test_metrics[col_p]["benign_rejection_rate_pct"]

        precision = (tp / float(tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        results_by_percentile[col_p] = {
            "percentile": p,
            "total_zero_day_flows": n_zd,
            "detected_zero_day": tp,
            "missed_zero_day": fn,
            "zero_day_recall_pct": round(recall, 2),
            "false_positives_known_test": fp,
            "unknown_precision_pct": round(precision, 2),
            "unknown_f1_score_pct": round(f1, 2),
            "known_test_acceptance_pct": known_acc,
            "benign_rejection_pct": benign_rej
        }

        print(f"{col_p:<12} | {tp:>6,d}/{n_zd:,} | {fn:>6,d}     | {recall:>7.2f}%   | {precision:>8.2f}%   | {f1:>7.2f}%   | {known_acc:>7.2f}%")

    # 5. Primary Operating Point Decisions & Subgroup Analysis
    tau_prim = np.array([df_thresholds.loc[c, primary_label] for c in pred_indices])
    is_detected_prim = (s_unified_zd > tau_prim)
    zd_signals["threshold_primary"] = tau_prim
    zd_signals["is_detected"] = is_detected_prim
    zd_signals["final_decision"] = np.where(is_detected_prim, "UNKNOWN_ATTACK", zd_signals["pred_class"].values)

    # Subgroup breakdown by predicted known class
    subgroup_records = []
    print(f"\n--- Service_Scan Breakdown by XGBoost Predicted Class ({primary_label} Operating Point) ---")
    print(f"{'Predicted Class':<18} | {'Total Flows':<12} | {'Detected':<10} | {'Missed':<10} | {'Recall':<10} | {'Mean S_unified':<14} | {'Mean Threshold':<14}")
    print("-" * 100)

    for c in range(len(idx_to_class)):
        c_name = idx_to_class[c]
        mask_c = (pred_indices == c)
        n_c = int(mask_c.sum())
        if n_c == 0:
            continue

        tp_c = int(is_detected_prim[mask_c].sum())
        fn_c = n_c - tp_c
        rec_c = (tp_c / float(n_c)) * 100.0

        sub_s_u = zd_signals.loc[mask_c, "S_unified"].values
        sub_tau = zd_signals.loc[mask_c, "threshold_primary"].values
        sub_pmax = zd_signals.loc[mask_c, "p_max"].values
        sub_dm = zd_signals.loc[mask_c, "signal_mahalanobis"].values
        sub_rel = zd_signals.loc[mask_c, "signal_relative"].values
        sub_leaf = zd_signals.loc[mask_c, "signal_leaf"].values

        subgroup_records.append({
            "predicted_class": c_name,
            "total_flows": n_c,
            "detected_flows": tp_c,
            "missed_flows": fn_c,
            "subgroup_recall_pct": round(rec_c, 2),
            "mean_p_max": round(float(np.mean(sub_pmax)), 4),
            "mean_mahalanobis": round(float(np.mean(sub_dm)), 4),
            "mean_relative_distance": round(float(np.mean(sub_rel)), 4),
            "mean_leaf_novelty": round(float(np.mean(sub_leaf)), 4),
            "mean_s_unified": round(float(np.mean(sub_s_u)), 4),
            "mean_threshold": round(float(np.mean(sub_tau)), 4)
        })

        print(f"{c_name:<18} | {n_c:>8,d} flows | {tp_c:>6,d}   | {fn_c:>6,d}   | {rec_c:>7.2f}%  | {np.mean(sub_s_u):>10.4f}     | {np.mean(sub_tau):>10.4f}")

    df_subgroup = pd.DataFrame(subgroup_records)

    # 6. Combined 3-Way Open-Set Confusion Matrix across All Test Traffic (Known Test + Zero-Day Test)
    # Load known test predictions
    kt_pred_path = os.path.join(STEP10_DIR, "outputs", "known_test", "predictions.parquet")
    df_kt_pred = pd.read_parquet(kt_pred_path)

    all_true_cat = np.concatenate([
        np.where(df_kt_pred["true_class"].values == "Normal", "BENIGN", "KNOWN_ATTACK"),
        np.full(len(zd_signals), "ZERO_DAY_ATTACK")
    ])
    all_pred_cat = np.concatenate([
        np.where(df_kt_pred["is_unknown"].values, "UNKNOWN_ATTACK", np.where(df_kt_pred["pred_class"].values == "Normal", "BENIGN", "KNOWN_ATTACK")),
        np.where(zd_signals["is_detected"].values, "UNKNOWN_ATTACK", np.where(zd_signals["pred_class"].values == "Normal", "BENIGN", "KNOWN_ATTACK"))
    ])

    cm_combined = confusion_matrix(
        all_true_cat,
        all_pred_cat,
        labels=["BENIGN", "KNOWN_ATTACK", "ZERO_DAY_ATTACK"]
    )
    df_cm_combined = pd.DataFrame(
        cm_combined,
        index=["TRUE_BENIGN", "TRUE_KNOWN_ATTACK", "TRUE_ZERO_DAY_ATTACK"],
        columns=["PRED_BENIGN", "PRED_KNOWN_ATTACK", "PRED_UNKNOWN_ATTACK"]
    )
    print("\n3-Way Master Open-Set Confusion Matrix (Known Test + Zero-Day Test):")
    print(df_cm_combined.to_string())

    # 7. Master Baseline Comparison Table (Step 6, Step 8, Step 9, Old Step 10, NEW Step 10)
    prim_zd = results_by_percentile[primary_label]
    os_rec_new = 0.0
    for s_rec in subgroup_records:
        if s_rec["predicted_class"] == "OS_Fingerprint":
            os_rec_new = s_rec["subgroup_recall_pct"]
            break

    comp_records = [
        {
            "Method": "Step 6 Conf + Mahalanobis OR",
            "Weights": "OR Fusion [1.0, 1.0, 0, 0]",
            "Threshold": "Class-Cond P95",
            "Zero-Day Recall (%)": 41.22,
            "Unknown Precision (%)": 52.99,
            "Unknown F1 (%)": 46.37,
            "Known Acceptance (%)": 95.06,
            "Benign Rejection (%)": 2.82,
            "OS_Fingerprint subgroup recall (%)": 0.12
        },
        {
            "Method": "Step 8 Relative Distance OR",
            "Weights": "OR Fusion [1.0, 0, 0, 1.0]",
            "Threshold": "Class-Cond P95",
            "Zero-Day Recall (%)": 45.77,
            "Unknown Precision (%)": 52.32,
            "Unknown F1 (%)": 48.82,
            "Known Acceptance (%)": 94.36,
            "Benign Rejection (%)": 9.86,
            "OS_Fingerprint subgroup recall (%)": 6.28
        },
        {
            "Method": "Step 9 AUC Weighted P95",
            "Weights": "[0.7353, 0.2099, 0.0000, 0.0548]",
            "Threshold": "Class-Cond P95",
            "Zero-Day Recall (%)": 32.94,
            "Unknown Precision (%)": 46.71,
            "Unknown F1 (%)": 38.63,
            "Known Acceptance (%)": 94.92,
            "Benign Rejection (%)": 1.41,
            "OS_Fingerprint subgroup recall (%)": 0.00
        },
        {
            "Method": "Step 9 AUC Weighted P90",
            "Weights": "[0.7353, 0.2099, 0.0000, 0.0548]",
            "Threshold": "Class-Cond P90",
            "Zero-Day Recall (%)": 47.21,
            "Unknown Precision (%)": 39.18,
            "Unknown F1 (%)": 42.82,
            "Known Acceptance (%)": 90.10,
            "Benign Rejection (%)": 5.63,
            "OS_Fingerprint subgroup recall (%)": 14.50
        },
        {
            "Method": "Old Step 10 (Conf-Only Collapse)",
            "Weights": "[1.0000, 0.0000, 0.0000, 0.0000]",
            "Threshold": "Class-Cond P95",
            "Zero-Day Recall (%)": 32.92,
            "Unknown Precision (%)": 47.07,
            "Unknown F1 (%)": 38.74,
            "Known Acceptance (%)": 95.00,
            "Benign Rejection (%)": 2.82,
            "OS_Fingerprint subgroup recall (%)": 0.00
        },
        {
            "Method": f"NEW Step 10 Learned Four-Signal",
            "Weights": f"[{builder.w[0]:.4f}, {builder.w[1]:.4f}, {builder.w[2]:.4f}, {builder.w[3]:.4f}]",
            "Threshold": f"Class-Cond {primary_label}",
            "Zero-Day Recall (%)": prim_zd["zero_day_recall_pct"],
            "Unknown Precision (%)": prim_zd["unknown_precision_pct"],
            "Unknown F1 (%)": prim_zd["unknown_f1_score_pct"],
            "Known Acceptance (%)": prim_zd["known_test_acceptance_pct"],
            "Benign Rejection (%)": prim_zd["benign_rejection_pct"],
            "OS_Fingerprint subgroup recall (%)": os_rec_new
        }
    ]
    df_comparison = pd.DataFrame(comp_records)

    print("\n--- Master Comparison Against Historical Baselines ---")
    print(df_comparison.to_string(index=False))

    # 8. Save Artifacts
    out_dir = os.path.join(STEP10_DIR, "outputs", "zero_day")
    zd_signals.to_parquet(os.path.join(out_dir, "predictions.parquet"), index=False)
    zd_signals.to_csv(os.path.join(out_dir, "predictions_sample.csv"), index=False)

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results_by_percentile, f, indent=2)

    df_subgroup.to_csv(os.path.join(out_dir, "subgroup_analysis.csv"), index=False)
    df_comparison.to_csv(os.path.join(out_dir, "master_comparison.csv"), index=False)
    df_cm_combined.to_csv(os.path.join(out_dir, "combined_confusion_matrix.csv"))

    print(f"\nSaved zero-day predictions, metrics, subgroup analysis, and master comparison to: {out_dir}")
    return results_by_percentile, df_subgroup, df_comparison

if __name__ == "__main__":
    run_evaluate_zero_day()
