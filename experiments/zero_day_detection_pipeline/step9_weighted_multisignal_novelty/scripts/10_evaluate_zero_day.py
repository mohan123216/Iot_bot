"""
10_evaluate_zero_day.py: Unbiased evaluation of Service_Scan zero-day detection
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from common_utils import load_step9_config, get_abs_path, load_frozen_artifacts
from importlib import import_module

compute_signals_mod = import_module("02_compute_novelty_signals")
compute_all_novelty_signals = compute_signals_mod.compute_all_novelty_signals

norm_mod = import_module("03_percentile_normalization")
PercentileNormalizer = norm_mod.PercentileNormalizer

score_mod = import_module("07_build_unified_score")
UnifiedScoreBuilder = score_mod.UnifiedScoreBuilder

def run_evaluate_zero_day():
    print("=" * 80)
    print(">>> STEP 9: 10 - EVALUATE HELD-OUT ZERO-DAY ATTACK (Service_Scan) <<<")
    print("=" * 80)

    step9_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_step9_config()
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
    assert (df_zd["subcategory"] == zero_day_class).all(), "Zero-day test contains non-Service_Scan flows!"

    # 2. Compute Signals & Normalization
    print("Computing novelty signals for Zero-Day Test...")
    zd_signals = compute_all_novelty_signals(df_zd, artifacts)

    norm_path = os.path.join(step9_dir, "outputs", "calibration", "percentile_reference.json")
    normalizer = PercentileNormalizer()
    normalizer.load(norm_path)
    Z_zd = normalizer.transform(zd_signals)

    # 3. Compute Unified Score
    auc_weights_path = os.path.join(step9_dir, "outputs", "weights", "auc_weights.json")
    with open(auc_weights_path, "r", encoding="utf-8") as f:
        w_data = json.load(f)["weights"]
    builder = UnifiedScoreBuilder(w_data)
    s_unified_zd = builder.compute(Z_zd)
    zd_signals["S_unified"] = s_unified_zd

    # 4. Load Class-Conditional Thresholds & Known Test Metrics
    thresh_csv_path = os.path.join(step9_dir, "outputs", "thresholds", "class_conditional_thresholds.csv")
    df_thresholds = pd.read_csv(thresh_csv_path).set_index("class_idx")

    kt_metrics_path = os.path.join(step9_dir, "outputs", "known_test", "metrics.json")
    with open(kt_metrics_path, "r", encoding="utf-8") as f:
        known_test_metrics = json.load(f)

    percentiles = cfg["parameters"]["thresholds"]["candidate_percentiles"]
    primary_pct = cfg["parameters"]["thresholds"]["primary_percentile"]

    # 5. Evaluate Rejection Decisions across Percentiles
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
        recall = (tp / n_zd) * 100.0

        # Retrieve FP from known test at this percentile
        fp = known_test_metrics[col_p]["false_unknowns"]
        known_acc = known_test_metrics[col_p]["known_acceptance_rate_pct"]

        precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
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
            "known_test_acceptance_pct": known_acc
        }

        print(f"{col_p:<12} | {tp:>6,d}/{n_zd:,} | {fn:>6,d}     | {recall:>7.2f}%   | {precision:>8.2f}%   | {f1:>7.2f}%   | {known_acc:>7.2f}%")

    # 6. Primary Operating Point (95.0%) Decisions & Subgroup Analysis
    prim_col = f"P{primary_pct:.1f}"
    tau_prim = np.array([df_thresholds.loc[c, prim_col] for c in pred_indices])
    is_detected_prim = (s_unified_zd > tau_prim)
    zd_signals["threshold_P95"] = tau_prim
    zd_signals["is_detected"] = is_detected_prim
    zd_signals["final_decision"] = np.where(is_detected_prim, "UNKNOWN_ATTACK", zd_signals["pred_class"].values)

    # Subgroup breakdown by predicted known class
    subgroup_records = []
    print("\n--- Service_Scan Breakdown by XGBoost Predicted Class (P95 Operating Point) ---")
    print(f"{'Predicted Class':<18} | {'Total Flows':<12} | {'Detected':<10} | {'Missed':<10} | {'Recall':<10} | {'Mean S_unified':<14} | {'Mean tau_S':<10}")
    print("-" * 95)

    for c in range(len(idx_to_class)):
        c_name = idx_to_class[c]
        mask_c = (pred_indices == c)
        n_c = mask_c.sum()
        if n_c == 0:
            continue

        tp_c = int(is_detected_prim[mask_c].sum())
        fn_c = n_c - tp_c
        rec_c = (tp_c / n_c) * 100.0

        sub_s_u = zd_signals.loc[mask_c, "S_unified"].values
        sub_tau = zd_signals.loc[mask_c, "threshold_P95"].values
        sub_pmax = zd_signals.loc[mask_c, "p_max"].values
        sub_dm = zd_signals.loc[mask_c, "signal_mahalanobis"].values
        sub_rel = zd_signals.loc[mask_c, "signal_relative"].values
        sub_leaf = zd_signals.loc[mask_c, "signal_leaf"].values

        subgroup_records.append({
            "predicted_class": c_name,
            "total_flows": int(n_c),
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

        print(f"{c_name:<18} | {n_c:>8,d} flows | {tp_c:>6,d}   | {fn_c:>6,d}   | {rec_c:>7.2f}%  | {np.mean(sub_s_u):>10.4f}     | {np.mean(sub_tau):>8.4f}")

    df_subgroup = pd.DataFrame(subgroup_records)

    # 7. Comparison with Historical Step 6 and Step 8 Baselines
    prim_zd = results_by_percentile[prim_col]
    comp_records = [
        {
            "Method": "Confidence + Mahalanobis OR (Step 6)",
            "Zero-Day Recall (%)": 41.22,
            "Unknown Precision (%)": 52.99,
            "Unknown F1 (%)": 46.37,
            "Known Acceptance (%)": 95.06,
            "Benign Rejection (%)": 2.82,
            "Detected ZD Flows": 3010,
            "Decision Rule": "P < 0.99 OR M_norm > 1.0"
        },
        {
            "Method": "Mahalanobis + Leaf OR (Step 6)",
            "Zero-Day Recall (%)": 36.62,
            "Unknown Precision (%)": 35.67,
            "Unknown F1 (%)": 36.14,
            "Known Acceptance (%)": 91.08,
            "Benign Rejection (%)": 7.04,
            "Detected ZD Flows": 2674,
            "Decision Rule": "M_norm > 1.0 OR L_norm > 1.0"
        },
        {
            "Method": "Conf + Mah + RelDist (Step 8)",
            "Zero-Day Recall (%)": 45.77,
            "Unknown Precision (%)": 52.32,
            "Unknown F1 (%)": 48.82,
            "Known Acceptance (%)": 94.36,
            "Benign Rejection (%)": 9.86,
            "Detected ZD Flows": 3342,
            "Decision Rule": "P < 0.99 OR M_norm > 1 OR Rel_norm > 1"
        },
        {
            "Method": f"New Weighted Unified Score (P{primary_pct:.1f})",
            "Zero-Day Recall (%)": prim_zd["zero_day_recall_pct"],
            "Unknown Precision (%)": prim_zd["unknown_precision_pct"],
            "Unknown F1 (%)": prim_zd["unknown_f1_score_pct"],
            "Known Acceptance (%)": prim_zd["known_test_acceptance_pct"],
            "Benign Rejection (%)": known_test_metrics[prim_col]["benign_rejection_rate_pct"],
            "Detected ZD Flows": prim_zd["detected_zero_day"],
            "Decision Rule": "S_unified > tau_S(c) [Adaptive]"
        }
    ]
    df_comparison = pd.DataFrame(comp_records)

    print("\n--- Master Comparison Against Historical Baselines ---")
    print(df_comparison.to_string(index=False))

    # 8. Save Output Files
    out_dir = os.path.join(step9_dir, "outputs", "zero_day")
    zd_signals.to_parquet(os.path.join(out_dir, "predictions.parquet"), index=False)
    zd_signals.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results_by_percentile, f, indent=2)

    df_subgroup.to_csv(os.path.join(out_dir, "subgroup_analysis.csv"), index=False)
    df_comparison.to_csv(os.path.join(out_dir, "baseline_comparison.csv"), index=False)
    print(f"\nSaved zero-day predictions, metrics, subgroup analysis, and baseline comparison to: {out_dir}")

    return results_by_percentile, df_subgroup, df_comparison

if __name__ == "__main__":
    run_evaluate_zero_day()
