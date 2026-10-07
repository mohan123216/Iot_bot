"""
10_evaluate_known_test.py: Unbiased evaluation on known test traffic
Step 10: Optimized Multi-Signal Novelty Fusion
Strict Rule: Evaluates known traffic (54,043 flows). Zero-day attack is quarantined.
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

def run_evaluate_known_test():
    print("=" * 80)
    print(">>> STEP 10: 10 - EVALUATE KNOWN TEST TRAFFIC (UNBIASED BENCHMARK) <<<")
    print("=" * 80)

    ensure_output_dirs()
    cfg = load_step10_config()
    artifacts = load_frozen_artifacts()
    mapping = artifacts["mapping"]
    zero_day_class = mapping["zero_day_class"]
    idx_to_class = {int(k): v for k, v in mapping["idx_to_class"].items()}

    # 1. Load Known Test Data
    test_path = get_abs_path(cfg["paths"]["known_test_path"])
    df_test = pd.read_parquet(test_path)
    n_test = len(df_test)
    print(f"Loaded Known Test Set: {n_test:,} flows")

    # Assert Service_Scan is strictly absent
    zd_count = int((df_test["subcategory"] == zero_day_class).sum())
    assert zd_count == 0, f"FATAL LEAKAGE: {zero_day_class} found in known_test!"

    # 2. Compute Signals & Normalization
    print("Computing 4 raw novelty signals for Known Test...")
    test_signals = compute_all_novelty_signals(df_test, artifacts)

    norm_path = os.path.join(STEP10_DIR, "outputs", "calibration", "per_signal_calibration.json")
    normalizer = EmpiricalCDFNormalizer()
    normalizer.load(norm_path)
    Z_test = normalizer.transform(test_signals)

    # 3. Compute Unified Score using Learned Optimal Weights
    opt_weights_path = os.path.join(STEP10_DIR, "outputs", "weights", "optimized_weights.json")
    with open(opt_weights_path, "r", encoding="utf-8") as f:
        weights_info = json.load(f)
    builder = UnifiedScoreBuilder(weights_info["selected_optimal_weights"])
    s_unified_test = builder.compute(Z_test)
    test_signals["S_unified"] = s_unified_test

    # 4. Load Class-Conditional Thresholds & Selected Operating Point
    thresh_csv_path = os.path.join(STEP10_DIR, "outputs", "thresholds", "class_conditional_thresholds.csv")
    df_thresholds = pd.read_csv(thresh_csv_path).set_index("class_idx")

    op_path = os.path.join(STEP10_DIR, "outputs", "thresholds", "operating_point_selection.json")
    with open(op_path, "r", encoding="utf-8") as f:
        op_info = json.load(f)
    primary_pct = float(op_info["selected_percentile"])
    primary_label = str(op_info["selected_label"])

    percentiles = cfg["parameters"]["thresholds"]["candidate_percentiles"]

    # 5. Evaluate Rejection Decisions across Percentiles
    pred_indices = test_signals["pred_idx"].values
    true_labels = test_signals["true_class"].values
    is_normal = (true_labels == "Normal")
    n_normal = int(is_normal.sum())
    n_known_attack = n_test - n_normal

    results_by_percentile = {}

    for p in percentiles:
        col_p = f"P{p:.1f}"
        tau_array = np.array([df_thresholds.loc[c, col_p] for c in pred_indices])
        is_unknown = (s_unified_test > tau_array)

        fp_count = int(is_unknown.sum())
        acc_rate = (1.0 - (fp_count / float(n_test))) * 100.0
        fur_rate = (fp_count / float(n_test)) * 100.0

        fp_normal = int(is_unknown[is_normal].sum())
        benign_rej_rate = (fp_normal / float(n_normal)) * 100.0 if n_normal > 0 else 0.0

        fp_known_attack = int(is_unknown[~is_normal].sum())
        known_attack_acc_rate = (1.0 - (fp_known_attack / float(n_known_attack))) * 100.0 if n_known_attack > 0 else 0.0

        # Class-wise acceptance
        class_acc = {}
        for c_idx, c_name in idx_to_class.items():
            mask_c = (true_labels == c_name)
            n_c = int(mask_c.sum())
            rej_c = int(is_unknown[mask_c].sum())
            acc_c = (1.0 - (rej_c / float(max(1, n_c)))) * 100.0
            class_acc[c_name] = {
                "support": n_c,
                "rejected_as_unknown": rej_c,
                "accepted_as_known": n_c - rej_c,
                "acceptance_rate_pct": round(acc_c, 2)
            }

        results_by_percentile[col_p] = {
            "percentile": p,
            "total_known_flows": n_test,
            "false_unknowns": fp_count,
            "known_acceptance_rate_pct": round(acc_rate, 2),
            "false_unknown_rate_pct": round(fur_rate, 2),
            "normal_benign_flows": n_normal,
            "normal_false_alarms": fp_normal,
            "benign_rejection_rate_pct": round(benign_rej_rate, 2),
            "known_attack_flows": n_known_attack,
            "known_attack_rejections": fp_known_attack,
            "known_attack_acceptance_rate_pct": round(known_attack_acc_rate, 2),
            "class_wise_acceptance": class_acc
        }

    # 6. Primary Operating Point Evaluation
    prim_metrics = results_by_percentile[primary_label]
    tau_prim = np.array([df_thresholds.loc[c, primary_label] for c in pred_indices])
    is_unknown_prim = (s_unified_test > tau_prim)
    test_signals["is_unknown"] = is_unknown_prim
    test_signals["threshold_primary"] = tau_prim
    test_signals["final_decision"] = np.where(is_unknown_prim, "UNKNOWN_ATTACK", test_signals["pred_class"].values)

    # Classification Accuracy among accepted known attacks
    accepted_attacks_mask = (~is_normal) & (~is_unknown_prim)
    correct_pred_attacks = (test_signals.loc[accepted_attacks_mask, "pred_class"].values == test_signals.loc[accepted_attacks_mask, "true_class"].values)
    attack_acc_pct = (correct_pred_attacks.sum() / float(accepted_attacks_mask.sum()) * 100.0) if accepted_attacks_mask.sum() > 0 else 0.0

    prim_metrics["accepted_known_attack_classification_accuracy_pct"] = round(float(attack_acc_pct), 2)

    # 3-Way Confusion Matrix
    cat_true = np.where(true_labels == "Normal", "BENIGN", "KNOWN_ATTACK")
    cat_pred = np.where(is_unknown_prim, "UNKNOWN_ATTACK", np.where(test_signals["pred_class"] == "Normal", "BENIGN", "KNOWN_ATTACK"))
    cm = confusion_matrix(cat_true, cat_pred, labels=["BENIGN", "KNOWN_ATTACK", "UNKNOWN_ATTACK"])
    df_cm = pd.DataFrame(cm, index=["TRUE_BENIGN", "TRUE_KNOWN_ATTACK", "TRUE_UNKNOWN_ATTACK"], columns=["PRED_BENIGN", "PRED_KNOWN_ATTACK", "PRED_UNKNOWN_ATTACK"])

    # Class Breakdown Table for Primary Point
    breakdown_rows = []
    for c_name, c_data in prim_metrics["class_wise_acceptance"].items():
        breakdown_rows.append({
            "Class": c_name,
            "Total Flows": c_data["support"],
            "Accepted as Known": c_data["accepted_as_known"],
            "Rejected as Unknown": c_data["rejected_as_unknown"],
            "Acceptance Rate (%)": c_data["acceptance_rate_pct"]
        })
    df_breakdown = pd.DataFrame(breakdown_rows)

    print(f"\n--- Known Test Evaluation Summary (Primary Operating Point: {primary_label}) ---")
    print(f"  Overall Known Acceptance:       {prim_metrics['known_acceptance_rate_pct']:.2f}% ({n_test - prim_metrics['false_unknowns']:,} / {n_test:,} flows accepted)")
    print(f"  False Unknown Rate (FUR):       {prim_metrics['false_unknown_rate_pct']:.2f}% ({prim_metrics['false_unknowns']:,} false alarms)")
    print(f"  Benign ('Normal') Rejection:    {prim_metrics['benign_rejection_rate_pct']:.2f}% ({prim_metrics['normal_false_alarms']:,} / {n_normal:,} benign flows rejected)")
    print(f"  Known Attack Acceptance Rate:   {prim_metrics['known_attack_acceptance_rate_pct']:.2f}% ({n_known_attack - prim_metrics['known_attack_rejections']:,} / {n_known_attack:,} flows accepted)")
    print(f"  Accepted Attack Class Accuracy: {attack_acc_pct:.2f}%")

    print("\nClass-Wise Known Acceptance Breakdown:")
    print(df_breakdown.to_string(index=False))

    print("\n3-Way Open-Set Confusion Matrix (Known Test Traffic):")
    print(df_cm.to_string())

    # 7. Save Artifacts
    out_dir = os.path.join(STEP10_DIR, "outputs", "known_test")
    test_signals.to_parquet(os.path.join(out_dir, "predictions.parquet"), index=False)
    test_signals.head(5000).to_csv(os.path.join(out_dir, "predictions_sample.csv"), index=False)

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results_by_percentile, f, indent=2)

    df_cm.to_csv(os.path.join(out_dir, "confusion_matrix.csv"))
    df_breakdown.to_csv(os.path.join(out_dir, "class_breakdown.csv"), index=False)

    print(f"\nSaved known test predictions, metrics, confusion matrix to: {out_dir}")
    return results_by_percentile, df_cm

if __name__ == "__main__":
    run_evaluate_known_test()
