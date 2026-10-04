"""
09_evaluate_known_test.py: Unbiased evaluation of known-traffic acceptance & benign FPR
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from common_utils import load_step9_config, get_abs_path, load_frozen_artifacts
from importlib import import_module

compute_signals_mod = import_module("02_compute_novelty_signals")
compute_all_novelty_signals = compute_signals_mod.compute_all_novelty_signals

norm_mod = import_module("03_percentile_normalization")
PercentileNormalizer = norm_mod.PercentileNormalizer

score_mod = import_module("07_build_unified_score")
UnifiedScoreBuilder = score_mod.UnifiedScoreBuilder

def run_evaluate_known_test():
    print("=" * 80)
    print(">>> STEP 9: 09 - EVALUATE KNOWN TEST TRAFFIC <<<")
    print("=" * 80)

    step9_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_step9_config()
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
    print("Computing novelty signals for Known Test...")
    test_signals = compute_all_novelty_signals(df_test, artifacts)

    norm_path = os.path.join(step9_dir, "outputs", "calibration", "percentile_reference.json")
    normalizer = PercentileNormalizer()
    normalizer.load(norm_path)
    Z_test = normalizer.transform(test_signals)

    # 3. Compute Unified Score
    auc_weights_path = os.path.join(step9_dir, "outputs", "weights", "auc_weights.json")
    with open(auc_weights_path, "r", encoding="utf-8") as f:
        w_data = json.load(f)["weights"]
    builder = UnifiedScoreBuilder(w_data)
    s_unified_test = builder.compute(Z_test)
    test_signals["S_unified"] = s_unified_test

    # 4. Load Class-Conditional Thresholds
    thresh_csv_path = os.path.join(step9_dir, "outputs", "thresholds", "class_conditional_thresholds.csv")
    df_thresholds = pd.read_csv(thresh_csv_path).set_index("class_idx")

    percentiles = cfg["parameters"]["thresholds"]["candidate_percentiles"]
    primary_pct = cfg["parameters"]["thresholds"]["primary_percentile"]

    # 5. Evaluate Rejection Decisions across Percentiles
    pred_indices = test_signals["pred_idx"].values
    true_labels = test_signals["true_class"].values
    is_normal = (true_labels == "Normal")
    n_normal = is_normal.sum()

    results_by_percentile = {}

    for p in percentiles:
        col_p = f"P{p:.1f}"
        tau_array = np.array([df_thresholds.loc[c, col_p] for c in pred_indices])
        is_unknown = (s_unified_test > tau_array)

        fp_count = int(is_unknown.sum())
        acc_rate = (1.0 - (fp_count / n_test)) * 100.0
        fur_rate = (fp_count / n_test) * 100.0

        fp_normal = int(is_unknown[is_normal].sum())
        benign_rej_rate = (fp_normal / max(1, n_normal)) * 100.0

        # Class-wise acceptance
        class_acc = {}
        for c_idx, c_name in idx_to_class.items():
            mask_c = (true_labels == c_name)
            n_c = mask_c.sum()
            rej_c = int(is_unknown[mask_c].sum())
            acc_c = (1.0 - (rej_c / max(1, n_c))) * 100.0
            class_acc[c_name] = {
                "support": int(n_c),
                "rejected_as_unknown": rej_c,
                "acceptance_rate_pct": round(acc_c, 2)
            }

        results_by_percentile[col_p] = {
            "percentile": p,
            "total_known_flows": n_test,
            "false_unknowns": fp_count,
            "known_acceptance_rate_pct": round(acc_rate, 2),
            "false_unknown_rate_pct": round(fur_rate, 2),
            "normal_benign_flows": int(n_normal),
            "normal_false_alarms": fp_normal,
            "benign_rejection_rate_pct": round(benign_rej_rate, 2),
            "class_wise_acceptance": class_acc
        }

    # 6. Primary Operating Point (95.0%) Summary & Confusion Matrix
    prim_col = f"P{primary_pct:.1f}"
    prim_metrics = results_by_percentile[prim_col]
    tau_prim = np.array([df_thresholds.loc[c, prim_col] for c in pred_indices])
    is_unknown_prim = (s_unified_test > tau_prim)
    test_signals["is_unknown"] = is_unknown_prim
    test_signals["final_decision"] = np.where(is_unknown_prim, "UNKNOWN_ATTACK", test_signals["pred_class"].values)

    # 3-Way Open-Set Categorization for Known Test
    # True Categories: BENIGN (Normal), KNOWN_ATTACK
    # Predicted Categories: BENIGN, KNOWN_ATTACK, UNKNOWN_ATTACK
    cat_true = np.where(true_labels == "Normal", "BENIGN", "KNOWN_ATTACK")
    cat_pred = np.where(is_unknown_prim, "UNKNOWN_ATTACK", np.where(test_signals["pred_class"] == "Normal", "BENIGN", "KNOWN_ATTACK"))
    cm = confusion_matrix(cat_true, cat_pred, labels=["BENIGN", "KNOWN_ATTACK", "UNKNOWN_ATTACK"])
    df_cm = pd.DataFrame(cm, index=["TRUE_BENIGN", "TRUE_KNOWN_ATTACK", "TRUE_UNKNOWN_ATTACK"], columns=["PRED_BENIGN", "PRED_KNOWN_ATTACK", "PRED_UNKNOWN_ATTACK"])

    print(f"\n--- Known Test Evaluation Summary (Primary Operating Point: P{primary_pct:.1f}) ---")
    print(f"  Known-Test Acceptance Rate:  {prim_metrics['known_acceptance_rate_pct']:.2f}% ({n_test - prim_metrics['false_unknowns']:,} / {n_test:,} flows accepted)")
    print(f"  False Unknown Rate:          {prim_metrics['false_unknown_rate_pct']:.2f}% ({prim_metrics['false_unknowns']:,} false unknowns)")
    print(f"  Benign ('Normal') Rejection: {prim_metrics['benign_rejection_rate_pct']:.2f}% ({prim_metrics['normal_false_alarms']} / {n_normal} benign flows rejected)")

    print("\nClass-Wise Known Acceptance Breakdown:")
    for c_name, c_dict in prim_metrics["class_wise_acceptance"].items():
        print(f"  {c_name:<18}: {c_dict['acceptance_rate_pct']:>6.2f}% accepted ({c_dict['rejected_as_unknown']:>4,d} / {c_dict['support']:>5,d} rejected)")

    print("\n3-Way Open-Set Confusion Matrix (Known Test Traffic):")
    print(df_cm.to_string())

    # 7. Save Output Files
    out_dir = os.path.join(step9_dir, "outputs", "known_test")
    # Save sampled predictions CSV (first 5,000 for space) plus full parquet
    test_signals.to_parquet(os.path.join(out_dir, "predictions.parquet"), index=False)
    test_signals.head(5000).to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results_by_percentile, f, indent=2)

    df_cm.to_csv(os.path.join(out_dir, "confusion_matrix.csv"))
    print(f"\nSaved known-test predictions, metrics, and confusion matrix to: {out_dir}")

    return results_by_percentile

if __name__ == "__main__":
    run_evaluate_known_test()
