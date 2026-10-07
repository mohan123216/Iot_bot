"""
12_ablation_study.py: Comprehensive ablation study across individual signals & fusion combinations
Step 10: Optimized Multi-Signal Novelty Fusion
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from common_utils import load_step10_config, get_abs_path, load_frozen_artifacts, ensure_output_dirs, STEP10_DIR
from importlib import import_module

score_mod = import_module("08_build_unified_score")
UnifiedScoreBuilder = score_mod.UnifiedScoreBuilder

def run_ablation_study():
    print("=" * 80)
    print(">>> STEP 10: 12 - COMPREHENSIVE ABLATION STUDY <<<")
    print("=" * 80)

    ensure_output_dirs()
    cfg = load_step10_config()
    artifacts = load_frozen_artifacts()
    mapping = artifacts["mapping"]
    idx_to_class = {int(k): v for k, v in mapping["idx_to_class"].items()}
    num_classes = len(idx_to_class)

    # 1. Load Data
    val_calib_path = os.path.join(STEP10_DIR, "outputs", "calibration", "calibrated_validation_signals.parquet")
    df_val = pd.read_parquet(val_calib_path)
    val_pred_idx = df_val["pred_idx"].values
    val_is_normal = (df_val["true_class"].values == "Normal")
    n_val = len(df_val)
    n_normal_val = int(val_is_normal.sum())

    kt_pred_path = os.path.join(STEP10_DIR, "outputs", "known_test", "predictions.parquet")
    df_kt = pd.read_parquet(kt_pred_path)
    kt_pred_idx = df_kt["pred_idx"].values
    kt_is_normal = (df_kt["true_class"].values == "Normal")
    n_kt = len(df_kt)
    n_normal_kt = int(kt_is_normal.sum())

    zd_pred_path = os.path.join(STEP10_DIR, "outputs", "zero_day", "predictions.parquet")
    df_zd = pd.read_parquet(zd_pred_path)
    zd_pred_idx = df_zd["pred_idx"].values
    n_zd = len(df_zd)

    # Load Normalizer to get calibrated Z for test sets
    norm_path = os.path.join(STEP10_DIR, "outputs", "calibration", "per_signal_calibration.json")
    norm_mod = import_module("03_calibrate_signals")
    normalizer = norm_mod.EmpiricalCDFNormalizer()
    normalizer.load(norm_path)

    sig_cols = ["Z_confidence", "Z_mahalanobis", "Z_leaf", "Z_relative"]
    Z_val = df_val[sig_cols]
    Z_kt = normalizer.transform(df_kt)
    Z_zd = normalizer.transform(df_zd)

    # Load Weights
    opt_weights_path = os.path.join(STEP10_DIR, "outputs", "weights", "optimized_weights.json")
    with open(opt_weights_path, "r", encoding="utf-8") as f:
        weights_info = json.load(f)
    w_opt = weights_info["selected_optimal_weights"]
    w_step9 = weights_info["step9_auc_weights"]

    class_weights_path = os.path.join(STEP10_DIR, "outputs", "weights", "class_conditional_weights.json")
    with open(class_weights_path, "r", encoding="utf-8") as f:
        class_weights_dict = json.load(f)

    # Operating percentile for consistent comparison
    op_path = os.path.join(STEP10_DIR, "outputs", "thresholds", "operating_point_selection.json")
    with open(op_path, "r", encoding="utf-8") as f:
        op_info = json.load(f)
    p_op = float(op_info["selected_percentile"])
    p_label = str(op_info["selected_label"])

    ablation_definitions = [
        ("A. Confidence Only", [1.0, 0.0, 0.0, 0.0], False),
        ("B. Mahalanobis Only", [0.0, 1.0, 0.0, 0.0], False),
        ("C. Leaf Only", [0.0, 0.0, 1.0, 0.0], False),
        ("D. Relative Distance Only", [0.0, 0.0, 0.0, 1.0], False),
        ("E. Conf + Mahalanobis", [0.5, 0.5, 0.0, 0.0], False),
        ("F. Conf + Mah + RelDist", [1.0/3, 1.0/3, 0.0, 1.0/3], False),
        ("G. Equal Four-Signal Fusion", [0.25, 0.25, 0.25, 0.25], False),
        ("H. NEW Learned Four-Signal", [w_opt["w_confidence"], w_opt["w_mahalanobis"], w_opt["w_leaf"], w_opt["w_relative"]], False),
        ("I. Step 9 Historical AUC Weights", [w_step9["w_confidence"], w_step9["w_mahalanobis"], w_step9["w_leaf"], w_step9["w_relative"]], False),
        ("J. Class-Conditional Learned Weights", class_weights_dict, True)
    ]

    ablation_results = []
    print(f"\nEvaluating All Ablations at Operating Point {p_label} (Class-Conditional Thresholds)...")
    print(f"{'Variant':<32} | {'ZD Recall':<10} | {'Precision':<10} | {'F1 Score':<10} | {'Known Acc':<10} | {'Benign Rej':<10}")
    print("-" * 92)

    for name, w_input, is_cc in ablation_definitions:
        builder = UnifiedScoreBuilder(w_input, is_class_conditional=is_cc)

        # 1. Compute scores
        s_val = builder.compute(Z_val, val_pred_idx)
        s_kt = builder.compute(Z_kt, kt_pred_idx)
        s_zd = builder.compute(Z_zd, zd_pred_idx)

        # 2. Calibrate class-conditional thresholds on validation score at p_op
        taus = np.zeros(num_classes, dtype=np.float64)
        for c in range(num_classes):
            mask_c = (val_pred_idx == c)
            taus[c] = np.percentile(s_val[mask_c], p_op) if np.any(mask_c) else np.percentile(s_val, p_op)

        # 3. Known Test Rejection
        tau_kt = taus[kt_pred_idx]
        is_unknown_kt = (s_kt > tau_kt)
        fp_kt = int(is_unknown_kt.sum())
        known_acc = (1.0 - (fp_kt / float(n_kt))) * 100.0
        benign_rej = (is_unknown_kt[kt_is_normal].sum() / float(n_normal_kt)) * 100.0

        # 4. Zero-Day Detection
        tau_zd = taus[zd_pred_idx]
        is_detected_zd = (s_zd > tau_zd)
        tp_zd = int(is_detected_zd.sum())
        fn_zd = n_zd - tp_zd
        zd_recall = (tp_zd / float(n_zd)) * 100.0

        # Precision & F1
        prec_zd = (tp_zd / float(tp_zd + fp_kt) * 100.0) if (tp_zd + fp_kt) > 0 else 0.0
        f1_zd = (2.0 * prec_zd * zd_recall / (prec_zd + zd_recall)) if (prec_zd + zd_recall) > 0 else 0.0

        # Extract weights representation
        if is_cc:
            w_c_str = "Class-Conditional"
            w_m_str = "Class-Conditional"
            w_l_str = "Class-Conditional"
            w_r_str = "Class-Conditional"
        else:
            w_vec = builder.w
            w_c_str = f"{w_vec[0]:.4f}"
            w_m_str = f"{w_vec[1]:.4f}"
            w_l_str = f"{w_vec[2]:.4f}"
            w_r_str = f"{w_vec[3]:.4f}"

        ablation_results.append({
            "Variant": name,
            "wC": w_c_str,
            "wM": w_m_str,
            "wL": w_l_str,
            "wR": w_r_str,
            "Zero-Day Recall (%)": round(zd_recall, 2),
            "Detected ZD Flows": tp_zd,
            "Missed ZD Flows": fn_zd,
            "False Positives Known Test": fp_kt,
            "Unknown Precision (%)": round(prec_zd, 2),
            "Unknown F1 (%)": round(f1_zd, 2),
            "Known Acceptance (%)": round(known_acc, 2),
            "Benign Rejection (%)": round(benign_rej, 2)
        })

        print(f"{name:<32} | {zd_recall:>7.2f}%   | {prec_zd:>7.2f}%   | {f1_zd:>7.2f}%   | {known_acc:>7.2f}%   | {benign_rej:>7.2f}%")

    df_ablation = pd.DataFrame(ablation_results)
    out_csv_path = os.path.join(STEP10_DIR, "outputs", "ablation", "ablation_results.csv")
    df_ablation.to_csv(out_csv_path, index=False)

    out_json_path = os.path.join(STEP10_DIR, "outputs", "ablation", "ablation_summary.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)

    print(f"\nSaved ablation results to: {out_csv_path} and {out_json_path}")
    return df_ablation

if __name__ == "__main__":
    run_ablation_study()
