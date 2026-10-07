"""
08_calibrate_adaptive_thresholds.py: Calibrate class-specific adaptive thresholds on validation known data
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from common_utils import load_step9_config, load_frozen_artifacts
from importlib import import_module

norm_mod = import_module("03_percentile_normalization")
PercentileNormalizer = norm_mod.PercentileNormalizer

score_mod = import_module("07_build_unified_score")
UnifiedScoreBuilder = score_mod.UnifiedScoreBuilder

def run_calibrate_adaptive_thresholds():
    print("=" * 80)
    print(">>> STEP 9: 08 - CALIBRATE CLASS-CONDITIONAL ADAPTIVE THRESHOLDS <<<")
    print("=" * 80)

    step9_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_step9_config()
    artifacts = load_frozen_artifacts()
    mapping = artifacts["mapping"]
    idx_to_class = {int(k): v for k, v in mapping["idx_to_class"].items()}
    num_classes = len(idx_to_class)

    # 1. Load Validation Known Signals
    val_signals_path = os.path.join(step9_dir, "outputs", "calibration", "calibration_signal_scores.parquet")
    df_val = pd.read_parquet(val_signals_path)

    # 2. Transform to Z_i
    norm_path = os.path.join(step9_dir, "outputs", "calibration", "percentile_reference.json")
    normalizer = PercentileNormalizer()
    normalizer.load(norm_path)
    Z_val = normalizer.transform(df_val)

    # 3. Load Learned Weights (Using Method 1 AUC-derived as primary; also load optimized)
    auc_weights_path = os.path.join(step9_dir, "outputs", "weights", "auc_weights.json")
    with open(auc_weights_path, "r", encoding="utf-8") as f:
        w_data = json.load(f)["weights"]

    builder = UnifiedScoreBuilder(w_data)
    s_unified_val = builder.compute(Z_val)
    df_val["S_unified"] = s_unified_val

    # 4. Compute Class-Conditional Percentiles
    percentiles = cfg["parameters"]["thresholds"]["candidate_percentiles"]

    threshold_rows = []
    print("\n--- Calibrated Class-Conditional Adaptive Percentiles (VALIDATION DATA ONLY) ---")
    header_str = f"{'Class':<18} | {'Val Flows':<10}" + "".join([f" | {'P'+str(p):<8}" for p in percentiles])
    print(header_str)
    print("-" * len(header_str))

    for c in range(num_classes):
        c_name = idx_to_class[c]
        mask_c = (df_val["pred_idx"] == c)
        n_c = mask_c.sum()
        sub_scores = df_val.loc[mask_c, "S_unified"].values

        row = {
            "class_idx": c,
            "class_name": c_name,
            "val_flow_count": int(n_c)
        }

        row_print_str = f"{c_name:<18} | {n_c:>10,d}"
        for p in percentiles:
            if n_c > 0:
                tau = float(np.percentile(sub_scores, p))
            else:
                tau = float(np.percentile(df_val["S_unified"], p))
            row[f"P{p:.1f}"] = round(tau, 5)
            row_print_str += f" | {tau:<8.5f}"

        threshold_rows.append(row)
        print(row_print_str)

    # Also compute overall global threshold for comparison
    row_global = {
        "class_idx": -1,
        "class_name": "GLOBAL_AVERAGE",
        "val_flow_count": len(df_val)
    }
    row_global_str = f"{'GLOBAL_AVERAGE':<18} | {len(df_val):>10,d}"
    for p in percentiles:
        tau_g = float(np.percentile(df_val["S_unified"], p))
        row_global[f"P{p:.1f}"] = round(tau_g, 5)
        row_global_str += f" | {tau_g:<8.5f}"
    threshold_rows.append(row_global)
    print("-" * len(header_str))
    print(row_global_str)

    df_thresholds = pd.DataFrame(threshold_rows)
    out_thresh_csv = os.path.join(step9_dir, "outputs", "thresholds", "class_conditional_thresholds.csv")
    df_thresholds.to_csv(out_thresh_csv, index=False)
    print(f"\nSaved class-conditional thresholds to: {out_thresh_csv}")

    return df_thresholds

if __name__ == "__main__":
    run_calibrate_adaptive_thresholds()
