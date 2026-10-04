"""
07_validate_weight_robustness.py: Cross-mechanism robustness validation & Class-conditional weights study
Step 10: Optimized Multi-Signal Novelty Fusion
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from common_utils import load_step10_config, load_frozen_artifacts, ensure_output_dirs, STEP10_DIR
from importlib import import_module

opt_weights_mod = import_module("06_optimize_weights")
evaluate_weight_vector = opt_weights_mod.evaluate_weight_vector

def run_validate_weight_robustness():
    print("=" * 80)
    print(">>> STEP 10: 07 - VALIDATE WEIGHT ROBUSTNESS & CLASS-CONDITIONAL STUDY <<<")
    print("=" * 80)

    ensure_output_dirs()
    artifacts = load_frozen_artifacts()
    mapping = artifacts["mapping"]
    idx_to_class = {int(k): v for k, v in mapping["idx_to_class"].items()}
    num_classes = len(idx_to_class)

    # 1. Load Data
    val_calib_path = os.path.join(STEP10_DIR, "outputs", "calibration", "calibrated_validation_signals.parquet")
    df_val = pd.read_parquet(val_calib_path)

    pseudo_calib_path = os.path.join(STEP10_DIR, "outputs", "pseudo_unknown", "pseudo_calibrated_signals.parquet")
    df_pseudo = pd.read_parquet(pseudo_calib_path)

    sig_cols = ["Z_confidence", "Z_mahalanobis", "Z_leaf", "Z_relative"]
    Z_val_mat = df_val[sig_cols].values.astype(np.float64)
    val_pred_idx = df_val["pred_idx"].values.astype(np.int32)
    val_is_normal = (df_val["true_class"].values == "Normal")

    # Load Learned Weights
    weights_path = os.path.join(STEP10_DIR, "outputs", "weights", "optimized_weights.json")
    with open(weights_path, "r", encoding="utf-8") as f:
        weights_data = json.load(f)

    strategies = {
        "Equal Weights": [0.25, 0.25, 0.25, 0.25],
        "Step 9 AUC Weights": [
            weights_data["step9_auc_weights"]["w_confidence"],
            weights_data["step9_auc_weights"]["w_mahalanobis"],
            weights_data["step9_auc_weights"]["w_leaf"],
            weights_data["step9_auc_weights"]["w_relative"]
        ],
        "Old Step 10 (Conf Only)": [
            weights_data.get("old_step10_weights", {}).get("w_confidence", 1.0),
            weights_data.get("old_step10_weights", {}).get("w_mahalanobis", 0.0),
            weights_data.get("old_step10_weights", {}).get("w_leaf", 0.0),
            weights_data.get("old_step10_weights", {}).get("w_relative", 0.0)
        ],
        "New Learned Four-Signal": [
            weights_data["selected_optimal_weights"]["w_confidence"],
            weights_data["selected_optimal_weights"]["w_mahalanobis"],
            weights_data["selected_optimal_weights"]["w_leaf"],
            weights_data["selected_optimal_weights"]["w_relative"]
        ]
    }

    # 2. Evaluate Robustness Across Generation Mechanisms
    generator_types = ["cross_class_interpolation", "controlled_perturbation", "boundary_low_density", "hard_pseudo_unknown"]
    robustness_records = []

    print("\n--- Weight Generalization Across Pseudo-Unknown Generation Mechanisms ---")
    print(f"{'Strategy':<24} | {'Cohort':<28} | {'Recall':<8} | {'Precision':<9} | {'F1':<8} | {'AUROC':<7}")
    print("-" * 92)

    for strat_name, w_vec in strategies.items():
        # Overall
        Z_pseudo_mat = df_pseudo[sig_cols].values.astype(np.float64)
        pseudo_pred_idx = df_pseudo["pred_idx"].values.astype(np.int32)
        res_overall = evaluate_weight_vector(w_vec, Z_val_mat, Z_pseudo_mat, val_pred_idx, pseudo_pred_idx, val_is_normal)

        robustness_records.append({
            "strategy": strat_name,
            "cohort": f"Aggregate (All {len(df_pseudo):,})",
            "sample_count": len(df_pseudo),
            "recall": round(res_overall["recall_pseudo"] * 100.0, 2),
            "precision": round(res_overall["precision_pseudo"] * 100.0, 2),
            "f1": round(res_overall["f1_pseudo"] * 100.0, 2),
            "auroc": round(res_overall["auroc"], 4)
        })
        print(f"{strat_name:<24} | {f'Aggregate (All {len(df_pseudo):,})':<28} | {res_overall['recall_pseudo']*100:>6.2f}% | {res_overall['precision_pseudo']*100:>7.2f}% | {res_overall['f1_pseudo']*100:>6.2f}% | {res_overall['auroc']:.4f}")

        # Per generator cohort
        for g_type in generator_types:
            sub_pseudo = df_pseudo[df_pseudo["generator_type"] == g_type]
            sub_z = sub_pseudo[sig_cols].values.astype(np.float64)
            sub_pred_idx = sub_pseudo["pred_idx"].values.astype(np.int32)

            res_cohort = evaluate_weight_vector(w_vec, Z_val_mat, sub_z, val_pred_idx, sub_pred_idx, val_is_normal)

            robustness_records.append({
                "strategy": strat_name,
                "cohort": g_type,
                "sample_count": len(sub_pseudo),
                "recall": round(res_cohort["recall_pseudo"] * 100.0, 2),
                "precision": round(res_cohort["precision_pseudo"] * 100.0, 2),
                "f1": round(res_cohort["f1_pseudo"] * 100.0, 2),
                "auroc": round(res_cohort["auroc"], 4)
            })
            print(f"{strat_name:<22} | {g_type:<26} | {res_cohort['recall_pseudo']*100:>6.2f}% | {res_cohort['precision_pseudo']*100:>7.2f}% | {res_cohort['f1_pseudo']*100:>6.2f}% | {res_cohort['auroc']:.4f}")

    df_robustness = pd.DataFrame(robustness_records)
    rob_csv_path = os.path.join(STEP10_DIR, "outputs", "weights", "weight_robustness.csv")
    df_robustness.to_csv(rob_csv_path, index=False)

    # 3. Class-Conditional Weights Study (Experimental Extension)
    print("\n--- Investigating Class-Conditional Weights (Experimental Extension) ---")
    w_global = np.array(strategies["New Learned Four-Signal"], dtype=np.float64)

    class_counts = df_val["pred_idx"].value_counts().to_dict()
    print("Validation Flows per Predicted Class:")
    for c in range(num_classes):
        c_name = idx_to_class[c]
        cnt = class_counts.get(c, 0)
        print(f"  Class {c} ({c_name:<18}): {cnt:>6,d} flows")

    # Rule: Only optimize for sufficiently populated classes (N >= 500)
    # Small classes: Data_Exfiltration (2), Keylogging (8), HTTP (40) stick strictly to global weights
    class_weights = {}
    shrinkage_factor = 0.50  # 50% local + 50% global shrinkage

    for c in range(num_classes):
        c_name = idx_to_class[c]
        val_mask_c = (val_pred_idx == c)
        n_c = val_mask_c.sum()

        if n_c < 500:
            class_weights[c_name] = {
                "class_idx": c,
                "val_flow_count": int(n_c),
                "optimization_status": "FROZEN_TO_GLOBAL (Small Sample Count)",
                "weights": [round(float(v), 4) for v in w_global]
            }
            continue

        # Fit local weights for class c
        Z_val_c = Z_val_mat[val_mask_c]
        pseudo_mask_c = (df_pseudo["pred_idx"].values == c)
        n_pseudo_c = pseudo_mask_c.sum()

        if n_pseudo_c < 50:
            class_weights[c_name] = {
                "class_idx": c,
                "val_flow_count": int(n_c),
                "pseudo_flow_count": int(n_pseudo_c),
                "optimization_status": "FROZEN_TO_GLOBAL (Low Pseudo Support)",
                "weights": [round(float(v), 4) for v in w_global]
            }
            continue

        Z_pseudo_c = df_pseudo.loc[pseudo_mask_c, sig_cols].values.astype(np.float64)

        # Simple grid search on local class data
        best_local_f1 = -1.0
        best_local_w = w_global

        test_grid = [
            w_global,
            [0.5, 0.3, 0.1, 0.1],
            [0.3, 0.5, 0.1, 0.1],
            [0.2, 0.6, 0.1, 0.1],
            [0.4, 0.2, 0.2, 0.2],
            [0.1, 0.7, 0.1, 0.1],
            [0.2, 0.4, 0.1, 0.3],
            [0.3, 0.3, 0.2, 0.2]
        ]
        for w_cand in test_grid:
            w_cand = np.array(w_cand) / np.sum(w_cand)
            s_val = Z_val_c @ w_cand
            s_pse = Z_pseudo_c @ w_cand
            th = np.percentile(s_val, 95.0)
            rec = np.mean(s_pse > th)
            fp = np.sum(s_val > th)
            tp = np.sum(s_pse > th)
            prec = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            if f1 > best_local_f1:
                best_local_f1 = f1
                best_local_w = w_cand

        # Apply shrinkage toward global weights
        w_shrunk = shrinkage_factor * best_local_w + (1.0 - shrinkage_factor) * w_global
        w_shrunk /= np.sum(w_shrunk)

        class_weights[c_name] = {
            "class_idx": c,
            "val_flow_count": int(n_c),
            "pseudo_flow_count": int(n_pseudo_c),
            "optimization_status": "LEARNED_WITH_SHRINKAGE",
            "shrinkage_factor": shrinkage_factor,
            "local_weights": [round(float(v), 4) for v in best_local_w],
            "shrunk_weights": [round(float(v), 4) for v in w_shrunk],
            "weights": [round(float(v), 4) for v in w_shrunk]
        }

    print("\nClass-Conditional Weight Allocations:")
    for c_name, c_info in class_weights.items():
        w_str = ", ".join([f"{v:.4f}" for v in c_info["weights"]])
        print(f"  {c_name:<18} [{c_info['optimization_status']}]: [{w_str}]")

    # Save outputs
    class_weights_json = os.path.join(STEP10_DIR, "outputs", "weights", "class_conditional_weights.json")
    with open(class_weights_json, "w", encoding="utf-8") as f:
        json.dump(class_weights, f, indent=2)

    print(f"\nSaved robustness results to: {rob_csv_path}")
    print(f"Saved class-conditional weights to: {class_weights_json}")
    return df_robustness, class_weights

if __name__ == "__main__":
    run_validate_weight_robustness()
