"""
06_learn_weights.py: Learn data-driven multi-signal weights from calibration data
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from common_utils import load_step9_config, load_frozen_artifacts
from importlib import import_module

compute_signals_mod = import_module("02_compute_novelty_signals")
compute_all_novelty_signals = compute_signals_mod.compute_all_novelty_signals

norm_mod = import_module("03_percentile_normalization")
PercentileNormalizer = norm_mod.PercentileNormalizer

def run_learn_weights():
    print("=" * 80)
    print(">>> STEP 9: 06 - LEARN DATA-DRIVEN SIGNAL WEIGHTS <<<")
    print("=" * 80)

    step9_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_step9_config()

    # 1. Load Signal Metrics & Extract Aggregate AUCs
    metrics_path = os.path.join(step9_dir, "outputs", "signal_analysis", "signal_metrics.csv")
    df_metrics = pd.read_csv(metrics_path)

    df_agg = df_metrics[df_metrics["cohort"] == "All Generators (Aggregate)"].set_index("signal_key")
    auc_c = df_agg.loc["Z_confidence", "auroc"]
    auc_m = df_agg.loc["Z_mahalanobis", "auroc"]
    auc_l = df_agg.loc["Z_leaf", "auroc"]
    auc_r = df_agg.loc["Z_relative", "auroc"]

    print("Aggregate AUROC Values on Calibration Pseudo-Unknowns:")
    print(f"  Confidence Novelty:        {auc_c:.4f}")
    print(f"  Mahalanobis Novelty:       {auc_m:.4f}")
    print(f"  Leaf Novelty:              {auc_l:.4f}")
    print(f"  Relative Class-Separation: {auc_r:.4f}")

    # Method 1: AUC-Derived Weights
    # q_i = max(AUC_i - 0.5, 0); w_i = q_i / sum(q)
    q_c = max(auc_c - 0.5, 0.0)
    q_m = max(auc_m - 0.5, 0.0)
    q_l = max(auc_l - 0.5, 0.0)
    q_r = max(auc_r - 0.5, 0.0)
    q_sum = q_c + q_m + q_l + q_r

    if q_sum > 0:
        w_auc = np.array([q_c, q_m, q_l, q_r]) / q_sum
    else:
        w_auc = np.array([0.25, 0.25, 0.25, 0.25])

    auc_weights_dict = {
        "method": "AUC-Derived Discrimination Above Chance",
        "formula": "w_i = max(AUC_i - 0.5, 0) / sum_j max(AUC_j - 0.5, 0)",
        "weights": {
            "w_confidence": round(float(w_auc[0]), 4),
            "w_mahalanobis": round(float(w_auc[1]), 4),
            "w_leaf": round(float(w_auc[2]), 4),
            "w_relative": round(float(w_auc[3]), 4)
        },
        "signal_aucs": {
            "confidence": float(auc_c),
            "mahalanobis": float(auc_m),
            "leaf": float(auc_l),
            "relative": float(auc_r)
        }
    }
    auc_json_path = os.path.join(step9_dir, "outputs", "weights", "auc_weights.json")
    with open(auc_json_path, "w", encoding="utf-8") as f:
        json.dump(auc_weights_dict, f, indent=2)

    # Method 2: Direct Optimization on Validation vs Pseudo-Unknowns
    print("\nMethod 2: Direct Optimization on Calibration Data...")
    val_signals_path = os.path.join(step9_dir, "outputs", "calibration", "calibration_signal_scores.parquet")
    df_val_signals = pd.read_parquet(val_signals_path)

    pseudo_path = os.path.join(step9_dir, "outputs", "pseudo_unknown", "pseudo_unknown_samples.parquet")
    df_pseudo = pd.read_parquet(pseudo_path)
    artifacts = load_frozen_artifacts()
    df_pseudo_signals = compute_all_novelty_signals(df_pseudo, artifacts)

    norm_path = os.path.join(step9_dir, "outputs", "calibration", "percentile_reference.json")
    normalizer = PercentileNormalizer()
    normalizer.load(norm_path)

    Z_val = normalizer.transform(df_val_signals)[["Z_confidence", "Z_mahalanobis", "Z_leaf", "Z_relative"]].values
    Z_pseudo = normalizer.transform(df_pseudo_signals)[["Z_confidence", "Z_mahalanobis", "Z_leaf", "Z_relative"]].values

    # Optimize weights to minimize loss L = alpha * (1 - Recall_pseudo) + (1 - alpha) * FPR_known at 95th percentile
    alphas = cfg["parameters"]["weighting"]["alpha_candidates"]
    selected_alpha = cfg["parameters"]["weighting"]["selected_alpha"]
    optimized_by_alpha = {}

    def objective_fn(w, alpha_param):
        w = np.array(w)
        score_val = Z_val @ w
        score_pseudo = Z_pseudo @ w

        # Threshold at 95th percentile of validation score
        th = np.percentile(score_val, 95.0)
        fpr = np.mean(score_val > th)
        recall = np.mean(score_pseudo > th)

        # Loss: balance pseudo detection vs known FPR
        loss = alpha_param * (1.0 - recall) + (1.0 - alpha_param) * fpr
        # Add smooth tie-breaker against degeneracy (-AUROC)
        y_true = np.concatenate([np.zeros(len(score_val)), np.ones(len(score_pseudo))])
        y_scores = np.concatenate([score_val, score_pseudo])
        auc = roc_auc_score(y_true, y_scores)
        return loss - 0.05 * auc

    bounds = [(0.0, 1.0) for _ in range(4)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    init_w = [0.25, 0.25, 0.25, 0.25]

    for a in alphas:
        res = minimize(
            objective_fn,
            init_w,
            args=(a,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-6}
        )
        w_opt = np.maximum(0.0, res.x)
        w_opt /= np.sum(w_opt)
        optimized_by_alpha[str(a)] = [round(float(v), 4) for v in w_opt]

    best_opt_w = optimized_by_alpha[str(selected_alpha)]
    opt_weights_dict = {
        "method": "Direct Validation vs Pseudo-Unknown Constrained Optimization",
        "selected_alpha": selected_alpha,
        "selected_weights": {
            "w_confidence": best_opt_w[0],
            "w_mahalanobis": best_opt_w[1],
            "w_leaf": best_opt_w[2],
            "w_relative": best_opt_w[3]
        },
        "weights_by_alpha": optimized_by_alpha
    }
    opt_json_path = os.path.join(step9_dir, "outputs", "weights", "optimized_weights.json")
    with open(opt_json_path, "w", encoding="utf-8") as f:
        json.dump(opt_weights_dict, f, indent=2)

    # Comparison Table
    weight_comp_df = pd.DataFrame([
        {
            "Method": "Equal Baseline",
            "wC (Confidence)": 0.2500,
            "wM (Mahalanobis)": 0.2500,
            "wL (Leaf Space)": 0.2500,
            "wR (Relative Distance)": 0.2500
        },
        {
            "Method": "AUC-Derived (Method 1)",
            "wC (Confidence)": auc_weights_dict["weights"]["w_confidence"],
            "wM (Mahalanobis)": auc_weights_dict["weights"]["w_mahalanobis"],
            "wL (Leaf Space)": auc_weights_dict["weights"]["w_leaf"],
            "wR (Relative Distance)": auc_weights_dict["weights"]["w_relative"]
        },
        {
            "Method": f"Directly Optimized (alpha={selected_alpha})",
            "wC (Confidence)": best_opt_w[0],
            "wM (Mahalanobis)": best_opt_w[1],
            "wL (Leaf Space)": best_opt_w[2],
            "wR (Relative Distance)": best_opt_w[3]
        }
    ])
    weight_comp_csv = os.path.join(step9_dir, "outputs", "weights", "weight_comparison.csv")
    weight_comp_df.to_csv(weight_comp_csv, index=False)

    print("\n--- Weight Comparison Table ---")
    print(weight_comp_df.to_string(index=False))
    print(f"\nSaved weights to: {auc_json_path}, {opt_json_path}, {weight_comp_csv}")
    return weight_comp_df

if __name__ == "__main__":
    run_learn_weights()
