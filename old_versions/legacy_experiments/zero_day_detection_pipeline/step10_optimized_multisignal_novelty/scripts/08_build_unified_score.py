"""
08_build_unified_score.py: Construct continuous unified novelty score S_unified
Step 10: Optimized Multi-Signal Novelty Fusion
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from common_utils import load_step10_config, ensure_output_dirs, STEP10_DIR

class UnifiedScoreBuilder:
    """
    Constructs unified continuous novelty score:
    S_unified(x) = sum_i w_i * Z_i(x) in [0.0, 1.0]
    Supports:
      - Global weight vector
      - Class-conditional weight matrix
    """
    def __init__(self, weights_input=None, is_class_conditional=False):
        self.is_class_conditional = is_class_conditional

        if is_class_conditional and isinstance(weights_input, dict):
            self.class_weights = {}
            for c_name, c_dict in weights_input.items():
                w = np.array(c_dict["weights"], dtype=np.float64)
                w = np.maximum(0.0, w)
                self.class_weights[c_dict["class_idx"]] = w / np.sum(w)
        else:
            if weights_input is None:
                self.w = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
            elif isinstance(weights_input, dict):
                w_c = float(weights_input.get("w_confidence", weights_input.get("wC", 0.25)))
                w_m = float(weights_input.get("w_mahalanobis", weights_input.get("wM", 0.25)))
                w_l = float(weights_input.get("w_leaf", weights_input.get("wL", 0.25)))
                w_r = float(weights_input.get("w_relative", weights_input.get("wR", 0.25)))
                self.w = np.array([w_c, w_m, w_l, w_r], dtype=np.float64)
            else:
                self.w = np.array(weights_input, dtype=np.float64)

            self.w = np.maximum(0.0, self.w)
            w_sum = np.sum(self.w)
            self.w = self.w / w_sum if w_sum > 0 else np.array([0.25, 0.25, 0.25, 0.25])

    def compute(self, Z_df, pred_indices=None):
        z_c = Z_df["Z_confidence"].values.astype(np.float64)
        z_m = Z_df["Z_mahalanobis"].values.astype(np.float64)
        z_l = Z_df["Z_leaf"].values.astype(np.float64)
        z_r = Z_df["Z_relative"].values.astype(np.float64)
        Z_mat = np.column_stack([z_c, z_m, z_l, z_r])

        if self.is_class_conditional and pred_indices is not None:
            n = len(Z_mat)
            s_u = np.zeros(n, dtype=np.float64)
            for c_idx, w_c in self.class_weights.items():
                mask_c = (pred_indices == c_idx)
                if np.any(mask_c):
                    s_u[mask_c] = Z_mat[mask_c] @ w_c
        else:
            s_u = Z_mat @ self.w

        return np.clip(s_u, 0.0, 1.0)

def run_build_unified_score():
    print("=" * 80)
    print(">>> STEP 10: 08 - BUILD UNIFIED NOVELTY SCORE FOR VALIDATION SET <<<")
    print("=" * 80)

    ensure_output_dirs()

    # Load Learned Optimal Weights
    opt_weights_path = os.path.join(STEP10_DIR, "outputs", "weights", "optimized_weights.json")
    with open(opt_weights_path, "r", encoding="utf-8") as f:
        weights_info = json.load(f)
    opt_w = weights_info["selected_optimal_weights"]

    builder = UnifiedScoreBuilder(opt_w)
    print(f"Loaded Primary Learned Weights: wC={builder.w[0]:.4f}, wM={builder.w[1]:.4f}, wL={builder.w[2]:.4f}, wR={builder.w[3]:.4f}")

    # Load Calibrated Validation Signals
    val_calib_path = os.path.join(STEP10_DIR, "outputs", "calibration", "calibrated_validation_signals.parquet")
    df_val = pd.read_parquet(val_calib_path)

    s_u = builder.compute(df_val)
    df_val["S_unified"] = s_u

    out_parquet = os.path.join(STEP10_DIR, "outputs", "calibration", "validation_unified_scores.parquet")
    df_val.to_parquet(out_parquet, index=False)
    print(f"Computed S_unified for {len(df_val):,} validation flows. Saved to: {out_parquet}")

    print("\nValidation S_unified Distribution:")
    print(f"  Min={s_u.min():.4f} | Mean={s_u.mean():.4f} | Median={np.median(s_u):.4f} | P95={np.percentile(s_u, 95.0):.4f} | Max={s_u.max():.4f}")

    return builder

if __name__ == "__main__":
    run_build_unified_score()
