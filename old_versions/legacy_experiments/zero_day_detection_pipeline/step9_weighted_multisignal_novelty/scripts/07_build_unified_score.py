"""
07_build_unified_score.py: Construct unified continuous novelty score S_unified
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from common_utils import load_step9_config

class UnifiedScoreBuilder:
    def __init__(self, weights_dict=None):
        if weights_dict is None:
            self.w_c = 0.25
            self.w_m = 0.25
            self.w_l = 0.25
            self.w_r = 0.25
        else:
            self.w_c = float(weights_dict.get("w_confidence", weights_dict.get("wC", 0.25)))
            self.w_m = float(weights_dict.get("w_mahalanobis", weights_dict.get("wM", 0.25)))
            self.w_l = float(weights_dict.get("w_leaf", weights_dict.get("wL", 0.25)))
            self.w_r = float(weights_dict.get("w_relative", weights_dict.get("wR", 0.25)))

        total_w = self.w_c + self.w_m + self.w_l + self.w_r
        if total_w > 0:
            self.w_c /= total_w
            self.w_m /= total_w
            self.w_l /= total_w
            self.w_r /= total_w

    def compute(self, Z_df):
        z_c = Z_df["Z_confidence"].values.astype(np.float64)
        z_m = Z_df["Z_mahalanobis"].values.astype(np.float64)
        z_l = Z_df["Z_leaf"].values.astype(np.float64)
        z_r = Z_df["Z_relative"].values.astype(np.float64)

        s_u = self.w_c * z_c + self.w_m * z_m + self.w_l * z_l + self.w_r * z_r
        return np.clip(s_u, 0.0, 1.0)

def run_build_unified_score():
    print("=" * 80)
    print(">>> STEP 9: 07 - BUILD UNIFIED NOVELTY SCORE <<<")
    print("=" * 80)

    step9_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    auc_weights_path = os.path.join(step9_dir, "outputs", "weights", "auc_weights.json")
    with open(auc_weights_path, "r", encoding="utf-8") as f:
        w_data = json.load(f)["weights"]

    builder = UnifiedScoreBuilder(w_data)
    print(f"Loaded weights: wC={builder.w_c:.4f}, wM={builder.w_m:.4f}, wL={builder.w_l:.4f}, wR={builder.w_r:.4f}")
    return builder

if __name__ == "__main__":
    run_build_unified_score()
