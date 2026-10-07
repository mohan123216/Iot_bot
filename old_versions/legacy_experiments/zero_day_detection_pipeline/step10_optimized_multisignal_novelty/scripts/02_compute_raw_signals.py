"""
02_compute_raw_signals.py: Compute the 4 foundational raw novelty signals
Step 10: Optimized Multi-Signal Novelty Fusion
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from common_utils import load_step10_config, get_abs_path, load_frozen_artifacts, ensure_output_dirs, STEP10_DIR

def compute_all_novelty_signals(df, artifacts):
    """
    Computes the 4 novelty signals for any given dataframe:
      1. S_C: Confidence Novelty (1 - Pmax)
      2. S_M: Mahalanobis Distance to predicted class centroid
      3. S_L: Leaf-Space Novelty (1 - LeafSimilarity across 700 trees)
      4. S_R: Relative Distance Novelty (D_M(x, y_hat) / (min_{c != y_hat} D_M(x, c) + eps))
    """
    cfg = artifacts["cfg"]
    clf = artifacts["clf"]
    booster = artifacts["booster"]
    mah_stats = artifacts["mah_stats"]
    leaf_profiles = artifacts["leaf_profiles"]
    mapping = artifacts["mapping"]
    idx_to_class = {int(k): v for k, v in mapping["idx_to_class"].items()}
    num_classes = len(idx_to_class)
    epsilon = float(cfg["parameters"]["epsilon"])

    feats_a = cfg["features"]["representation_a_xgboost"]
    feats_b = cfg["features"]["representation_b_geometry"]

    n_samples = len(df)

    # 1. XGBoost Inference & Signal 1 (Confidence Novelty)
    prob = clf.predict_proba(df[feats_a])
    p_max = prob.max(axis=1)
    pred_idx = np.argmax(prob, axis=1)
    pred_class = [idx_to_class[i] for i in pred_idx]
    signal_confidence = 1.0 - p_max

    # 2. Geometry Standardization (Representation B)
    means_b = mah_stats["means_b"]
    stds_b = mah_stats["stds_b"]
    Z = (df[feats_b].values - means_b) / stds_b

    # 3. Signal 2 (Mahalanobis Novelty) & Signal 4 (Relative Separation)
    centroids = mah_stats["centroids"]
    precisions = mah_stats["precisions"]

    # Compute distance to all 7 known classes (fully vectorized)
    all_dists = np.zeros((n_samples, num_classes), dtype=np.float64)
    for c in range(num_classes):
        diff = Z - centroids[c]
        diff_P = diff @ precisions[c]
        sq_dist = np.sum(diff_P * diff, axis=1)
        all_dists[:, c] = np.sqrt(np.maximum(0.0, sq_dist))

    # Distance to predicted class
    signal_mahalanobis = all_dists[np.arange(n_samples), pred_idx]

    # Distance to closest alternative class (vectorized masking)
    mask = np.ones_like(all_dists, dtype=bool)
    mask[np.arange(n_samples), pred_idx] = False
    masked_dists = np.where(mask, all_dists, np.inf)
    d_other = np.min(masked_dists, axis=1)

    signal_relative = signal_mahalanobis / (d_other + epsilon)

    # 4. Signal 3 (Leaf-Space Novelty)
    leaves = booster.predict(xgb.DMatrix(df[feats_a]), pred_leaf=True).astype(np.int32)
    num_trees = leaves.shape[1]

    leaf_sim = np.zeros(n_samples, dtype=np.float64)
    for c in range(num_classes):
        mask_c = (pred_idx == c)
        if not np.any(mask_c):
            continue
        leaves_c = leaves[mask_c]
        prof_c_tree_freqs = leaf_profiles[c]["tree_freqs"]
        sim_acc = np.zeros(len(leaves_c), dtype=np.float64)
        for t in range(num_trees):
            tree_freq = prof_c_tree_freqs[t]
            t_leaves = leaves_c[:, t]
            t_probs = np.array([tree_freq.get(lid, 0.0) for lid in t_leaves], dtype=np.float64)
            sim_acc += t_probs
        leaf_sim[mask_c] = sim_acc / float(num_trees)

    signal_leaf = 1.0 - leaf_sim

    out_df = pd.DataFrame({
        "pred_idx": pred_idx,
        "pred_class": pred_class,
        "p_max": p_max,
        "signal_confidence": signal_confidence,
        "signal_mahalanobis": signal_mahalanobis,
        "signal_leaf": signal_leaf,
        "signal_relative": signal_relative,
        "d_other": d_other
    })
    if "subcategory" in df.columns:
        out_df["true_class"] = df["subcategory"].values

    return out_df

def run_compute_raw_signals():
    print("=" * 80)
    print(">>> STEP 10: 02 - COMPUTE VALIDATION RAW NOVELTY SIGNALS <<<")
    print("=" * 80)

    ensure_output_dirs()
    artifacts = load_frozen_artifacts()
    cfg = artifacts["cfg"]
    val_path = get_abs_path(cfg["paths"]["validation_path"])
    df_val = pd.read_parquet(val_path)

    t0 = time.time()
    print(f"Computing 4 raw novelty signals for validation set ({len(df_val):,} flows)...")
    val_signals = compute_all_novelty_signals(df_val, artifacts)
    elapsed = time.time() - t0
    print(f"Computed raw signals in {elapsed:.2f}s.")

    out_path = os.path.join(STEP10_DIR, "outputs", "calibration", "raw_validation_signals.parquet")
    val_signals.to_parquet(out_path, index=False)
    print(f"Saved raw validation signals to: {out_path} ({os.path.getsize(out_path)/(1024*1024):.2f} MB)")

    print("\n--- Validation Raw Novelty Signals Summary ---")
    for sig in ["signal_confidence", "signal_mahalanobis", "signal_leaf", "signal_relative"]:
        s = val_signals[sig]
        print(f"  {sig:<20}: Mean={s.mean():.4f} | Std={s.std():.4f} | Min={s.min():.4f} | Median={s.median():.4f} | P95={s.quantile(0.95):.4f} | Max={s.max():.4f}")

    return val_signals

if __name__ == "__main__":
    run_compute_raw_signals()
