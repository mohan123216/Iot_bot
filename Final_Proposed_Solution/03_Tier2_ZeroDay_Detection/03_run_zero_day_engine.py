#!/usr/bin/env python3
"""
========================================================================================
STEP 3: TIER 2 - 4-SIGNAL OPEN-SET ZERO-DAY REJECTION ENGINE
========================================================================================
Project: Hybrid Open-Set Multi-Signal Zero-Day Engine (HOMZ-Engine)

Architecture:
  Canonical 4-Signal Novelty Detection & Open-Set Rejection Architecture (Step 10):
    - Signal 1: Confidence Novelty (S_C = 1.0 - P_max)
    - Signal 2: Log-Manifold Ledoit-Wolf Mahalanobis Distance (S_M)
    - Signal 3: Decision Tree Leaf-Space Traversal Novelty (S_L = 1.0 - LeafSim)
    - Signal 4: Relative Neighborhood Margin Distance Ratio (S_R = D_M / D_other)

Calibration & Rejection Protocol:
  1. Non-Parametric Empirical CDF Calibration:
     - Fits empirical cumulative distribution functions on known validation traffic.
     - Uses np.searchsorted to map continuous raw signals into calibrated percentile
       ranks Z_i in [0.0, 1.0].
  2. Multi-Signal Fusion:
     - Unified Novelty Score: S_unified(x) = sum(w_i * Z_i(x)) over the 4 signals.
  3. Operational Percentile Thresholding (tau):
     - tau = np.percentile(S_val, 95.0) calibrated strictly on validation known traffic.
  4. The Rejection Decision:
     - If S_unified(x) > tau  ==> REJECT as Zero-Day Novel Threat
     - Else                   ==> ACCEPT as Known Traffic
========================================================================================
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.covariance import LedoitWolf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, "..", "01_Data_Preprocessing", "dataset", "fulldataset_cleaned_sample.parquet")
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
METRICS_CSV = os.path.join(RESULTS_DIR, "zero_day_detection_standard_metrics.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Feature subsets (Representation A for Trees, Representation B for Geometry)
FEATS_A = [
    "proto_number", "flgs_number", "state_number", "sport", "dport",
    "dur", "pkts", "bytes", "spkts", "dpkts", "sbytes", "dbytes",
    "rate", "srate", "drate", "mean", "stddev", "sum", "min", "max"
]

FEATS_B = [
    "proto_number", "flgs_number", "state_number",
    "dur", "pkts", "bytes", "spkts", "dpkts", "sbytes", "dbytes",
    "rate", "srate"
]

LOG_COLS = ["dur", "pkts", "bytes", "spkts", "dpkts", "sbytes", "dbytes", "rate", "srate"]

# --------------------------------------------------------------------------------------
# 1. Non-Parametric Empirical CDF Normalizer (Percentile Rank Calibration)
# --------------------------------------------------------------------------------------
class EmpiricalCDFNormalizer:
    """
    Nonparametric Empirical CDF calibrator.
    Maps raw novelty score S_i into calibrated percentile rank Z_i in [0.0, 1.0].
    Fitted strictly on Validation Known Traffic ONLY.
    Z_i(s) = F_i(s) = P(S_known <= s) via np.searchsorted.
    """
    def __init__(self):
        self.signal_names = ["signal_confidence", "signal_mahalanobis", "signal_leaf", "signal_relative"]
        self.short_keys = {
            "signal_confidence": "Z_confidence",
            "signal_mahalanobis": "Z_mahalanobis",
            "signal_leaf": "Z_leaf",
            "signal_relative": "Z_relative"
        }
        self.sorted_references = {}
        self.n_samples = {}

    def fit(self, val_signals_df):
        for sig in self.signal_names:
            vals = val_signals_df[sig].values.astype(np.float64)
            sorted_v = np.sort(vals)
            self.sorted_references[sig] = sorted_v
            self.n_samples[sig] = len(sorted_v)

    def transform(self, signals_df):
        out_df = pd.DataFrame(index=signals_df.index)
        for sig in self.signal_names:
            raw_s = signals_df[sig].values.astype(np.float64)
            sorted_ref = self.sorted_references[sig]
            n = self.n_samples[sig]
            ranks = np.searchsorted(sorted_ref, raw_s, side="right") / float(n)
            col_name = self.short_keys[sig]
            out_df[col_name] = np.clip(ranks, 0.0, 1.0)
        return out_df

# --------------------------------------------------------------------------------------
# 2. Geometry Statistics (Ledoit-Wolf Regularized Mahalanobis Distance)
# --------------------------------------------------------------------------------------
def extract_mahalanobis_statistics(df_train, feats_b, known_classes, class_to_idx):
    """
    Fits class centroids and Ledoit-Wolf precision matrices strictly on training known data.
    Uses log1p transformation on continuous heavy-tailed flow features.
    """
    X_train_b = df_train[feats_b].values.copy().astype(np.float64)
    for ci, col in enumerate(feats_b):
        if col in LOG_COLS:
            X_train_b[:, ci] = np.log1p(np.maximum(0.0, X_train_b[:, ci]))

    means_b = np.mean(X_train_b, axis=0)
    stds_b = np.std(X_train_b, axis=0)
    stds_b[stds_b < 1e-6] = 1.0

    Z_train = (X_train_b - means_b) / stds_b

    centroids = {}
    precisions = {}

    for c_name in known_classes:
        c_idx = class_to_idx[c_name]
        mask_c = (df_train["subcategory"].values == c_name)
        Z_c = Z_train[mask_c]

        centroid = np.mean(Z_c, axis=0)
        centroids[c_idx] = centroid

        if len(Z_c) > 20:
            lw = LedoitWolf(assume_centered=False)
            lw.fit(Z_c)
            precisions[c_idx] = lw.precision_
        else:
            precisions[c_idx] = np.eye(len(feats_b))

    return {
        "means_b": means_b,
        "stds_b": stds_b,
        "centroids": centroids,
        "precisions": precisions
    }

# --------------------------------------------------------------------------------------
# 3. Decision Tree Leaf-Space Traversal Profiles
# --------------------------------------------------------------------------------------
def extract_leaf_profiles(df_train, booster, feats_a, known_classes, class_to_idx):
    """
    Builds tree-frequency distributions for each leaf across all trees strictly from training data.
    """
    X_train_a = df_train[feats_a].values
    leaves = booster.predict(xgb.DMatrix(X_train_a), pred_leaf=True).astype(np.int32)
    num_trees = leaves.shape[1]

    leaf_profiles = {}
    for c_name in known_classes:
        c_idx = class_to_idx[c_name]
        mask_c = (df_train["subcategory"].values == c_name)
        leaves_c = leaves[mask_c]
        n_c = len(leaves_c)

        tree_freqs = []
        for t in range(num_trees):
            t_leaves = leaves_c[:, t]
            vals, counts = np.unique(t_leaves, return_counts=True)
            freq_dict = {int(v): float(cnt) / float(n_c) for v, cnt in zip(vals, counts)}
            tree_freqs.append(freq_dict)

        leaf_profiles[c_idx] = {
            "class_name": c_name,
            "sample_count": n_c,
            "tree_freqs": tree_freqs
        }
    return leaf_profiles

# --------------------------------------------------------------------------------------
# 4. Computation of the 4 Foundational Novelty Signals
# --------------------------------------------------------------------------------------
def compute_4_novelty_signals(df, clf, booster, mah_stats, leaf_profiles, feats_a, feats_b, idx_to_class, epsilon=1e-12):
    """
    Computes the 4 foundational novelty signals:
      1. S_C: Confidence Novelty (1 - P_max)
      2. S_M: Mahalanobis Distance to predicted class centroid
      3. S_L: Leaf-Space Novelty (1 - LeafSimilarity across all trees)
      4. S_R: Relative Distance Novelty (D_M(pred) / (min_{c != pred} D_M(c) + eps))
    """
    n_samples = len(df)
    num_classes = len(idx_to_class)

    # 1. XGBoost Probabilities & Signal 1 (Confidence Novelty)
    X_a = df[feats_a].values
    prob = clf.predict_proba(X_a)
    p_max = prob.max(axis=1)
    pred_idx = np.argmax(prob, axis=1)
    signal_confidence = 1.0 - p_max

    # 2. Geometry Standardization (Representation B)
    means_b = mah_stats["means_b"]
    stds_b = mah_stats["stds_b"]
    X_b = df[feats_b].values.copy().astype(np.float64)
    for ci, col in enumerate(feats_b):
        if col in LOG_COLS:
            X_b[:, ci] = np.log1p(np.maximum(0.0, X_b[:, ci]))

    Z = (X_b - means_b) / stds_b

    # 3. Signal 2 (Mahalanobis Distance) & Signal 4 (Relative Distance Ratio)
    centroids = mah_stats["centroids"]
    precisions = mah_stats["precisions"]

    all_dists = np.zeros((n_samples, num_classes), dtype=np.float64)
    for c in range(num_classes):
        diff = Z - centroids[c]
        diff_P = diff @ precisions[c]
        sq_dist = np.sum(diff_P * diff, axis=1)
        all_dists[:, c] = np.sqrt(np.maximum(0.0, sq_dist))

    signal_mahalanobis = all_dists[np.arange(n_samples), pred_idx]

    mask = np.ones_like(all_dists, dtype=bool)
    mask[np.arange(n_samples), pred_idx] = False
    masked_dists = np.where(mask, all_dists, np.inf)
    d_other = np.min(masked_dists, axis=1)
    signal_relative = signal_mahalanobis / (d_other + epsilon)

    # 4. Signal 3 (Leaf Novelty across 100 Trees)
    leaves = booster.predict(xgb.DMatrix(X_a), pred_leaf=True).astype(np.int32)
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

    return pd.DataFrame({
        "pred_idx": pred_idx,
        "signal_confidence": signal_confidence,
        "signal_mahalanobis": signal_mahalanobis,
        "signal_leaf": signal_leaf,
        "signal_relative": signal_relative
    }, index=df.index)

# --------------------------------------------------------------------------------------
# 5. Live Demonstration of the 4-Signal Rejection Mechanism
# --------------------------------------------------------------------------------------
def run_live_rejection_demo(df_all, heldout_attack="Data_Exfiltration"):
    print("\n" + "=" * 85)
    print(f">>> LIVE REJECTION DEMONSTRATION ON HELDOUT ZERO-DAY: {heldout_attack} <<<")
    print("=" * 85)

    df_zd = df_all[df_all["subcategory"] == heldout_attack].copy()
    df_known = df_all[df_all["subcategory"] != heldout_attack].copy()

    known_classes = sorted(df_known["subcategory"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(known_classes)}
    idx_to_class = {i: c for i, c in enumerate(known_classes)}
    num_classes = len(known_classes)

    # 70% Train, 15% Validation, 15% Test
    train_df, temp_df = train_test_split(df_known, test_size=0.30, stratify=df_known["subcategory"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["subcategory"], random_state=42)

    print(f"1. Partitions: Train={len(train_df):,} | Val={len(val_df):,} | Known Test={len(test_df):,} | Heldout Zero-Day={len(df_zd):,}")

    # Train Tier-1 XGBoost on Known Classes
    y_train = train_df["subcategory"].map(class_to_idx).values
    class_counts = Counter(y_train)
    total_samples = len(y_train)
    weights_map = {c: total_samples / (num_classes * count) for c, count in class_counts.items()}
    sample_weights = np.array([weights_map[y] for y in y_train])

    clf = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        objective="multi:softprob", num_class=num_classes,
        random_state=42, n_jobs=-1
    )
    clf.fit(train_df[FEATS_A].values, y_train, sample_weight=sample_weights)
    booster = clf.get_booster()

    # Extract Mahalanobis Geometry & Leaf Profiles on Train Data
    print("2. Extracting Ledoit-Wolf Mahalanobis Geometry & 100-Tree Leaf Profiles on Train Data...")
    mah_stats = extract_mahalanobis_statistics(train_df, FEATS_B, known_classes, class_to_idx)
    leaf_profiles = extract_leaf_profiles(train_df, booster, FEATS_A, known_classes, class_to_idx)

    # Compute 4 Signals on Validation Data & Calibrate Empirical CDF
    print("3. Calibrating Empirical CDF Percentile Normalizer strictly on Validation Known Traffic...")
    val_sig = compute_4_novelty_signals(val_df, clf, booster, mah_stats, leaf_profiles, FEATS_A, FEATS_B, idx_to_class)
    normalizer = EmpiricalCDFNormalizer()
    normalizer.fit(val_sig)

    # Convert to Percentile Ranks Z_i in [0.0, 1.0]
    Z_val = normalizer.transform(val_sig)
    sig_cols = ["Z_confidence", "Z_mahalanobis", "Z_leaf", "Z_relative"]
    # Unified Score (Equal weights w = [0.25, 0.25, 0.25, 0.25])
    s_val_unified = Z_val[sig_cols].mean(axis=1).values

    # Calibrate Operating Threshold at 95.0th percentile of Known Validation Traffic
    tau = np.percentile(s_val_unified, 95.0)
    print(f"4. Calibrated Operating Threshold (tau at 95.0 percentile): {tau:.4f}")

    # Evaluate on Known Test Traffic and Heldout Zero-Day Traffic
    print("5. Evaluating Open-Set Rejection Rule on Heldout Test Sets...")
    test_sig = compute_4_novelty_signals(test_df, clf, booster, mah_stats, leaf_profiles, FEATS_A, FEATS_B, idx_to_class)
    Z_test = normalizer.transform(test_sig)
    s_test_unified = Z_test[sig_cols].mean(axis=1).values

    zd_sig = compute_4_novelty_signals(df_zd, clf, booster, mah_stats, leaf_profiles, FEATS_A, FEATS_B, idx_to_class)
    Z_zd = normalizer.transform(zd_sig)
    s_zd_unified = Z_zd[sig_cols].mean(axis=1).values

    # REJECTION DECISION:
    # Score > tau  ==> REJECT as Zero-Day (1)
    # Score <= tau ==> ACCEPT as Known Traffic (0)
    pred_reject_test = (s_test_unified > tau).astype(np.int32)
    pred_reject_zd = (s_zd_unified > tau).astype(np.int32)

    known_acceptance = float(np.mean(pred_reject_test == 0)) * 100.0
    zeroday_catch_rate = float(np.mean(pred_reject_zd == 1)) * 100.0

    # Specifically check Benign Normal False Alarm Rate
    normal_mask = (test_df["subcategory"].values == "Normal")
    normal_false_alarm_rate = float(np.mean(pred_reject_test[normal_mask] == 1)) * 100.0
    normal_specificity = 100.0 - normal_false_alarm_rate

    print("\n" + "-" * 85)
    print(f"REJECTION MECHANISM VERIFICATION RESULTS:")
    print("-" * 85)
    print(f"  * Known Traffic Acceptance Rate      : {known_acceptance:.2f}% (Expected: ~95.0%)")
    print(f"  * Benign Normal Specificity          : {normal_specificity:.2f}% (Benign False Alarms: {normal_false_alarm_rate:.2f}%)")
    print(f"  * Held-Out Zero-Day Catch Rate       : {zeroday_catch_rate:.2f}% ({int(np.sum(pred_reject_zd))}/{len(df_zd)} caught)")
    print("-" * 85)

# --------------------------------------------------------------------------------------
# 6. Main Execution
# --------------------------------------------------------------------------------------
def run_step_3(live_demo=True):
    print("=" * 85)
    print(">>> STEP 3: TIER 2 - 4-SIGNAL OPEN-SET ZERO-DAY REJECTION ENGINE <<<")
    print("=" * 85)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Parquet missing at: {DATA_PATH}")

    df_all = pd.read_parquet(DATA_PATH)

    # 1. Run live demonstration of the rejection mechanism
    if live_demo:
        run_live_rejection_demo(df_all, heldout_attack="Data_Exfiltration")

    # 2. Display official benchmark across all 10 held-out zero-day attacks
    if os.path.exists(METRICS_CSV):
        print("\n" + "=" * 95)
        print(">>> OFFICIAL BENCHMARK ACROSS ALL 10 HELD-OUT ZERO-DAY ATTACK VECTORS <<<")
        print("=" * 95)
        df_metrics = pd.read_csv(METRICS_CSV)

        print("-" * 95)
        print(f"{'Attack Category':<15} | {'Subclass':<18} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Error Rate':<10}")
        print("-" * 95)
        for _, row in df_metrics.iterrows():
            cat = str(row['Attack_Category'])
            sub = str(row['Attack_Subclass'])
            acc = f"{row['Accuracy_pct']:.2f}%"
            prec = f"{row['Precision_pct']:.2f}%"
            rec = f"{row['Recall_pct']:.2f}%"
            f1 = f"{row['F1_Score_pct']:.2f}%"
            err = f"{row['Error_Rate_pct']:.2f}%"
            print(f"{cat:<15} | {sub:<18} | {acc:>10} | {prec:>10} | {rec:>10} | {f1:>10} | {err:>10}")
        print("-" * 95)

        avg_acc = df_metrics['Accuracy_pct'].mean()
        avg_prec = df_metrics['Precision_pct'].mean()
        avg_rec = df_metrics['Recall_pct'].mean()
        avg_f1 = df_metrics['F1_Score_pct'].mean()
        avg_err = df_metrics['Error_Rate_pct'].mean()

        print(f"{'OVERALL AVERAGE':<15} | {'10 Held-Out Attacks':<18} | {avg_acc:>9.2f}% | {avg_prec:>9.2f}% | {avg_rec:>9.2f}% | {avg_f1:>9.2f}% | {avg_err:>9.2f}%")
        print("=" * 95)

    print("\n>>> STEP 3 COMPLETED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", default=True, help="Run live rejection demo")
    args = parser.parse_args()
    run_step_3(live_demo=args.demo)
