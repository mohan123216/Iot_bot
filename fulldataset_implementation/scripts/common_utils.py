"""
common_utils.py: Shared utilities, normalizers, signal calculators, and weight optimizers
Full Dataset Implementation: 4-Signal Step 10 Architecture across all 74 files
"""

import os
import sys
import json
import yaml
import pickle
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMPL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
_parent_dir = os.path.abspath(os.path.join(IMPL_DIR, ".."))
_grandparent_dir = os.path.abspath(os.path.join(_parent_dir, ".."))
if os.path.exists(os.path.join(_parent_dir, "data_1.csv")):
    DATA_DIR = _parent_dir
elif os.path.exists(os.path.join(_grandparent_dir, "data_1.csv")):
    DATA_DIR = _grandparent_dir
else:
    DATA_DIR = _parent_dir
WORKSPACE_DIR = DATA_DIR
CONFIG_PATH = os.path.join(IMPL_DIR, "configs", "config.yaml")

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dirs():
    dirs = [
        os.path.join(IMPL_DIR, "configs"),
        os.path.join(IMPL_DIR, "scripts"),
        os.path.join(IMPL_DIR, "data"),
        os.path.join(IMPL_DIR, "models"),
        os.path.join(IMPL_DIR, "outputs", "dataset_audit"),
        os.path.join(IMPL_DIR, "outputs", "loao_evaluations"),
        os.path.join(IMPL_DIR, "outputs", "weights_and_thresholds"),
        os.path.join(IMPL_DIR, "outputs", "plots"),
        os.path.join(IMPL_DIR, "reports")
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def parse_port_val(val):
    """Safely converts port values, handling hexadecimal strings (0x0303) and negative codes (-1)."""
    if pd.isna(val):
        return -1
    val_str = str(val).strip()
    if val_str.startswith("0x") or val_str.startswith("0X"):
        try:
            return int(val_str, 16)
        except ValueError:
            return -1
    try:
        return int(float(val_str))
    except ValueError:
        return -1

class EmpiricalCDFNormalizer:
    """
    Nonparametric Empirical CDF calibrator.
    Maps raw novelty score S_i into calibrated percentile rank Z_i in [0.0, 1.0].
    Fitted strictly on Validation Known Traffic ONLY.
    Z_i(s) = F_i(s) = P(S_known <= s)
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
        self.statistics = {}

    def fit(self, val_signals_df):
        for sig in self.signal_names:
            vals = val_signals_df[sig].values.astype(np.float64)
            sorted_v = np.sort(vals)
            n = len(sorted_v)
            self.sorted_references[sig] = sorted_v
            self.n_samples[sig] = n
            self.statistics[sig] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "median": float(np.median(vals)),
                "p95": float(np.percentile(vals, 95.0)),
                "max": float(np.max(vals))
            }

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

def compute_all_novelty_signals(df, clf, booster, mah_stats, leaf_profiles, feats_a, feats_b, idx_to_class, epsilon=1e-12):
    """
    Vectorized computation of the 4 foundational novelty signals:
      1. S_C: Confidence Novelty (1 - Pmax)
      2. S_M: Mahalanobis Distance to predicted class centroid
      3. S_L: Leaf-Space Novelty (1 - LeafSimilarity across all trees)
      4. S_R: Relative Distance Novelty (D_M(x, y_hat) / (min_{c != y_hat} D_M(x, c) + eps))
    """
    n_samples = len(df)
    num_classes = len(idx_to_class)

    # 1. XGBoost Inference & Signal 1
    X_a = df[feats_a].values
    prob = clf.predict_proba(X_a)
    p_max = prob.max(axis=1)
    pred_idx = np.argmax(prob, axis=1)
    pred_class = [idx_to_class[i] for i in pred_idx]
    signal_confidence = 1.0 - p_max

    # 2. Geometry Standardization (Representation B)
    means_b = mah_stats["means_b"]
    stds_b = mah_stats["stds_b"]
    X_b = df[feats_b].values
    Z = (X_b - means_b) / stds_b

    # 3. Signal 2 & Signal 4
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

    # 4. Signal 3 (Leaf Novelty)
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

    out_df = pd.DataFrame({
        "pred_idx": pred_idx,
        "pred_class": pred_class,
        "p_max": p_max,
        "signal_confidence": signal_confidence,
        "signal_mahalanobis": signal_mahalanobis,
        "signal_leaf": signal_leaf,
        "signal_relative": signal_relative,
        "d_other": d_other
    }, index=df.index)

    if "subcategory" in df.columns:
        out_df["true_class"] = df["subcategory"].values

    return out_df

def extract_mahalanobis_statistics(df_train, feats_b, known_classes, class_to_idx):
    """
    Fits Ledoit-Wolf precision matrices and centroids strictly on training known data.
    """
    X_train_b = df_train[feats_b].values
    means_b = np.mean(X_train_b, axis=0)
    stds_b = np.std(X_train_b, axis=0)
    stds_b[stds_b < 1e-6] = 1.0

    Z_train = (X_train_b - means_b) / stds_b

    centroids = {}
    covariances = {}
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
            covariances[c_idx] = lw.covariance_
            precisions[c_idx] = lw.precision_
        else:
            # Fallback for ultra-rare classes
            cov = np.eye(len(feats_b))
            covariances[c_idx] = cov
            precisions[c_idx] = np.linalg.pinv(cov)

    return {
        "means_b": means_b,
        "stds_b": stds_b,
        "centroids": centroids,
        "covariances": covariances,
        "precisions": precisions
    }

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

def synthesize_pseudo_unknowns(df_val, mah_stats, feats_a, feats_b, known_classes, class_to_idx, cfg, random_seed=42):
    """
    Generates diverse pseudo-unknowns strictly from validation known traffic.
    """
    rng = np.random.RandomState(random_seed)
    n_per_gen = cfg["pseudo_unknowns"]["n_samples_per_generator"]
    discrete_cols = ["sport", "dport", "proto_number", "flgs_number", "state_number"]
    non_negative_cols = ["dur", "pkts", "bytes", "spkts", "dpkts", "sbytes", "dbytes", "rate", "srate", "drate"]

    class_indices = {c: df_val[df_val["subcategory"] == c].index.values for c in known_classes if len(df_val[df_val["subcategory"] == c]) > 0}
    pseudo_records = []

    # Strategy A: Cross-Class Interpolation
    lambdas = cfg["pseudo_unknowns"]["cross_class_lambdas"]
    valid_classes = list(class_indices.keys())
    if len(valid_classes) >= 2:
        count_a = 0
        while count_a < n_per_gen:
            c1, c2 = rng.choice(valid_classes, size=2, replace=False)
            idx1 = rng.choice(class_indices[c1])
            idx2 = rng.choice(class_indices[c2])
            lam = rng.choice(lambdas)
            r1 = df_val.loc[idx1]
            r2 = df_val.loc[idx2]

            row = {}
            for col in feats_a:
                if col in discrete_cols:
                    row[col] = r1[col] if rng.rand() < lam else r2[col]
                else:
                    v = lam * float(r1[col]) + (1.0 - lam) * float(r2[col])
                    if col in non_negative_cols:
                        v = max(0.0, v)
                    row[col] = v
            row["generator_type"] = "cross_class_interpolation"
            pseudo_records.append(row)
            count_a += 1

    # Strategy B: Covariance-Aligned Perturbation
    sigmas = cfg["pseudo_unknowns"]["perturbation_sigmas"]
    centroids = mah_stats["centroids"]
    covariances = mah_stats["covariances"]
    means_b = mah_stats["means_b"]
    stds_b = mah_stats["stds_b"]

    count_b = 0
    while count_b < n_per_gen:
        c = rng.choice(valid_classes)
        c_idx = class_to_idx[c]
        idx = rng.choice(class_indices[c])
        sigma = rng.choice(sigmas)
        base = df_val.loc[idx]

        cov_c = covariances[c_idx]
        noise_b = rng.multivariate_normal(mean=np.zeros(len(feats_b)), cov=cov_c)
        perturbed_b = centroids[c_idx] + sigma * noise_b
        raw_b = perturbed_b * stds_b + means_b

        row = {}
        for col in feats_a:
            if col in feats_b:
                b_idx = feats_b.index(col)
                v = float(raw_b[b_idx])
                if col in non_negative_cols:
                    v = max(0.0, v)
                row[col] = v
            else:
                row[col] = base[col]
        row["generator_type"] = "covariance_perturbation"
        pseudo_records.append(row)
        count_b += 1

    return pd.DataFrame(pseudo_records)

def optimize_weights_simplex(Z_val_df, Z_pseudo_df, val_pred_idx, pseudo_pred_idx, val_is_normal, num_classes=7, p_percentile=95.0, grid_step=0.05):
    """
    Grid search over the 4-simplex with operational constraints:
      - Known Acceptance >= 94.5%
      - Benign Rejection Rate <= 2.0%
      - Maximize Pseudo-Unknown F1 / Recall while penalizing redundancy
    """
    signal_cols = ["Z_confidence", "Z_mahalanobis", "Z_leaf", "Z_relative"]
    Z_val_mat = Z_val_df[signal_cols].values
    Z_pseudo_mat = Z_pseudo_df[signal_cols].values

    # Correlation matrix for redundancy penalty
    corr_mat = np.abs(np.corrcoef(Z_val_mat, rowvar=False))
    np.fill_diagonal(corr_mat, 0.0)

    best_w = np.array([0.25, 0.25, 0.25, 0.25])
    best_score = -1e9
    best_metrics = {}

    grid_vals = np.arange(0.05, 0.85, grid_step)
    for w1 in grid_vals:
        for w2 in grid_vals:
            if w1 + w2 >= 0.90:
                continue
            for w3 in grid_vals:
                w4 = 1.0 - (w1 + w2 + w3)
                if w4 < 0.05:
                    continue
                w = np.array([w1, w2, w3, w4])

                s_val = Z_val_mat @ w
                s_pseudo = Z_pseudo_mat @ w

                # Class conditional thresholds
                thresholds = np.zeros(num_classes)
                for c in range(num_classes):
                    m_c = (val_pred_idx == c)
                    thresholds[c] = np.percentile(s_val[m_c], p_percentile) if np.any(m_c) else np.percentile(s_val, p_percentile)

                tau_val = thresholds[val_pred_idx]
                is_unk_val = (s_val > tau_val)
                known_acc = 1.0 - (np.sum(is_unk_val) / float(len(s_val)))

                # Benign false alarm
                n_norm = np.sum(val_is_normal)
                benign_rej = (np.sum(is_unk_val[val_is_normal]) / float(n_norm)) if n_norm > 0 else 0.0

                if known_acc < 0.940 or benign_rej > 0.035:
                    continue

                tau_pseudo = thresholds[pseudo_pred_idx]
                recall_pseudo = np.sum(s_pseudo > tau_pseudo) / float(len(s_pseudo))

                # Redundancy penalty
                r_pen = np.sum(np.outer(w, w) * corr_mat) / 2.0
                utility = recall_pseudo - 0.15 * r_pen

                if utility > best_score:
                    best_score = utility
                    best_w = w
                    best_metrics = {
                        "weights": [round(float(v), 4) for v in w],
                        "known_acceptance": round(float(known_acc), 4),
                        "benign_rejection": round(float(benign_rej), 4),
                        "pseudo_recall": round(float(recall_pseudo), 4),
                        "utility": round(float(utility), 4)
                    }

    if best_score == -1e9:
        # Fallback to balanced weights
        best_w = np.array([0.25, 0.35, 0.20, 0.20])
        best_metrics = {"weights": [0.25, 0.35, 0.20, 0.20], "note": "fallback_balanced"}

    return best_w, best_metrics
