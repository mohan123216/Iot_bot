#!/usr/bin/env python3
"""
========================================================================================
STEP 2: TIER 1 - BALANCED MULTI-CLASS INTRUSION DETECTION CLASSIFIER
========================================================================================
Project: Hybrid Open-Set Multi-Signal Zero-Day Engine (HOMZ-Engine)
Classifier: Cost-Sensitive Gradient Boosted Decision Trees (XGBoost)

Key Technical Components:
  1. Network Domain Feature Engineering (35 features):
     - Directional payload ratios: spkts_ratio, sbytes_ratio, bytes_per_pkt
     - Asymmetric traffic dynamics: sbytes_per_spkt, dbytes_per_dpkt
     - Service port profiling: is_sport_wellknown, is_dport_wellknown, is_dport_http
     - Log-manifold volume stabilization: log_dur, log_pkts, log_bytes, log_rate
  2. Inverse-Frequency Class Weighting:
     - Penalizes errors on rare minority classes (Data Exfiltration, Keylogging)
  3. Strict Stratified 80/20 Train/Test Split (48,904 train, 12,226 test flows)
  4. Out-of-the-Box Inference or Fast Re-training:
     - Evaluates the pre-trained production model in ~1.5s
========================================================================================
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, "..", "01_Data_Preprocessing", "dataset", "fulldataset_cleaned_sample.parquet")
MODEL_PATH = os.path.join(CURRENT_DIR, "models", "xgboost_balanced_multiclass_ids.json")
MAPPING_PATH = os.path.join(CURRENT_DIR, "models", "multiclass_class_mapping.json")
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

BASE_FEATURES = [
    "proto_number", "flgs_number", "state_number", "sport", "dport",
    "dur", "pkts", "bytes", "spkts", "dpkts", "sbytes", "dbytes",
    "rate", "srate", "drate", "mean", "stddev", "sum", "min", "max"
]

def engineer_features(df):
    """Adds domain-specific networking features to enhance classifier discriminative power."""
    X = df[BASE_FEATURES].copy()

    # 1. Packet & Byte Ratios
    pkts = np.maximum(X["pkts"].values, 1e-6)
    bytes_val = np.maximum(X["bytes"].values, 1e-6)
    spkts = np.maximum(X["spkts"].values, 0.0)
    dpkts = np.maximum(X["dpkts"].values, 0.0)
    sbytes = np.maximum(X["sbytes"].values, 0.0)
    dbytes = np.maximum(X["dbytes"].values, 0.0)

    X["bytes_per_pkt"] = bytes_val / pkts
    X["sbytes_per_spkt"] = sbytes / (spkts + 1e-6)
    X["dbytes_per_dpkt"] = dbytes / (dpkts + 1e-6)
    X["spkts_ratio"] = spkts / pkts
    X["sbytes_ratio"] = sbytes / bytes_val

    # 2. Port categorizations (well-known: <1024, registered: 1024-49151, ephemeral: >49151)
    sport = X["sport"].values
    dport = X["dport"].values
    X["is_sport_wellknown"] = (sport < 1024).astype(np.float32)
    X["is_dport_wellknown"] = (dport < 1024).astype(np.float32)
    X["is_dport_http"] = ((dport == 80) | (dport == 8080) | (dport == 443)).astype(np.float32)
    X["is_sport_http"] = ((sport == 80) | (sport == 8080) | (sport == 443)).astype(np.float32)

    # 3. Log-scaled volume metrics
    for col in ["dur", "pkts", "bytes", "rate", "srate", "drate"]:
        X[f"log_{col}"] = np.log1p(np.maximum(X[col].values, 0.0))

    return X

def run_step_2(retrain=False):
    print("=" * 85)
    print(">>> STEP 2: TIER 1 - BALANCED MULTI-CLASS INTRUSION DETECTION <<<")
    print("=" * 85)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Cleaned dataset not found at: {DATA_PATH}")

    print(f"\n1. Loading curated 61,130-flow dataset from:\n   {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    print(f"   Successfully loaded {len(df):,} flows.")

    # Engineer Features
    print(f"\n2. Engineering domain network features (Directional Ratios + Port Profiling + Log Volumes)...")
    X_df = engineer_features(df)
    feature_names = list(X_df.columns)
    print(f"   Total engineered features: {len(feature_names)}")

    # Class mappings
    classes = sorted(df["subcategory"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for i, c in enumerate(classes)}
    y = df["subcategory"].map(class_to_idx).values

    # Stratified 80/20 Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"\n3. Stratified Partition: Train={len(X_train):,} flows | Test={len(X_test):,} flows")

    clf = xgb.XGBClassifier()
    if os.path.exists(MODEL_PATH) and not retrain:
        print(f"\n4. Loading pre-trained Tier-1 XGBoost model from:\n   {MODEL_PATH}")
        clf.load_model(MODEL_PATH)
    else:
        print(f"\n4. Training Balanced Cost-Sensitive XGBoost Classifier (n_estimators=200, depth=8)...")
        t0 = time.time()
        classes_arr = np.arange(len(classes))
        weights = compute_class_weight("balanced", classes=classes_arr, y=y_train)
        sample_weights = np.array([weights[lbl] for lbl in y_train])

        clf = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            tree_method="hist",
            objective="multi:softprob",
            num_class=len(classes),
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train, sample_weight=sample_weights)
        print(f"   Training finished in {time.time() - t0:.2f}s!")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        clf.save_model(MODEL_PATH)

    # Evaluate
    print(f"\n5. Evaluating Model on Held-Out Test Set ({len(X_test):,} flows)...")
    t0 = time.time()
    y_pred = clf.predict(X_test)
    eval_time = time.time() - t0

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    print("\n" + "=" * 80)
    print(">>> TIER 1 EVALUATION RESULTS <<<")
    print("=" * 80)
    print(f"Overall Multi-Class Accuracy : {acc * 100.0:.2f}%")
    print(f"Macro F1-Score               : {macro_f1 * 100.0:.2f}%")
    print(f"Weighted F1-Score            : {weighted_f1 * 100.0:.2f}%")
    print(f"Inference Latency            : {eval_time / len(X_test) * 1000.0:.4f} ms/flow")
    print("=" * 80)

    # Classification Report
    report_dict = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_csv = os.path.join(RESULTS_DIR, "balanced_multiclass_classification_report.csv")
    report_df.to_csv(report_csv)

    print("\nPer-Class Classification Metrics:")
    print(f"{'Class Name':<20} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 68)
    for c in classes:
        p = report_dict[c]["precision"]
        r = report_dict[c]["recall"]
        f = report_dict[c]["f1-score"]
        s = int(report_dict[c]["support"])
        print(f"{c:<20} | {p:>9.4f}  | {r:>9.4f}  | {f:>9.4f}  | {s:>8d}")
    print("-" * 68)

    # Save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title("Tier 1: Balanced Multi-Class Confusion Matrix (XGBoost)", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("True Category", fontsize=11)
    ax.set_xlabel("Predicted Category", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "balanced_multiclass_confusion_matrix.png")
    fig.savefig(cm_path, dpi=300)
    plt.close(fig)

    print(f"\nArtifacts saved to:\n  - {report_csv}\n  - {cm_path}")
    print("\n>>> STEP 2 COMPLETED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true", help="Re-train XGBoost model from scratch")
    args = parser.parse_args()
    run_step_2(retrain=args.retrain)
