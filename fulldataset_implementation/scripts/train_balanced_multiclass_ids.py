#!/usr/bin/env python3
"""
train_balanced_multiclass_ids.py: High-Performance Multi-Class Intrusion Detection
Targets >99% Accuracy and Macro-F1 across all 8 classes on the UNSW Bot-IoT dataset.
Includes:
  - Balanced Class Weighting & Optional SMOTE / Random Oversampling
  - Rich Feature Engineering (packet ratios, log transformations, byte rates)
  - Gradient Boosted Trees (XGBoost) with tuned hyperparameters
  - Comprehensive Evaluation: Accuracy, Macro/Weighted F1, Per-Class Precision/Recall,
    Confusion Matrix, and ROC-AUC
  - Model serialization and confusion matrix visualization
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from common_utils import load_config, ensure_dirs, IMPL_DIR

def engineer_features(df, feature_cols):
    """Adds domain-specific networking features to enhance classifier discriminative power."""
    X = df[feature_cols].copy()
    
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

def run_multiclass_training():
    start_time = time.time()
    print("=" * 85)
    print(">>> STEP 2: HIGH-PERFORMANCE BALANCED MULTI-CLASS INTRUSION DETECTION <<<")
    print(">>> TARGET: >99.0% ACCURACY AND MACRO F1 ACROSS ALL ATTACK CLASSES <<<")
    print("=" * 85)

    ensure_dirs()
    cfg = load_config()
    cleaned_sample_path = cfg["paths"]["cleaned_sample_path"]
    models_dir = os.path.join(IMPL_DIR, "models")
    outputs_dir = os.path.join(IMPL_DIR, "outputs")
    plots_dir = os.path.join(outputs_dir, "plots")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(cleaned_sample_path):
        raise FileNotFoundError(f"Parquet missing at {cleaned_sample_path}. Run Step 1 stream cleaning first.")

    print(f"\n1. Loading representative dataset from: {cleaned_sample_path}")
    df = pd.read_parquet(cleaned_sample_path)
    print(f"Total flows loaded: {len(df):,}")

    # Inspect class counts
    class_counts = df["subcategory"].value_counts()
    print("\nClass distribution in loaded sample:")
    for cls, count in class_counts.items():
        print(f"  {cls:<20}: {count:>8,d} flows ({count/len(df)*100.0:>6.2f}%)")

    # Base features
    base_features = cfg["features"]["representation_a_xgboost"]
    print(f"\n2. Engineering network domain features from {len(base_features)} base features...")
    X_df = engineer_features(df, base_features)
    feature_names = list(X_df.columns)
    print(f"Total engineered features: {len(feature_names)}")

    # Encode labels
    classes = sorted(df["subcategory"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for i, c in enumerate(classes)}
    num_classes = len(classes)
    y = df["subcategory"].map(class_to_idx).values

    # Stratified Train/Test Split (80% train, 20% test)
    rand_seed = cfg["experiment"]["random_seed"]
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=0.20, stratify=y, random_state=rand_seed
    )
    print(f"\n3. Data Partitions: Train={len(X_train_df):,} | Test={len(X_test_df):,}")

    # Compute balanced class weights to give rare classes (Data_Exfiltration, Keylogging) equal voice
    classes_arr = np.arange(num_classes)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes_arr, y=y_train)
    weights_dict = {c: class_weights[c] for c in classes_arr}
    sample_weights_train = np.array([weights_dict[lbl] for lbl in y_train])

    print("\nClass weights computed for loss penalization:")
    for c_idx, c_name in idx_to_class.items():
        print(f"  Class {c_idx:2d} ({c_name:<20}): weight = {weights_dict[c_idx]:.4f} (train samples: {np.sum(y_train==c_idx):,})")

    # Train XGBoost Multi-Class Classifier
    print("\n4. Training High-Performance XGBoost Model (n_estimators=200, max_depth=8, hist)...")
    t_train_0 = time.time()
    xgb_params = {
        "n_estimators": 200,
        "max_depth": 8,
        "learning_rate": 0.08,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "tree_method": "hist",
        "objective": "multi:softprob",
        "num_class": num_classes,
        "random_state": rand_seed,
        "n_jobs": -1
    }

    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train_df, y_train, sample_weight=sample_weights_train)
    t_train = time.time() - t_train_0
    print(f"Training completed in {t_train:.2f} seconds!")

    # Save model and feature names
    model_save_path = os.path.join(models_dir, "xgboost_balanced_multiclass_ids.json")
    model.save_model(model_save_path)
    with open(os.path.join(models_dir, "multiclass_class_mapping.json"), "w") as f:
        json.dump({"class_to_idx": class_to_idx, "idx_to_class": idx_to_class, "features": feature_names}, f, indent=2)
    print(f"Model saved to: {model_save_path}")

    # Evaluation on Unseen Test Set
    print("\n5. Evaluating model on unseen test set (20% held-out)...")
    y_pred = model.predict(X_test_df)
    y_prob = model.predict_proba(X_test_df)

    test_acc = accuracy_score(y_test, y_pred) * 100.0
    macro_prec = precision_score(y_test, y_pred, average="macro", zero_division=0) * 100.0
    macro_rec = recall_score(y_test, y_pred, average="macro", zero_division=0) * 100.0
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0) * 100.0
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0) * 100.0
    ovr_auc = roc_auc_score(y_test, y_prob, multi_class="ovr") * 100.0

    print("\n" + "=" * 80)
    print(">>> OVERALL TEST EVALUATION METRICS <<<")
    print("=" * 80)
    print(f"  Overall Accuracy:          {test_acc:.4f}%")
    print(f"  Macro-Average Precision:   {macro_prec:.4f}%")
    print(f"  Macro-Average Recall:      {macro_rec:.4f}%")
    print(f"  Macro-Average F1-Score:    {macro_f1:.4f}%")
    print(f"  Weighted F1-Score:         {weighted_f1:.4f}%")
    print(f"  One-vs-Rest ROC-AUC:       {ovr_auc:.4f}%")
    print("=" * 80)

    # Detailed Per-Class Classification Report
    print("\n>>> DETAILED PER-CLASS PERFORMANCE BREAKDOWN <<<")
    report_dict = classification_report(y_test, y_pred, target_names=[idx_to_class[i] for i in range(num_classes)], output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    print(report_df.round(4).to_string())

    report_csv_path = os.path.join(outputs_dir, "balanced_multiclass_classification_report.csv")
    report_df.to_csv(report_csv_path)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = confusion_matrix(y_test, y_pred, normalize="true") * 100.0

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=[idx_to_class[i] for i in range(num_classes)],
        yticklabels=[idx_to_class[i] for i in range(num_classes)]
    )
    plt.title(f"Normalized Confusion Matrix (%) - Test Accuracy: {test_acc:.2f}%, Macro-F1: {macro_f1:.2f}%", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("True Class", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    cm_plot_path = os.path.join(plots_dir, "balanced_multiclass_confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"\nSaved confusion matrix plot to: {cm_plot_path}")

    # Feature Importance Plot
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1][:20]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), importance[sorted_idx][::-1], align="center", color="#1f77b4")
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx][::-1])
    plt.xlabel("Feature Importance Score", fontsize=11)
    plt.title("Top 20 Most Predictive Network Features for IoT Intrusion Detection", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fi_plot_path = os.path.join(plots_dir, "feature_importance_top20.png")
    plt.savefig(fi_plot_path, dpi=300)
    plt.close()
    print(f"Saved feature importance plot to: {fi_plot_path}")

    total_time = time.time() - start_time
    print(f"\nAll operations completed in {total_time:.2f}s ({total_time/60.0:.2f} minutes).")
    return {
        "accuracy": test_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "ovr_auc": ovr_auc
    }

if __name__ == "__main__":
    run_multiclass_training()
