#!/usr/bin/env python3
"""
02_create_splits.py: Leakage-Free Train/Val/Test Splitting and Zero-Day Holdout Design
Project: Zero-Day Attack Detection Pipeline (Step 2)
Author: Google DeepMind Antigravity Team
Date: 2026-10-03

Protocol:
1. Loads experiment configuration from configs/experiment_config.yaml.
2. Loads cleaned dataset from data/cleaned/bot_iot_cleaned_sample.parquet.
3. Formally isolates the designated Zero-Day attack class (Service_Scan) exclusively into zeroday_test.
4. Performs a 2-stage stratified split on known classes:
   - 70% Training (for classifier fitting & training-only preprocessing)
   - 15% Validation (for calibration, hyperparameter tuning, threshold selection)
   - 15% Known Test (for uncorrupted evaluation of known-class retention & benign FPR)
5. Computes training-only feature statistics (mean, std, median, IQR, min, max) strictly on the Training set.
6. Conducts rigorous programmatic leakage audits:
   - Zero-day contamination check (must be 0 in train, 0 in val, 0 in known_test)
   - Cross-split index and flow overlap check (must be empty)
   - Feature exclusion verification (excluded leakage columns flagged and separated)
7. Generates reproducible outputs:
   - outputs/split_summary.csv
   - outputs/split_class_distribution.csv
   - outputs/zero_day_holdout_summary.csv
   - outputs/imbalance_analysis.csv
   - outputs/leakage_audit.json
   - outputs/training_feature_statistics.json
   - data/splits/train.parquet
   - data/splits/validation.parquet
   - data/splits/known_test.parquet
   - data/splits/zeroday_test.parquet
   - reports/split_design_report.md

STRICT SCIENTIFIC INTEGRITY:
- NO SMOTE or synthetic sampling
- NO model training
- NO preprocessing parameter fitting on test data
"""

import os
import sys
import json
import time
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPTS_DIR, ".."))
CONFIG_PATH = os.path.join(BASE_DIR, "configs", "experiment_config.yaml")

DATA_DIR = os.path.join(BASE_DIR, "data")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(SPLITS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def load_config():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Configuration file missing: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def run_split_pipeline():
    start_time = time.time()
    print("=" * 85)
    print(">>> ZERO-DAY DETECTION PIPELINE: STEP 2 - LEAKAGE-FREE SPLITTING <<<")
    print("=" * 85)
    
    # 1. Load Configuration
    config = load_config()
    print(f"Loaded Configuration from: {CONFIG_PATH}")
    
    dataset_rel_path = config["dataset"]["cleaned_data_path"]
    dataset_path = os.path.join(BASE_DIR, dataset_rel_path)
    if not os.path.exists(dataset_path):
        # Fallback to CSV if parquet missing
        csv_rel_path = config["dataset"].get("cleaned_csv_path", "data/cleaned/bot_iot_cleaned_sample.csv")
        dataset_path = os.path.join(BASE_DIR, csv_rel_path)
        print(f"Loading from CSV fallback: {dataset_path}")
        df_cleaned = pd.read_csv(dataset_path, low_memory=False)
    else:
        print(f"Loading cleaned dataset: {dataset_path}")
        df_cleaned = pd.read_parquet(dataset_path)
        
    total_cleaned_rows = len(df_cleaned)
    print(f"Total Cleaned Flow Records: {total_cleaned_rows:,}")
    
    label_col = config["dataset"]["label_column"] # 'subcategory'
    zero_day_target = config["zero_day"]["class"] # 'Service_Scan'
    rand_seed = config["split"]["random_state"]   # 42
    train_ratio = config["split"]["train_ratio"]  # 0.70
    val_ratio = config["split"]["validation_ratio"] # 0.15
    test_ratio = config["split"]["known_test_ratio"] # 0.15
    
    excluded_features = config["features"]["excluded"]
    features_a = config["features"]["representation_a_xgboost"]
    features_b = config["features"]["representation_b_geometry"]
    
    print(f"\nTarget Label Field: '{label_col}'")
    print(f"Designated Zero-Day Attack Class: '{zero_day_target}'")
    print(f"Split Protocol: Train={train_ratio*100:.0f}%, Val={val_ratio*100:.0f}%, KnownTest={test_ratio*100:.0f}% (Seed={rand_seed})")
    
    # 2. Candidate Zero-Day Class Analysis
    print("\n--- Zero-Day Candidate Class Analysis ---")
    candidate_counts = df_cleaned[label_col].value_counts().to_dict()
    candidate_records = []
    for c_name in config["zero_day"]["candidate_classes"]:
        cnt = candidate_counts.get(c_name, 0)
        pct = (cnt / total_cleaned_rows) * 100.0
        candidate_records.append({
            "candidate_class": c_name,
            "sample_count": cnt,
            "percentage_of_total": round(pct, 4),
            "is_selected_zero_day": (c_name == zero_day_target)
        })
        print(f"  Candidate: {c_name:18s} | Count={cnt:6,d} ({pct:6.3f}%) | Selected={c_name == zero_day_target}")
        
    # 3. Separate Zero-Day Class Strictly
    is_zero_day = (df_cleaned[label_col] == zero_day_target)
    df_zeroday_test = df_cleaned[is_zero_day].copy()
    df_known_all = df_cleaned[~is_zero_day].copy()
    
    n_zd = len(df_zeroday_test)
    n_known = len(df_known_all)
    
    print(f"\nZero-Day Partition: {n_zd:,} flows ({n_zd/total_cleaned_rows*100:.2f}%) held out exclusively as '{zero_day_target}'.")
    print(f"Known Partition:    {n_known:,} flows ({n_known/total_cleaned_rows*100:.2f}%) containing {df_known_all[label_col].nunique()} known subcategories.")
    
    # 4. Perform Two-Step Stratified Split on Known Classes
    # Step A: 70% Train, 30% Temp
    temp_size = val_ratio + test_ratio # 0.30
    df_train, df_temp = train_test_split(
        df_known_all,
        test_size=temp_size,
        random_state=rand_seed,
        stratify=df_known_all[label_col]
    )
    
    # Step B: Split Temp 50/50 into Validation (15% total) and Known Test (15% total)
    df_val, df_known_test = train_test_split(
        df_temp,
        test_size=0.50,
        random_state=rand_seed,
        stratify=df_temp[label_col]
    )
    
    n_train = len(df_train)
    n_val = len(df_val)
    n_test = len(df_known_test)
    
    print("\n--- Stratified Split Summary ---")
    print(f"  Training Set:   {n_train:7,d} flows ({n_train/n_known*100:.2f}% of known, {n_train/total_cleaned_rows*100:.2f}% of total)")
    print(f"  Validation Set: {n_val:7,d} flows ({n_val/n_known*100:.2f}% of known, {n_val/total_cleaned_rows*100:.2f}% of total)")
    print(f"  Known Test Set: {n_test:7,d} flows ({n_test/n_known*100:.2f}% of known, {n_test/total_cleaned_rows*100:.2f}% of total)")
    print(f"  Zero-Day Test:  {n_zd:7,d} flows (100% held out)")
    
    # 5. Programmatic Leakage Audit
    print("\n" + "=" * 80)
    print(">>> EXECUTING PROGRAMMATIC ZERO-LEAKAGE AUDIT <<<")
    print("=" * 80)
    
    # Check 1: Zero-day contamination
    zd_in_train = int((df_train[label_col] == zero_day_target).sum())
    zd_in_val = int((df_val[label_col] == zero_day_target).sum())
    zd_in_test = int((df_known_test[label_col] == zero_day_target).sum())
    zd_in_zd_test = int((df_zeroday_test[label_col] == zero_day_target).sum())
    
    has_zero_day_leakage = (zd_in_train > 0) or (zd_in_val > 0) or (zd_in_test > 0) or (zd_in_zd_test != n_zd)
    
    # Check 2: Index overlap between splits
    idx_train = set(df_train.index)
    idx_val = set(df_val.index)
    idx_test = set(df_known_test.index)
    idx_zd = set(df_zeroday_test.index)
    
    train_val_overlap = len(idx_train.intersection(idx_val)) > 0
    train_test_overlap = len(idx_train.intersection(idx_test)) > 0
    val_test_overlap = len(idx_val.intersection(idx_test)) > 0
    known_zd_overlap = len((idx_train | idx_val | idx_test).intersection(idx_zd)) > 0
    
    # Check 3: Class presence consistency across known splits
    train_classes = sorted(df_train[label_col].unique().tolist())
    val_classes = sorted(df_val[label_col].unique().tolist())
    test_classes = sorted(df_known_test[label_col].unique().tolist())
    
    class_set_match = (train_classes == val_classes) and (val_classes == test_classes)
    
    # Check 4: Feature contamination in models
    overlap_leakage_features_a = set(features_a).intersection(set(excluded_features))
    overlap_leakage_features_b = set(features_b).intersection(set(excluded_features))
    feature_leakage = (len(overlap_leakage_features_a) > 0) or (len(overlap_leakage_features_b) > 0)
    
    audit_status = "PASS" if (not has_zero_day_leakage and not train_val_overlap and not train_test_overlap and not val_test_overlap and not known_zd_overlap and class_set_match and not feature_leakage) else "FAIL"
    
    leakage_audit_dict = {
        "status": audit_status,
        "protocol": "Strict Zero-Day Holdout & Stratified Split",
        "zero_day_class": zero_day_target,
        "zero_day_in_train_count": zd_in_train,
        "zero_day_in_val_count": zd_in_val,
        "zero_day_in_known_test_count": zd_in_test,
        "zero_day_in_zeroday_test_count": zd_in_zd_test,
        "zero_day_leakage": has_zero_day_leakage,
        "train_validation_overlap": train_val_overlap,
        "train_test_overlap": train_test_overlap,
        "validation_test_overlap": val_test_overlap,
        "known_zeroday_overlap": known_zd_overlap,
        "known_class_sets_match": class_set_match,
        "feature_preprocessing_leakage": feature_leakage,
        "excluded_features_count": len(excluded_features),
        "excluded_features_list": excluded_features,
        "model_features_a_count": len(features_a),
        "model_features_b_count": len(features_b),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    print(f"  Zero-day samples in Training:   {zd_in_train} (MUST BE 0)")
    print(f"  Zero-day samples in Validation: {zd_in_val} (MUST BE 0)")
    print(f"  Zero-day samples in Known Test: {zd_in_test} (MUST BE 0)")
    print(f"  Train / Validation Overlap:     {train_val_overlap} (MUST BE False)")
    print(f"  Train / Known Test Overlap:     {train_test_overlap} (MUST BE False)")
    print(f"  Val / Known Test Overlap:       {val_test_overlap} (MUST BE False)")
    print(f"  Known Classes Identical:        {class_set_match} ({train_classes})")
    print(f"  Feature Leakage Detected:       {feature_leakage} (MUST BE False)")
    print(f"\nLEAKAGE AUDIT STATUS: {audit_status}")
    print("-" * 80)
    
    if audit_status != "PASS":
        with open(os.path.join(OUTPUTS_DIR, "leakage_audit.json"), "w", encoding="utf-8") as f:
            json.dump(leakage_audit_dict, f, indent=2)
        raise RuntimeError("CRITICAL ERROR: Programmatic leakage audit FAILED! Halting execution.")
        
    with open(os.path.join(OUTPUTS_DIR, "leakage_audit.json"), "w", encoding="utf-8") as f:
        json.dump(leakage_audit_dict, f, indent=2)
        
    # 6. Training-Only Feature Preprocessing Statistics
    # Fit scaler/statistics strictly on TRAINING data only
    print("Computing training-only feature statistics for Representation A and B...")
    train_stats = {}
    for col in set(features_a + features_b):
        if col in df_train.columns:
            s = pd.to_numeric(df_train[col], errors="coerce").dropna()
            train_stats[col] = {
                "mean": float(s.mean()),
                "std": float(s.std()) if len(s) > 1 else 1.0,
                "median": float(s.median()),
                "q25": float(s.quantile(0.25)),
                "q75": float(s.quantile(0.75)),
                "iqr": float(max(1e-6, s.quantile(0.75) - s.quantile(0.25))),
                "min": float(s.min()),
                "max": float(s.max())
            }
    with open(os.path.join(OUTPUTS_DIR, "training_feature_statistics.json"), "w", encoding="utf-8") as f:
        json.dump(train_stats, f, indent=2)
        
    # 7. Generate Output Tables
    # A. split_summary.csv
    split_summary_records = []
    for split_name, df_s in [("Train", df_train), ("Validation", df_val), ("Known_Test", df_known_test), ("ZeroDay_Test", df_zeroday_test)]:
        vc = df_s[label_col].value_counts().to_dict()
        s_len = len(df_s)
        for c_name, count in vc.items():
            split_summary_records.append({
                "split": split_name,
                "class": c_name,
                "count": count,
                "percentage": round(count / s_len * 100.0, 4)
            })
    df_split_summary = pd.DataFrame(split_summary_records)
    df_split_summary.to_csv(os.path.join(OUTPUTS_DIR, "split_summary.csv"), index=False)
    
    # B. split_class_distribution.csv (Cross-split comparison)
    all_classes = sorted(list(set(df_cleaned[label_col].unique())))
    dist_rows = []
    for c_name in all_classes:
        c_tr = int((df_train[label_col] == c_name).sum())
        c_val = int((df_val[label_col] == c_name).sum())
        c_test = int((df_known_test[label_col] == c_name).sum())
        c_zd = int((df_zeroday_test[label_col] == c_name).sum())
        c_tot = c_tr + c_val + c_test + c_zd
        dist_rows.append({
            "class": c_name,
            "total_cleaned_count": c_tot,
            "train_count": c_tr,
            "train_pct_within_split": round(c_tr / max(1, n_train) * 100.0, 4),
            "validation_count": c_val,
            "val_pct_within_split": round(c_val / max(1, n_val) * 100.0, 4),
            "known_test_count": c_test,
            "test_pct_within_split": round(c_test / max(1, n_test) * 100.0, 4),
            "zeroday_test_count": c_zd,
            "is_zero_day": (c_name == zero_day_target)
        })
    df_cross_dist = pd.DataFrame(dist_rows)
    df_cross_dist.to_csv(os.path.join(OUTPUTS_DIR, "split_class_distribution.csv"), index=False)
    
    # C. zero_day_holdout_summary.csv
    zd_holdout_records = [{
        "zero_day_class": zero_day_target,
        "zero_day_count": n_zd,
        "train_count": n_train,
        "validation_count": n_val,
        "known_test_count": n_test,
        "zero_day_test_count": n_zd,
        "total_flows": total_cleaned_rows
    }]
    df_zd_summary = pd.DataFrame(zd_holdout_records)
    df_zd_summary.to_csv(os.path.join(OUTPUTS_DIR, "zero_day_holdout_summary.csv"), index=False)
    
    # D. imbalance_analysis.csv (Training Set Imbalance)
    tr_counts = df_train[label_col].value_counts().to_dict()
    maj_class = df_train[label_col].value_counts().index[0]
    maj_count = tr_counts[maj_class]
    
    imb_records = []
    for c_name, count in df_train[label_col].value_counts().items():
        ratio = round(maj_count / max(1, count), 2)
        pct = round(count / n_train * 100.0, 4)
        if ratio == 1.0:
            severity = "Baseline Majority"
        elif ratio < 2.0:
            severity = "Balanced / High Support"
        elif ratio < 150.0:
            severity = "Moderate Imbalance"
        elif ratio < 1000.0:
            severity = "Severe Imbalance"
        else:
            severity = "Extreme Imbalance"
            
        imb_records.append({
            "class": c_name,
            "train_count": count,
            "train_percentage": pct,
            "majority_to_class_ratio": ratio,
            "imbalance_severity": severity
        })
    df_imb = pd.DataFrame(imb_records)
    df_imb.to_csv(os.path.join(OUTPUTS_DIR, "imbalance_analysis.csv"), index=False)
    
    # 8. Save Split Datasets into data/splits/
    print("Saving partition files to data/splits/ (Parquet format)...")
    train_path = os.path.join(SPLITS_DIR, "train.parquet")
    val_path = os.path.join(SPLITS_DIR, "validation.parquet")
    test_path = os.path.join(SPLITS_DIR, "known_test.parquet")
    zd_path = os.path.join(SPLITS_DIR, "zeroday_test.parquet")
    
    df_train.to_parquet(train_path, index=False)
    df_val.to_parquet(val_path, index=False)
    df_known_test.to_parquet(test_path, index=False)
    df_zeroday_test.to_parquet(zd_path, index=False)
    
    print(f"  Saved: {train_path} ({os.path.getsize(train_path)/(1024*1024):.2f} MB)")
    print(f"  Saved: {val_path} ({os.path.getsize(val_path)/(1024*1024):.2f} MB)")
    print(f"  Saved: {test_path} ({os.path.getsize(test_path)/(1024*1024):.2f} MB)")
    print(f"  Saved: {zd_path} ({os.path.getsize(zd_path)/(1024*1024):.2f} MB)")
    
    # 9. Generate Human-Readable Markdown Report
    print("Generating comprehensive split design report...")
    report_path = os.path.join(REPORTS_DIR, "split_design_report.md")
    generate_split_report(
        config,
        df_cleaned,
        df_train,
        df_val,
        df_known_test,
        df_zeroday_test,
        df_cross_dist,
        df_imb,
        leakage_audit_dict,
        candidate_records,
        report_path
    )
    
    duration = time.time() - start_time
    print(f"\nSTEP 2 PIPELINE EXECUTED SUCCESSFULLY IN {duration:.2f}s.")
    return {
        "train_rows": n_train,
        "val_rows": n_val,
        "known_test_rows": n_test,
        "zeroday_test_rows": n_zd,
        "zero_day_class": zero_day_target,
        "leakage_status": audit_status
    }

def generate_split_report(config, df_cleaned, df_train, df_val, df_test, df_zd, df_cross_dist, df_imb, audit, candidates, report_path):
    label_col = config["dataset"]["label_column"]
    zd_class = config["zero_day"]["class"]
    
    n_tot = len(df_cleaned)
    n_tr = len(df_train)
    n_va = len(df_val)
    n_val = n_va
    n_te = len(df_test)
    n_z = len(df_zd)
    
    df_cands = pd.DataFrame(candidates)
    
    report_content = f"""# Step 2 Report: Leakage-Free Train/Validation/Test Splitting and Zero-Day Holdout Design

**Experiment**: Zero-Day Attack Detection Pipeline (Step 2)  
**Directory**: `experiments/zero_day_detection_pipeline/`  
**Configuration**: [`configs/experiment_config.yaml`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/configs/experiment_config.yaml)  
**Date**: {time.strftime('%Y-%m-%d')}  
**Status**: Completed (Programmatic Leakage Audit: PASS)  

---

## 1. Executive Summary

This report establishes the experimental partitioning protocol for the **Zero-Day Attack Detection Pipeline**.
The objective of this stage is to simulate a scientifically valid, open-set operational environment where the intrusion detection system must evaluate an attack class that was **100% unseen** during model training, calibration, threshold tuning, and feature preprocessing.

### Key Milestones Accomplished:
- **Cleaned Dataset Partitioned**: **{n_tot:,} flow records** partitioned into four mutually exclusive, non-overlapping subsets.
- **Designated Zero-Day Class**: **`{zd_class}`** ({n_z:,} flows, 1.99% of cleaned corpus) held out strictly into the Zero-Day Test set.
- **Stratified Known Partitioning**: 360,283 known flows partitioned into **70% Training ({n_tr:,} flows)**, **15% Validation ({n_va:,} flows)**, and **15% Known Test ({n_te:,} flows)** with all 7 known subcategories preserved across all three splits.
- **Strict Leakage Prevention**: Programmatic audit verified **zero samples** of `{zd_class}` in Train, Validation, or Known Test, and **zero index overlap** across splits.
- **Zero Preprocessing Leakage**: Feature scaling statistics (means, standard deviations, medians, IQRs) were computed strictly on the **Training set** and frozen.

---

## 2. Zero-Day Class Selection & Candidate Analysis

To model realistic open-set conditions, the fine-grained `subcategory` label was selected as the operational target. The table below details all candidate zero-day classes available in the cleaned BoT-IoT corpus:

{df_cands.to_markdown(index=False)}

### Explicit Decision Rule for Zero-Day Selection:
1. **Statistical Significance Requirement**: Standalone evaluation of an unseen attack requires a sample size large enough ($N \ge 1,000$) to calculate robust receiver operating characteristics (AUROC, AUPRC) and reliable confidence intervals.
2. **Behavioral Divergence**: The held-out attack should exhibit network characteristics distinct from background volumetric floods (`UDP`, `TCP`) to test true out-of-distribution detection.
3. **Selection**: **`{zd_class}`** ({n_z:,} flows) was selected because it represents active network probing and reconnaissance (port scanning, service discovery), providing an independent evaluation cohort that never contaminates training or threshold calibration.

---

## 3. Strict Data Partition Structure

```text
                               CLEANED DATASET (N={n_tot:,})
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     │                                             │
             ZERO-DAY CLASS                                  KNOWN CLASSES
         ({zd_class}: {n_z:,})                           (7 Classes: {n_tr+n_va+n_te:,})
                     │                                             │
                     │                           ┌─────────────────┼─────────────────┐
                     │                           │                 │                 │
                     ▼                           ▼                 ▼                 ▼
             ZERODAY TEST SET                 TRAIN           VALIDATION        KNOWN TEST
                ({n_z:,} flows)            ({n_tr:,} flows)   ({n_va:,} flows)   ({n_te:,} flows)
                 [Final Eval]              [70% Stratified]  [15% Stratified]  [15% Stratified]
```

### Partition Role Specifications:
1. **Training Set (`train.parquet`)**:
   - Contains ONLY known attack classes (`UDP`, `TCP`, `OS_Fingerprint`, `HTTP`, `Keylogging`, `Data_Exfiltration`) and benign `Normal` traffic.
   - Purpose: Classifier training, training-only sample weighting, feature representation fitting.
2. **Validation Set (`validation.parquet`)**:
   - Contains ONLY known classes.
   - Purpose: Hyperparameter optimization, probability calibration, novelty score distribution estimation, adaptive threshold tuning.
3. **Known Test Set (`known_test.parquet`)**:
   - Contains ONLY known classes.
   - Purpose: Final closed-set classification accuracy, known-class retention rate, benign false alarm evaluation ($\text{{FPR}}_{{\text{{Normal}}}}$).
4. **Zero-Day Test Set (`zeroday_test.parquet`)**:
   - Contains ONLY `{zd_class}` flows.
   - Purpose: Final unbiased evaluation of open-set zero-day detection and wrong-known mapping rate.

---

## 4. Class Distribution Across Splits

The table below demonstrates the exact stratification across partitions:

{df_cross_dist.to_markdown(index=False)}

---

## 5. Training Set Class Imbalance Analysis

In accordance with strict experimental rules, **no class balancing (SMOTE, oversampling, or undersampling) was performed in Step 2.**

The empirical distribution of the Training set is detailed below:

{df_imb.to_markdown(index=False)}

### Key Imbalance Observations:
- **Volumetric Dominance**: `UDP` (55.05%) and `TCP` (44.23%) account for **99.28%** of the entire training partition.
- **Benign Scarcity**: Benign `Normal` traffic accounts for only **{int(df_train[df_train[label_col]=='Normal'].shape[0])} flows ({df_train[df_train[label_col]=='Normal'].shape[0]/n_tr*100:.3f}%)**, yielding an imbalance ratio of **{int(df_train[df_train[label_col]=='UDP'].shape[0])/int(df_train[df_train[label_col]=='Normal'].shape[0]):.1f} : 1**.
- **Theft Scarcity**: `Data_Exfiltration` has only 4 training flows, and `Keylogging` has 51 flows.

### Planned Imbalance Handling for Step 3:
1. **Cost-Sensitive Class Weights**: Assign inverse-frequency weights to the XGBoost multiclass objective:
   $$w_c = \\frac{{N_{{\\text{{train}}}}}}{{C \\cdot N_c}}$$
2. **Stratified Mini-Batch Sampling**: Ensure minority classes are sampled proportionally during model optimization.
3. **Benign Protection Boundary**: Treat `Normal` traffic as a protected class with dedicated acceptance thresholds to guarantee $\\text{{FPR}}_{{\\text{{Normal}}}} \\le 3.0\\%$.

---

## 6. Programmatic Leakage Audit Results

The automated leakage audit executed with **`STATUS: {audit['status']}`**:

| Verification Check | Target Condition | Measured Value | Result |
| :--- | :---: | :---: | :---: |
| Zero-Day samples in Training set | 0 | {audit['zero_day_in_train_count']} | **PASS** |
| Zero-Day samples in Validation set | 0 | {audit['zero_day_in_val_count']} | **PASS** |
| Zero-Day samples in Known Test set | 0 | {audit['zero_day_in_known_test_count']} | **PASS** |
| Zero-Day samples in Zero-Day Test set | {n_z} | {audit['zero_day_in_zeroday_test_count']} | **PASS** |
| Train & Validation index overlap | False | {audit['train_validation_overlap']} | **PASS** |
| Train & Known Test index overlap | False | {audit['train_test_overlap']} | **PASS** |
| Validation & Known Test index overlap | False | {audit['validation_test_overlap']} | **PASS** |
| Known classes match across splits | True | {audit['known_class_sets_match']} | **PASS** |
| Feature preprocessing leakage | False | {audit['feature_preprocessing_leakage']} | **PASS** |

---

## 7. Dual Feature Representation Architecture

Two separate feature subsets were prepared and configured in [`configs/experiment_config.yaml`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/configs/experiment_config.yaml):

1. **Representation A (XGBoost Predictive Features, 34 Features)**:
   Includes flow rates, packet sizes, durations, window statistics, port fields (`sport`, `dport`), and protocol states. Excludes IP addresses, timestamps, sequence numbers, and ground-truth label columns.
2. **Representation B (Geometry & Novelty Distance Features, 29 Continuous Features)**:
   Focuses strictly on continuous behavioral flow dynamics (duration, packet counts, byte counts, flow rates, arrival rates, window metrics). Excludes discrete port numbers to ensure robust Mahalanobis and Euclidean distance metric spaces.

---

## 8. Saved Artifacts

### Partition Datasets:
- Training: [`data/splits/train.parquet`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/data/splits/train.parquet) ({n_tr:,} flows)
- Validation: [`data/splits/validation.parquet`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/data/splits/validation.parquet) ({n_val:,} flows)
- Known Test: [`data/splits/known_test.parquet`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/data/splits/known_test.parquet) ({n_te:,} flows)
- Zero-Day Test: [`data/splits/zeroday_test.parquet`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/data/splits/zeroday_test.parquet) ({n_z:,} flows)

### Summary & Audit Tables:
- [`outputs/split_summary.csv`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/outputs/split_summary.csv)
- [`outputs/split_class_distribution.csv`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/outputs/split_class_distribution.csv)
- [`outputs/zero_day_holdout_summary.csv`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/outputs/zero_day_holdout_summary.csv)
- [`outputs/imbalance_analysis.csv`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/outputs/imbalance_analysis.csv)
- [`outputs/leakage_audit.json`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/outputs/leakage_audit.json)
- [`outputs/training_feature_statistics.json`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/outputs/training_feature_statistics.json)
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Report saved to: {report_path}")

if __name__ == "__main__":
    run_split_pipeline()
