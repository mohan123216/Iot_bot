"""
01_prepare_calibration.py: Integrity verification & strict zero-day quarantine audit
Step 10: Optimized Multi-Signal Novelty Fusion
"""

import os
import sys
import json
import pandas as pd
from common_utils import load_step10_config, get_abs_path, load_frozen_artifacts, ensure_output_dirs, STEP10_DIR

def run_prepare_calibration():
    print("=" * 80)
    print(">>> STEP 10: 01 - PREPARE CALIBRATION & INTEGRITY VERIFICATION <<<")
    print("=" * 80)

    ensure_output_dirs()
    cfg = load_step10_config()
    artifacts = load_frozen_artifacts()
    mapping = artifacts["mapping"]
    zero_day_class = mapping["zero_day_class"]
    known_classes = mapping["known_classes"]

    train_path = get_abs_path(cfg["paths"]["train_path"])
    val_path = get_abs_path(cfg["paths"]["validation_path"])
    known_test_path = get_abs_path(cfg["paths"]["known_test_path"])
    zd_test_path = get_abs_path(cfg["paths"]["zeroday_test_path"])

    print(f"Zero-Day Class Under Quarantine: '{zero_day_class}'")
    print(f"Known Classes ({len(known_classes)}): {known_classes}\n")

    # 1. Audit Train Set
    df_train = pd.read_parquet(train_path, columns=["subcategory"])
    zd_in_train = int((df_train["subcategory"] == zero_day_class).sum())
    print(f"Train Set ({len(df_train):,} flows): Service_Scan count = {zd_in_train}")
    assert zd_in_train == 0, f"FATAL LEAKAGE: {zero_day_class} found in train!"

    # 2. Audit Validation Set
    df_val = pd.read_parquet(val_path, columns=["subcategory"])
    zd_in_val = int((df_val["subcategory"] == zero_day_class).sum())
    print(f"Validation Set ({len(df_val):,} flows): Service_Scan count = {zd_in_val}")
    assert zd_in_val == 0, f"FATAL LEAKAGE: {zero_day_class} found in validation!"

    # 3. Audit Known Test Set
    df_known_test = pd.read_parquet(known_test_path, columns=["subcategory"])
    zd_in_known_test = int((df_known_test["subcategory"] == zero_day_class).sum())
    print(f"Known Test Set ({len(df_known_test):,} flows): Service_Scan count = {zd_in_known_test}")
    assert zd_in_known_test == 0, f"FATAL LEAKAGE: {zero_day_class} found in known_test!"

    # 4. Audit Held-Out Zero-Day Test Set
    df_zd_test = pd.read_parquet(zd_test_path, columns=["subcategory"])
    zd_count = int((df_zd_test["subcategory"] == zero_day_class).sum())
    non_zd_count = len(df_zd_test) - zd_count
    print(f"Held-Out Zero-Day Test Set ({len(df_zd_test):,} flows): Service_Scan count = {zd_count}, Non-{zero_day_class} count = {non_zd_count}")
    assert zd_count == 7302, f"FATAL: Expected exactly 7,302 flows of {zero_day_class}, got {zd_count}!"
    assert non_zd_count == 0, f"FATAL: Zero-day test contains non-{zero_day_class} traffic!"

    # 5. Model integrity check
    num_trees = artifacts["num_trees"]
    assert num_trees == 700, f"Expected 700 trees in frozen XGBoost model, got {num_trees}"
    print(f"\nFrozen XGBoost model verified: {num_trees} trees.")

    # Mahalanobis stats verification
    mah_stats = artifacts["mah_stats"]
    assert len(mah_stats["centroids"]) == 7, "Centroid count mismatch"
    assert len(mah_stats["precisions"]) == 7, "Precision matrices count mismatch"
    print(f"Mahalanobis Ledoit-Wolf models verified for {len(mah_stats['centroids'])} classes.")

    # Leaf profiles verification
    leaf_profiles = artifacts["leaf_profiles"]
    assert len(leaf_profiles) == 7, "Leaf profiles class count mismatch"
    for c_idx in range(7):
        assert "tree_freqs" in leaf_profiles[c_idx], f"tree_freqs missing in class {c_idx}"
        assert len(leaf_profiles[c_idx]["tree_freqs"]) == 700, f"Tree count mismatch in class {c_idx} leaf profile"
    print(f"Leaf-space empirical profiles verified: 7 classes x 700 trees with valid frequency distributions.")

    # Save verification report
    integrity_report = {
        "status": "PASSED",
        "zero_day_class": zero_day_class,
        "quarantine_verified": True,
        "dataset_counts": {
            "train": len(df_train),
            "validation": len(df_val),
            "known_test": len(df_known_test),
            "zeroday_test": len(df_zd_test)
        },
        "zero_day_counts": {
            "train": zd_in_train,
            "validation": zd_in_val,
            "known_test": zd_in_known_test,
            "zeroday_test": zd_count
        },
        "models": {
            "xgboost_trees": num_trees,
            "mahalanobis_classes": len(mah_stats["centroids"]),
            "leaf_classes": len(leaf_profiles)
        }
    }

    out_path = os.path.join(STEP10_DIR, "outputs", "calibration", "integrity_check.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(integrity_report, f, indent=2)

    print(f"\nIntegrity check PASSED. Saved report to: {out_path}")
    return integrity_report

if __name__ == "__main__":
    run_prepare_calibration()
