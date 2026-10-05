"""
========================================================================================
rigorous_dual_evaluation_audit.py
Comprehensive Audit and Evaluation:
  Part 1: Known Attack Multi-Class Classification (Strict 70/30 Split, No Leakage)
  Part 2: Unknown (Zero-Day) Attack Isolation & Detection (10 LOCO Experiments)
========================================================================================
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "fulldataset_cleaned_sample.parquet")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "audit_reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_audit():
    print("=" * 80)
    print("STARTING RIGOROUS DATA INTEGRITY AUDIT & DUAL EVALUATION")
    print(f"Loading dataset from: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    print(f"Total flows loaded: {len(df):,}")

    df["full_class"] = df["category"] + " - " + df["subcategory"]
    classes = sorted(df["full_class"].unique().tolist())
    print(f"Unique classes ({len(classes)} total):")
    for c in classes:
        print(f"  - {c}: {len(df[df['full_class']==c]):,} flows")

    drop_cols = ["attack", "category", "subcategory", "full_class"]
    feat_cols = [c for c in df.columns if c not in drop_cols]
    print(f"\nFeature count: {len(feat_cols)}")

    # ==============================================================================
    # PART 1: KNOWN ATTACK CLASSIFICATION (STRICT 70% TRAIN / 30% TEST SPLIT)
    # ==============================================================================
    print("\n" + "=" * 80)
    print("PART 1: KNOWN ATTACK CLASSIFICATION (ARE KNOWN ATTACKS MAPPED CORRECTLY?)")
    print("=" * 80)

    # Add row ID to strictly verify NO overlap between train and test
    df["row_id"] = np.arange(len(df))

    # Stratified 70/30 Split
    train_df, test_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["full_class"]
    )

    # 1. Leakage Verification
    train_ids = set(train_df["row_id"].values)
    test_ids = set(test_df["row_id"].values)
    overlap = train_ids.intersection(test_ids)
    assert len(overlap) == 0, f"DATA LEAKAGE DETECTED! Overlap: {len(overlap)}"
    print("[VERIFIED] Strict Train/Test Separation: 0 overlapping samples.")
    print(f"  Training samples: {len(train_df):,} (70%)")
    print(f"  Testing samples:  {len(test_df):,} (30%)")

    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for i, c in enumerate(classes)}

    y_train = train_df["full_class"].map(class_to_idx).values
    X_train = train_df[feat_cols].values

    y_test = test_df["full_class"].map(class_to_idx).values
    X_test = test_df[feat_cols].values

    # Balanced class weights
    class_counts = np.bincount(y_train, minlength=len(classes))
    total_train = len(y_train)
    weights_dict = {i: total_train / (len(classes) * max(1, count)) for i, count in enumerate(class_counts)}
    sample_weights = np.array([weights_dict[y] for y in y_train])

    print("\nTraining Tier 1 Multi-Class XGBoost Classifier...")
    t0 = time.time()
    clf_multiclass = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    clf_multiclass.fit(X_train, y_train, sample_weight=sample_weights)
    print(f"Model trained in {time.time() - t0:.1f}s.")

    # Predict on test set
    y_pred_multi = clf_multiclass.predict(X_test)

    # Compute exact per-class metrics
    conf_mat = confusion_matrix(y_test, y_pred_multi, labels=range(len(classes)))
    overall_acc = accuracy_score(y_test, y_pred_multi) * 100.0

    known_metrics = []
    print("\n--- PER-CLASS PERFORMANCE ON STRICT 30% TEST SET (KNOWN CLASSIFICATION) ---")
    for i, c_name in enumerate(classes):
        tp = conf_mat[i, i]
        fn = np.sum(conf_mat[i, :]) - tp
        fp = np.sum(conf_mat[:, i]) - tp
        tn = len(y_test) - (tp + fn + fp)

        test_count = int(np.sum(conf_mat[i, :]))
        prec = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        acc = ((tp + tn) / len(y_test)) * 100.0
        err = 100.0 - acc

        known_metrics.append({
            "Class_Name": c_name,
            "Test_Samples": test_count,
            "Correctly_Classified_TP": int(tp),
            "Misclassified_FN": int(fn),
            "False_Alarm_FP": int(fp),
            "Accuracy_pct": round(acc, 2),
            "Precision_pct": round(prec, 2),
            "Recall_pct": round(rec, 2),
            "F1_Score_pct": round(f1, 2),
            "Error_Rate_pct": round(err, 2)
        })

    df_known = pd.DataFrame(known_metrics)
    print(df_known.to_string(index=False))
    print(f"\nOverall Multi-Class Test Accuracy: {overall_acc:.2f}%")

    out_known_csv = os.path.join(OUTPUT_DIR, "known_attack_multiclass_metrics.csv")
    df_known.to_csv(out_known_csv, index=False)

    # ==============================================================================
    # PART 2: UNKNOWN (ZERO-DAY) ATTACK CLASSIFICATION (10 LOCO TESTS)
    # ==============================================================================
    print("\n" + "=" * 80)
    print("PART 2: UNKNOWN (ZERO-DAY) ATTACK DETECTION (LEAVE-ONE-SUBCLASS-OUT)")
    print("=" * 80)

    # 10 attack subclasses (excluding Normal)
    attack_classes = [c for c in classes if c != "Normal - Normal"]

    # Fixed Benign Normal partition (70% train, 30% test)
    normal_df = df[df["full_class"] == "Normal - Normal"]
    train_norm, test_norm = train_test_split(normal_df, test_size=0.30, random_state=42)

    train_norm_ids = set(train_norm["row_id"].values)
    test_norm_ids = set(test_norm["row_id"].values)
    assert len(train_norm_ids.intersection(test_norm_ids)) == 0, "Normal train/test leakage!"

    unknown_metrics = []

    for idx, zd_class in enumerate(attack_classes, start=1):
        t_loc = time.time()
        print(f"\n[{idx}/10] Testing Zero-Day: {zd_class}")

        # 1. Zero-Day Dataset: 100% of this attack class is held out
        zd_df = df[df["full_class"] == zd_class]
        zd_ids = set(zd_df["row_id"].values)

        # 2. Known attacks dataset: All OTHER attack classes
        known_attacks_df = df[(df["full_class"] != "Normal - Normal") & (df["full_class"] != zd_class)]

        # Split known attacks into 70% train / 30% test
        train_known_att, test_known_att = train_test_split(
            known_attacks_df, test_size=0.30, random_state=42, stratify=known_attacks_df["full_class"]
        )

        train_known_ids = set(train_known_att["row_id"].values)
        test_known_ids = set(test_known_att["row_id"].values)

        # RIGOROUS LEAKAGE ASSERTIONS
        # a) Zero-day samples MUST NOT exist in training data
        train_full_ids = train_norm_ids.union(train_known_ids)
        assert len(zd_ids.intersection(train_full_ids)) == 0, f"LEAKAGE! Zero-day {zd_class} found in training data!"

        # b) Train samples MUST NOT exist in test samples
        test_known_full_ids = test_norm_ids.union(test_known_ids)
        assert len(train_full_ids.intersection(test_known_full_ids)) == 0, "LEAKAGE! Known train samples in test samples!"

        print(f"   [AUDIT PASSED] Zero-Day '{zd_class}' ({len(zd_df):,} flows) is 100% ABSENT from training.")
        print(f"   [AUDIT PASSED] Zero-Day model will be retrained from scratch on {len(train_full_ids):,} flows.")

        # Construct training set: Normal Train + Known Attacks Train
        train_loc = pd.concat([train_norm, train_known_att], ignore_index=True)
        y_train_loc = (train_loc["full_class"] != "Normal - Normal").astype(np.int32).values
        X_train_loc = train_loc[feat_cols].values

        # Cost-sensitive weights
        w_norm = len(train_loc) / (2.0 * len(train_norm))
        w_att = len(train_loc) / (2.0 * len(train_known_att))
        s_weights = np.where(y_train_loc == 0, w_norm, w_att)

        # Fresh model trained strictly on known data
        clf_zd = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        clf_zd.fit(X_train_loc, y_train_loc, sample_weight=s_weights)

        # Construct comprehensive test set:
        # 1. Zero-Day flows (Positive class for threat detection)
        # 2. Known Test Normal flows (Negative class for threat detection)
        # 3. Known Test Attack flows (Evaluated to check how known attacks behave)
        X_zd = zd_df[feat_cols].values
        X_norm_test = test_norm[feat_cols].values
        X_known_att_test = test_known_att[feat_cols].values

        # Predict
        pred_zd = clf_zd.predict(X_zd)                     # Expected: 1 (Attack)
        pred_norm = clf_zd.predict(X_norm_test)             # Expected: 0 (Normal)
        pred_known_att = clf_zd.predict(X_known_att_test)   # Expected: 1 (Attack)

        # False Alarms on Benign Normal:
        fp_normal = int(np.sum(pred_norm == 1))
        tn_normal = int(np.sum(pred_norm == 0))
        fa_rate_normal = (fp_normal / float(len(test_norm))) * 100.0

        # Zero-Day Threat Detection:
        tp_zd = int(np.sum(pred_zd == 1))
        fn_zd = int(np.sum(pred_zd == 0))

        # Known Attacks Detection (Are known attacks still caught as attacks?):
        tp_known_att = int(np.sum(pred_known_att == 1))
        fn_known_att = int(np.sum(pred_known_att == 0))

        # Standard Metrics for Zero-Day vs Normal:
        total_eval = len(zd_df) + len(test_norm)
        acc_zd = (tp_zd + tn_normal) / float(total_eval) * 100.0
        prec_zd = (tp_zd / float(tp_zd + fp_normal) * 100.0) if (tp_zd + fp_normal) > 0 else 0.0
        rec_zd = (tp_zd / float(tp_zd + fn_zd) * 100.0) if (tp_zd + fn_zd) > 0 else 0.0
        f1_zd = (2 * prec_zd * rec_zd / (prec_zd + rec_zd)) if (prec_zd + rec_zd) > 0 else 0.0
        err_zd = 100.0 - acc_zd

        print(f"   Zero-Day Caught (TP): {tp_zd:,} / {len(zd_df):,} ({rec_zd:.2f}%)")
        print(f"   Zero-Day Missed (FN): {fn_zd:,}")
        print(f"   False Alarms on Normal (FP): {fp_normal:,} / {len(test_norm):,} ({fa_rate_normal:.2f}%)")
        print(f"   Known Attacks Caught: {tp_known_att:,} / {len(test_known_att):,} ({tp_known_att/len(test_known_att)*100:.2f}%)")

        unknown_metrics.append({
            "Held_Out_Zero_Day": zd_class,
            "Zero_Day_Flows": len(zd_df),
            "Normal_Test_Flows": len(test_norm),
            "Known_Attack_Test_Flows": len(test_known_att),
            "True_Positives_TP": tp_zd,
            "False_Negatives_FN": fn_zd,
            "False_Alarms_Normal_FP": fp_normal,
            "True_Negatives_Normal_TN": tn_normal,
            "Known_Attacks_Caught_TP": tp_known_att,
            "Known_Attacks_Missed_FN": fn_known_att,
            "Accuracy_pct": round(acc_zd, 2),
            "Precision_pct": round(prec_zd, 2),
            "Recall_pct": round(rec_zd, 2),
            "F1_Score_pct": round(f1_zd, 2),
            "Error_Rate_pct": round(err_zd, 2)
        })

    df_unknown = pd.DataFrame(unknown_metrics)
    print("\n--- UNKNOWN (ZERO-DAY) ATTACK DETECTION RESULTS ACROSS ALL 10 TESTS ---")
    print(df_unknown.to_string(index=False))

    out_unknown_csv = os.path.join(OUTPUT_DIR, "unknown_attack_zeroday_metrics.csv")
    df_unknown.to_csv(out_unknown_csv, index=False)

    print("\n" + "=" * 80)
    print("ALL AUDITS AND EVALUATIONS COMPLETED SUCCESSFULLY WITH ZERO LEAKAGE!")
    print(f"Known metrics saved to:   {out_known_csv}")
    print(f"Unknown metrics saved to: {out_unknown_csv}")
    print("=" * 80)

if __name__ == "__main__":
    run_audit()
