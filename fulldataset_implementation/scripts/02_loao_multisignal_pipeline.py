#!/usr/bin/env python3
"""
02_loao_multisignal_pipeline.py: Leave-One-Attack-Out (LOAO) Multi-Signal Novelty Detection
Trains XGBoost, Ledoit-Wolf Mahalanobis, and Leaf-Space profiles for each held-out attack.
Calibrates empirical CDF normalizers, optimizes simplex weights, and evaluates all metrics.
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import time
import json
import pickle
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, precision_recall_fscore_support
import xgboost as xgb

from common_utils import (
    load_config, ensure_dirs, EmpiricalCDFNormalizer,
    compute_all_novelty_signals, extract_mahalanobis_statistics,
    extract_leaf_profiles, synthesize_pseudo_unknowns,
    optimize_weights_simplex, IMPL_DIR
)

def run_loao_pipeline():
    start_time = time.time()
    print("=" * 85)
    print(">>> FULL DATASET IMPLEMENTATION: STEP 2 - LEAVE-ONE-ATTACK-OUT 4-SIGNAL PIPELINE <<<")
    print("=" * 85)

    ensure_dirs()
    cfg = load_config()
    feats_a = cfg["features"]["representation_a_xgboost"]
    feats_b = cfg["features"]["representation_b_geometry"]
    rand_seed = cfg["experiment"]["random_seed"]
    primary_pct = cfg["thresholds"]["primary_percentile"]
    cand_percentiles = cfg["thresholds"]["candidate_percentiles"]

    cleaned_sample_path = cfg["paths"]["cleaned_sample_path"]
    if not os.path.exists(cleaned_sample_path):
        raise FileNotFoundError(f"Cleaned sample parquet missing at {cleaned_sample_path}. Run 01_stream_clean_full_dataset.py first.")

    print(f"Loading cleaned full-dataset representative sample from: {cleaned_sample_path}")
    df_all = pd.read_parquet(cleaned_sample_path)
    print(f"Total loaded flows: {len(df_all):,}")

    # Identify all classes
    all_classes = sorted(df_all["subcategory"].unique().tolist())
    attack_classes = [c for c in all_classes if c != "Normal"]

    print(f"\nAll Classes ({len(all_classes)}): {all_classes}")
    print(f"Attack Classes to evaluate as Zero-Day ({len(attack_classes)}): {attack_classes}")

    master_results = []
    weights_summary = []
    thresholds_summary = []

    # Iterate through every attack class as the unseen zero-day
    for att_idx, heldout_attack in enumerate(attack_classes):
        t_att_0 = time.time()
        print("\n" + "=" * 80)
        print(f"[{att_idx+1}/{len(attack_classes)}] EVALUATING ZERO-DAY ATTACK: {heldout_attack.upper()}")
        print("=" * 80)

        # 1. Quarantine Held-Out Attack
        df_zd = df_all[df_all["subcategory"] == heldout_attack].copy()
        df_known = df_all[df_all["subcategory"] != heldout_attack].copy()

        n_zd = len(df_zd)
        n_known = len(df_known)
        print(f"  Quarantined Zero-Day Flows ({heldout_attack}): {n_zd:,}")
        print(f"  Available Known Flows (other 6 attacks + Normal): {n_known:,}")

        known_classes = sorted(df_known["subcategory"].unique().tolist())
        class_to_idx = {c: i for i, c in enumerate(known_classes)}
        idx_to_class = {i: c for i, c in enumerate(known_classes)}
        num_classes = len(known_classes)

        # 2. Stratified 70/15/15 Split on Known Classes
        # First split 70% train vs 30% temp
        train_df, temp_df = train_test_split(
            df_known,
            test_size=0.30,
            stratify=df_known["subcategory"],
            random_state=rand_seed
        )
        # Second split 30% temp into 15% val and 15% test
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.50,
            stratify=temp_df["subcategory"],
            random_state=rand_seed
        )

        print(f"  Known Partitions -> Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

        # Leakage Verification
        assert (train_df["subcategory"] == heldout_attack).sum() == 0, f"LEAKAGE: {heldout_attack} in train!"
        assert (val_df["subcategory"] == heldout_attack).sum() == 0, f"LEAKAGE: {heldout_attack} in val!"
        assert (test_df["subcategory"] == heldout_attack).sum() == 0, f"LEAKAGE: {heldout_attack} in test!"
        assert (df_zd["subcategory"] != heldout_attack).sum() == 0, f"CORRUPTION in zero-day partition!"

        # 3. Train Closed-Set Multi-Class XGBoost Model
        y_train = train_df["subcategory"].map(class_to_idx).values
        X_train_a = train_df[feats_a].values

        # Compute balanced class weights
        class_counts = Counter(y_train)
        total_samples = len(y_train)
        weights_map = {c: total_samples / (num_classes * count) for c, count in class_counts.items()}
        sample_weights = np.array([weights_map[y] for y in y_train])

        xgb_params = cfg["xgboost_params"].copy()
        xgb_params["num_class"] = num_classes

        print("  Training XGBoost Multi-Class Baseline...")
        clf = xgb.XGBClassifier(**xgb_params)
        clf.fit(X_train_a, y_train, sample_weight=sample_weights)
        booster = clf.get_booster()

        # Save model
        model_save_path = os.path.join(IMPL_DIR, "models", f"xgboost_heldout_{heldout_attack}.json")
        clf.save_model(model_save_path)

        # 4. Compute Mahalanobis Geometry & Leaf Profiles on Train Data
        print("  Fitting Ledoit-Wolf precision matrices and leaf-space traversal profiles...")
        mah_stats = extract_mahalanobis_statistics(train_df, feats_b, known_classes, class_to_idx)
        leaf_profiles = extract_leaf_profiles(train_df, booster, feats_a, known_classes, class_to_idx)

        # 5. Compute Raw Signals & Empirical CDF Calibration on Validation Data
        print("  Computing raw novelty signals on Validation set...")
        val_signals = compute_all_novelty_signals(val_df, clf, booster, mah_stats, leaf_profiles, feats_a, feats_b, idx_to_class)

        normalizer = EmpiricalCDFNormalizer()
        normalizer.fit(val_signals)
        Z_val_df = normalizer.transform(val_signals)

        # 6. Synthesize Pseudo-Unknowns & Optimize 4-Simplex Weights
        print("  Synthesizing multi-strategy pseudo-unknowns and optimizing fusion weights...")
        pseudo_df = synthesize_pseudo_unknowns(val_df, mah_stats, feats_a, feats_b, known_classes, class_to_idx, cfg, random_seed=rand_seed)
        pseudo_signals = compute_all_novelty_signals(pseudo_df, clf, booster, mah_stats, leaf_profiles, feats_a, feats_b, idx_to_class)
        Z_pseudo_df = normalizer.transform(pseudo_signals)

        val_pred_idx = val_signals["pred_idx"].values
        pseudo_pred_idx = pseudo_signals["pred_idx"].values
        val_is_normal = (val_df["subcategory"].values == "Normal")

        opt_weights, opt_info = optimize_weights_simplex(
            Z_val_df, Z_pseudo_df, val_pred_idx, pseudo_pred_idx, val_is_normal,
            num_classes=num_classes, p_percentile=primary_pct, grid_step=0.05
        )
        print(f"  Optimized Weights [Conf, Mah, Leaf, Rel]: {opt_weights} (Utility: {opt_info.get('utility', 'N/A')})")

        weights_summary.append({
            "heldout_attack": heldout_attack,
            "w_confidence": opt_weights[0],
            "w_mahalanobis": opt_weights[1],
            "w_leaf": opt_weights[2],
            "w_relative": opt_weights[3],
            "val_known_acceptance": opt_info.get("known_acceptance", 0.0),
            "val_benign_rejection": opt_info.get("benign_rejection", 0.0),
            "val_pseudo_recall": opt_info.get("pseudo_recall", 0.0)
        })

        # 7. Calibrate Class-Conditional Adaptive Thresholds
        sig_cols = ["Z_confidence", "Z_mahalanobis", "Z_leaf", "Z_relative"]
        s_val_unified = Z_val_df[sig_cols].values @ opt_weights

        threshold_dict = {}
        for c_idx in range(num_classes):
            c_name = idx_to_class[c_idx]
            m_c = (val_pred_idx == c_idx)
            sub_s = s_val_unified[m_c]
            c_thresh = {}
            for p in cand_percentiles:
                c_thresh[f"P{p:.1f}"] = float(np.percentile(sub_s, p)) if np.any(m_c) else float(np.percentile(s_val_unified, p))
            threshold_dict[c_idx] = c_thresh
            thresholds_summary.append({
                "heldout_attack": heldout_attack,
                "class_idx": c_idx,
                "class_name": c_name,
                **c_thresh
            })

        # Primary threshold vector for classes at primary_pct
        primary_key = f"P{primary_pct:.1f}"
        tau_primary = np.array([threshold_dict[c][primary_key] for c in range(num_classes)])

        # 8. Unbiased Evaluation on Test Known Traffic & Zero-Day Traffic
        print("  Evaluating on Known Test Set and Held-Out Zero-Day Set...")
        # Known test evaluation
        test_signals = compute_all_novelty_signals(test_df, clf, booster, mah_stats, leaf_profiles, feats_a, feats_b, idx_to_class)
        Z_test_df = normalizer.transform(test_signals)
        s_test_unified = Z_test_df[sig_cols].values @ opt_weights

        test_pred_idx = test_signals["pred_idx"].values
        test_tau = tau_primary[test_pred_idx]
        test_is_unknown = (s_test_unified > test_tau)

        test_true_subcat = test_df["subcategory"].values
        test_is_normal = (test_true_subcat == "Normal")
        test_is_known_attack = ~test_is_normal

        # A) Benign False Alarm Rate (FAR) on Normal
        n_normal_test = int(np.sum(test_is_normal))
        benign_false_alarms = int(np.sum(test_is_unknown[test_is_normal]))
        far_rate = (benign_false_alarms / float(n_normal_test) * 100.0) if n_normal_test > 0 else 0.0
        benign_acceptance_rate = 100.0 - far_rate

        # B) Known Attack Detection Rate & Accuracy
        n_known_attack_test = int(np.sum(test_is_known_attack))
        known_attacks_accepted = int(np.sum(~test_is_unknown[test_is_known_attack]))
        known_attack_acceptance_rate = (known_attacks_accepted / float(n_known_attack_test) * 100.0) if n_known_attack_test > 0 else 0.0

        # Closed-set classification accuracy on accepted known attacks
        test_true_idx = np.array([class_to_idx[c] for c in test_true_subcat])
        correct_class_mask = (test_pred_idx == test_true_idx)
        accepted_and_correct = np.sum((~test_is_unknown) & test_is_known_attack & correct_class_mask)
        known_attack_accuracy = (accepted_and_correct / float(known_attacks_accepted) * 100.0) if known_attacks_accepted > 0 else 0.0
        known_attack_effective_detection = (accepted_and_correct / float(n_known_attack_test) * 100.0) if n_known_attack_test > 0 else 0.0

        overall_known_acceptance = (np.sum(~test_is_unknown) / float(len(test_df)) * 100.0)

        # C) Held-Out Zero-Day Evaluation
        zd_signals = compute_all_novelty_signals(df_zd, clf, booster, mah_stats, leaf_profiles, feats_a, feats_b, idx_to_class)
        Z_zd_df = normalizer.transform(zd_signals)
        s_zd_unified = Z_zd_df[sig_cols].values @ opt_weights

        zd_pred_idx = zd_signals["pred_idx"].values
        zd_tau = tau_primary[zd_pred_idx]
        zd_is_detected = (s_zd_unified > zd_tau)

        tp_zd = int(np.sum(zd_is_detected))
        fn_zd = int(n_zd - tp_zd)
        zero_day_recall = (tp_zd / float(n_zd) * 100.0) if n_zd > 0 else 0.0

        # D) Open-Set AUROC & AUPRC (Zero-Day vs Known Test)
        y_open_true = np.concatenate([np.ones(n_zd, dtype=np.int32), np.zeros(len(test_df), dtype=np.int32)])
        y_open_scores = np.concatenate([s_zd_unified, s_test_unified])
        open_set_auroc = roc_auc_score(y_open_true, y_open_scores) * 100.0
        open_set_auprc = average_precision_score(y_open_true, y_open_scores) * 100.0

        # Precision & F1 for Zero-Day identification against all test traffic
        fp_total = int(np.sum(test_is_unknown))
        zd_precision = (tp_zd / float(tp_zd + fp_total) * 100.0) if (tp_zd + fp_total) > 0 else 0.0
        zd_f1 = (2.0 * zd_precision * zero_day_recall / (zd_precision + zero_day_recall)) if (zd_precision + zero_day_recall) > 0 else 0.0
        balanced_accuracy = (zero_day_recall + overall_known_acceptance) / 2.0

        elapsed_att = time.time() - t_att_0
        print(f"\n  === Results for Zero-Day: {heldout_attack} ===")
        print(f"    Zero-Day Detection Rate (Recall): {zero_day_recall:.2f}% ({tp_zd:,} / {n_zd:,})")
        print(f"    Benign False Alarm Rate (FAR):    {far_rate:.2f}% ({benign_false_alarms:,} / {n_normal_test:,})")
        print(f"    Benign Normal Acceptance Rate:    {benign_acceptance_rate:.2f}%")
        print(f"    Known Attack Acceptance Rate:     {known_attack_acceptance_rate:.2f}%")
        print(f"    Known Attack Classification Acc:  {known_attack_accuracy:.2f}%")
        print(f"    Overall Known Acceptance:         {overall_known_acceptance:.2f}%")
        print(f"    Open-Set AUROC:                   {open_set_auroc:.2f}%")
        print(f"    Open-Set AUPRC:                   {open_set_auprc:.2f}%")
        print(f"    Zero-Day F1 Score:                {zd_f1:.2f}%")
        print(f"    Execution Time:                   {elapsed_att:.1f}s")

        res_entry = {
            "heldout_attack": heldout_attack,
            "zero_day_samples": n_zd,
            "zero_day_detected": tp_zd,
            "zero_day_missed": fn_zd,
            "zero_day_detection_rate_pct": round(zero_day_recall, 2),
            "normal_test_samples": n_normal_test,
            "benign_false_alarms": benign_false_alarms,
            "false_alarm_rate_pct": round(far_rate, 2),
            "benign_acceptance_rate_pct": round(benign_acceptance_rate, 2),
            "known_attack_test_samples": n_known_attack_test,
            "known_attack_accepted": known_attacks_accepted,
            "known_attack_acceptance_rate_pct": round(known_attack_acceptance_rate, 2),
            "known_attack_classification_acc_pct": round(known_attack_accuracy, 2),
            "known_attack_effective_detection_pct": round(known_attack_effective_detection, 2),
            "overall_known_acceptance_pct": round(overall_known_acceptance, 2),
            "open_set_auroc_pct": round(open_set_auroc, 2),
            "open_set_auprc_pct": round(open_set_auprc, 2),
            "zero_day_precision_pct": round(zd_precision, 2),
            "zero_day_f1_score_pct": round(zd_f1, 2),
            "balanced_accuracy_pct": round(balanced_accuracy, 2),
            "optimal_weights": [round(float(w), 4) for w in opt_weights],
            "evaluation_time_sec": round(elapsed_att, 1)
        }
        master_results.append(res_entry)

    # 9. Aggregate Master Summary
    df_master = pd.DataFrame(master_results)
    master_csv_path = os.path.join(IMPL_DIR, "outputs", "loao_evaluations", "loao_master_results.csv")
    df_master.to_csv(master_csv_path, index=False)

    master_json_path = os.path.join(IMPL_DIR, "outputs", "loao_evaluations", "loao_master_results.json")
    with open(master_json_path, "w", encoding="utf-8") as f:
        json.dump(master_results, f, indent=2)

    df_weights = pd.DataFrame(weights_summary)
    weights_path = os.path.join(IMPL_DIR, "outputs", "weights_and_thresholds", "learned_weights_per_attack.csv")
    df_weights.to_csv(weights_path, index=False)

    df_thresh = pd.DataFrame(thresholds_summary)
    thresh_path = os.path.join(IMPL_DIR, "outputs", "weights_and_thresholds", "calibrated_thresholds_per_attack.csv")
    df_thresh.to_csv(thresh_path, index=False)

    # Compute Macro & Micro Averages
    macro_zd_recall = df_master["zero_day_detection_rate_pct"].mean()
    macro_far = df_master["false_alarm_rate_pct"].mean()
    macro_known_acc = df_master["known_attack_acceptance_rate_pct"].mean()
    macro_known_class_acc = df_master["known_attack_classification_acc_pct"].mean()
    macro_auroc = df_master["open_set_auroc_pct"].mean()
    macro_auprc = df_master["open_set_auprc_pct"].mean()
    macro_f1 = df_master["zero_day_f1_score_pct"].mean()
    macro_bal_acc = df_master["balanced_accuracy_pct"].mean()

    # Micro average (sample-weighted)
    total_zd_flows = df_master["zero_day_samples"].sum()
    total_zd_detected = df_master["zero_day_detected"].sum()
    micro_zd_recall = (total_zd_detected / float(total_zd_flows) * 100.0) if total_zd_flows > 0 else 0.0

    total_norm_flows = df_master["normal_test_samples"].sum()
    total_norm_fa = df_master["benign_false_alarms"].sum()
    micro_far = (total_norm_fa / float(total_norm_flows) * 100.0) if total_norm_flows > 0 else 0.0

    print("\n" + "=" * 95)
    print(">>> COMPLETE LEAVE-ONE-ATTACK-OUT MASTER PERFORMANCE SUMMARY <<<")
    print("=" * 95)
    print(f"{'Held-Out Zero-Day':<18} | {'ZD Flows':<9} | {'ZD Recall':<10} | {'FAR (Norm)':<10} | {'Known Acc':<10} | {'AUROC':<8} | {'F1':<8}")
    print("-" * 95)
    for _, r in df_master.iterrows():
        print(f"{r['heldout_attack']:<18} | {r['zero_day_samples']:>9,d} | {r['zero_day_detection_rate_pct']:>9.2f}% | {r['false_alarm_rate_pct']:>9.2f}% | {r['known_attack_acceptance_rate_pct']:>9.2f}% | {r['open_set_auroc_pct']:>7.2f}% | {r['zero_day_f1_score_pct']:>7.2f}%")
    print("-" * 95)
    print(f"{'MACRO AVERAGE':<18} | {'--':>9} | {macro_zd_recall:>9.2f}% | {macro_far:>9.2f}% | {macro_known_acc:>9.2f}% | {macro_auroc:>7.2f}% | {macro_f1:>7.2f}%")
    print(f"{'MICRO AVERAGE':<18} | {total_zd_flows:>9,d} | {micro_zd_recall:>9.2f}% | {micro_far:>9.2f}% | {'--':>10} | {'--':>8} | {'--':>8}")
    print("=" * 95)

    summary_stats = {
        "macro_zero_day_recall_pct": round(float(macro_zd_recall), 2),
        "micro_zero_day_recall_pct": round(float(micro_zd_recall), 2),
        "macro_false_alarm_rate_pct": round(float(macro_far), 2),
        "micro_false_alarm_rate_pct": round(float(micro_far), 2),
        "macro_known_attack_acceptance_pct": round(float(macro_known_acc), 2),
        "macro_known_attack_classification_acc_pct": round(float(macro_known_class_acc), 2),
        "macro_open_set_auroc_pct": round(float(macro_auroc), 2),
        "macro_open_set_auprc_pct": round(float(macro_auprc), 2),
        "macro_zero_day_f1_score_pct": round(float(macro_f1), 2),
        "macro_balanced_accuracy_pct": round(float(macro_bal_acc), 2),
        "total_zero_day_flows_evaluated": int(total_zd_flows),
        "total_zero_day_flows_detected": int(total_zd_detected)
    }
    summary_path = os.path.join(IMPL_DIR, "outputs", "loao_evaluations", "loao_summary_statistics.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2)

    total_pipeline_time = time.time() - start_time
    print(f"\nStep 2 LOAO Completed in {total_pipeline_time:.2f}s ({total_pipeline_time/60.0:.2f} minutes).")
    print(f"Master results saved to: {master_csv_path}")

if __name__ == "__main__":
    run_loao_pipeline()
