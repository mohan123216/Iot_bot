"""
========================================================================================
03_evaluate_paper_table9.py: Reproduction and Outperformance Benchmark for Table 9
Reference Paper: "Multi-Stage Enhanced Zero Trust IDS for Unknown Attack Detection in IoT"
                 ACM Transactions on Privacy and Security (ACM TOPS, 2025)
========================================================================================
This script performs Leave-One-Subclass-Out (LOCO / Type-B Unknown Attack) evaluation
across all 10 distinct attack combinations in the UNSW Bot-IoT dataset:
  - DoS: HTTP, TCP, UDP
  - DDoS: HTTP, TCP, UDP
  - Reconnaissance: OS_Fingerprint, Service_Scan
  - Theft: Keylogging, Data_Exfiltration

For each test:
  1. One attack subclass is held out as completely unknown (Zero-Day).
  2. The Zero-Trust IDS is trained on Benign Normal traffic + the other 9 known attack subclasses.
  3. The model is evaluated on the held-out unknown attack vs Normal test traffic.
  4. Standard metrics (Precision, Recall, Accuracy, F1-Score, Error Rate) are computed.
========================================================================================
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "fulldataset_cleaned_sample.parquet")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "loao_evaluations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_table9_evaluation():
    print("=" * 80)
    print("RUNNING TABLE 9 BENCHMARK: ZERO-TRUST DETECTION OF UNKNOWN ATTACKS")
    print(f"Loading balanced corpus from: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    print(f"Total flows loaded: {len(df):,}")

    drop_cols = ["attack", "category", "subcategory"]
    feat_cols = [c for c in df.columns if c not in drop_cols]

    # The 10 Class-Subclass pairs matching the research paper
    attack_pairs = [
        ("DoS", "HTTP"),
        ("DoS", "TCP"),
        ("DoS", "UDP"),
        ("DDoS", "HTTP"),
        ("DDoS", "TCP"),
        ("DDoS", "UDP"),
        ("Reconnaissance", "OS_Fingerprint"),
        ("Reconnaissance", "Service_Scan"),
        ("Theft", "Keylogging"),
        ("Theft", "Data_Exfiltration")
    ]

    # Fixed Benign Normal partition (70% train, 30% test)
    normal_df = df[df["category"] == "Normal"]
    train_norm, test_norm = train_test_split(normal_df, test_size=0.30, random_state=42)
    n_norm_train = len(train_norm)
    n_norm_test = len(test_norm)

    print(f"Benign Normal Traffic: {n_norm_train:,} Train flows | {n_norm_test:,} Test flows")
    print("-" * 80)

    results = []
    t_start = time.time()

    for idx, (cat, sub) in enumerate(attack_pairs, start=1):
        t0 = time.time()
        print(f"[{idx}/10] Evaluating Held-Out Attack: {cat} -> {sub}...")

        # Partition data: held-out unknown attack vs remaining known attacks
        held_out_attack = df[(df["category"] == cat) & (df["subcategory"] == sub)]
        known_attacks = df[(df["category"] != "Normal") & ~((df["category"] == cat) & (df["subcategory"] == sub))]

        n_heldout = len(held_out_attack)
        n_known = len(known_attacks)

        # Training set: Train Normal + 9 Known Attack Subclasses
        train_data = pd.concat([train_norm, known_attacks], ignore_index=True)
        y_train = (train_data["category"] != "Normal").astype(np.int32).values
        X_train = train_data[feat_cols].values

        # Sample weight balancing to prevent known floods from drowning out normal flows
        weight_normal = (len(train_data) / (2.0 * n_norm_train))
        weight_attack = (len(train_data) / (2.0 * n_known))
        sample_weights = np.where(y_train == 0, weight_normal, weight_attack)

        # Train Zero-Trust Binary Detection Model
        clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train, sample_weight=sample_weights)

        # Test set: Held-out unknown attack + Benign Normal test traffic
        test_data = pd.concat([held_out_attack, test_norm], ignore_index=True)
        y_test = (test_data["category"] != "Normal").astype(np.int32).values
        X_test = test_data[feat_cols].values

        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        err = 1.0 - acc

        elapsed = time.time() - t0
        print(f"      Result: Precision={prec:.2f} | Recall={rec:.2f} | Accuracy={acc:.2f} | F1={f1:.2f} | Error={err:.2f} ({elapsed:.1f}s)")

        results.append({
            "Class": cat,
            "Subclass": sub,
            "Test_Attack_Flows": n_heldout,
            "Test_Normal_Flows": n_norm_test,
            "Precision": round(prec, 2),
            "Recall": round(rec, 2),
            "Accuracy": round(acc, 2),
            "F1": round(f1, 2),
            "Error": round(err, 2)
        })

    res_df = pd.DataFrame(results)
    out_csv = os.path.join(OUTPUT_DIR, "paper_table9_reproduction_metrics.csv")
    res_df.to_csv(out_csv, index=False)

    print("=" * 80)
    print(f"ALL 10 TESTS COMPLETED IN {time.time() - t_start:.1f} SECONDS!")
    print(f"Results saved to: {out_csv}")
    print("\n" + res_df.to_string(index=False))
    return res_df

if __name__ == "__main__":
    run_table9_evaluation()
