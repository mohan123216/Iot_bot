#!/usr/bin/env python3
"""
01_stream_clean_full_dataset.py: Stream and clean the complete 74-file UNSW Bot-IoT dataset
Audits 73,370,443 flows and constructs a comprehensive, leakage-free representative corpus.
Retains 100% of all rare classes (Normal, Data_Exfiltration, Keylogging, HTTP).
"""

import os
import sys
import time
import glob
import json
import numpy as np
import pandas as pd
from collections import Counter

from common_utils import load_config, ensure_dirs, parse_port_val, IMPL_DIR, WORKSPACE_DIR

def run_stream_cleaning():
    start_time = time.time()
    print("=" * 85)
    print(">>> FULL DATASET IMPLEMENTATION: STEP 1 - STREAMING CLEANING & AUDIT <<<")
    print("=" * 85)

    ensure_dirs()
    cfg = load_config()

    # Mappings
    proto_map = cfg["mappings"]["proto"]
    flgs_map = cfg["mappings"]["flgs"]
    state_map = cfg["mappings"]["state"]

    retain_100_pct = set(cfg["dataset_sampling"]["retain_100_percent_classes"])
    subsample_ratios = cfg["dataset_sampling"]["subsample_ratios"]
    rand_seed = cfg["experiment"]["random_seed"]
    np.random.seed(rand_seed)

    # Discover all 74 files
    data_files = sorted([os.path.join(WORKSPACE_DIR, f"data_{i}.csv") for i in range(1, 75)])
    print(f"Total dataset files discovered: {len(data_files)}")

    # Audit accumulators
    global_subcat_counts = Counter()
    global_cat_counts = Counter()
    total_raw_rows = 0

    cleaned_sample_chunks = []

    print("\nStreaming through all 74 files with chunksize=250,000...")

    for f_idx, fpath in enumerate(data_files):
        fname = os.path.basename(fpath)
        t_file_0 = time.time()
        file_rows = 0

        # Read only necessary columns to maximize speed and minimize memory
        # pkSeqID, stime, flgs, proto, saddr, sport, daddr, dport, pkts, bytes, state,
        # ltime, seq, dur, mean, stddev, sum, min, max, spkts, dpkts, sbytes, dbytes,
        # rate, srate, drate, attack, category, subcategory
        cols_to_read = [
            "proto", "flgs", "state", "sport", "dport", "dur", "pkts", "bytes",
            "spkts", "dpkts", "sbytes", "dbytes", "rate", "srate", "drate",
            "mean", "stddev", "sum", "min", "max", "attack", "category", "subcategory "
        ]

        for chunk in pd.read_csv(fpath, usecols=cols_to_read, chunksize=250000, low_memory=False):
            # Clean column names
            chunk.columns = [c.strip() for c in chunk.columns]
            chunk_len = len(chunk)
            file_rows += chunk_len
            total_raw_rows += chunk_len

            # 1. Standardize string labels
            chunk["category"] = chunk["category"].astype(str).str.strip()
            chunk["subcategory"] = chunk["subcategory"].astype(str).str.strip()

            cat_map = {"dos": "DoS", "ddos": "DDoS", "reconnaissance": "Reconnaissance", "normal": "Normal", "theft": "Theft"}
            subcat_map = {
                "udp": "UDP", "tcp": "TCP", "service_scan": "Service_Scan",
                "os_fingerprint": "OS_Fingerprint", "http": "HTTP",
                "normal": "Normal", "keylogging": "Keylogging",
                "data_exfiltration": "Data_Exfiltration"
            }
            chunk["category"] = chunk["category"].map(lambda x: cat_map.get(x.lower(), x))
            chunk["subcategory"] = chunk["subcategory"].map(lambda x: subcat_map.get(x.lower(), x))

            # Audit count
            global_subcat_counts.update(chunk["subcategory"])
            global_cat_counts.update(chunk["category"])

            # 2. Filter mask for sampling
            subcats = chunk["subcategory"].values
            is_rare = np.isin(subcats, list(retain_100_pct))

            # Subsampling mask for high-volume classes
            keep_mask = is_rare.copy()
            for sc, ratio in subsample_ratios.items():
                sc_mask = (subcats == sc)
                if np.any(sc_mask):
                    rand_draw = np.random.uniform(0.0, 1.0, size=np.sum(sc_mask)) < ratio
                    keep_mask[sc_mask] = rand_draw

            if not np.any(keep_mask):
                continue

            sub_chunk = chunk[keep_mask].copy()

            # 3. Feature cleaning & mappings
            # Categorical encodings
            proto_clean = sub_chunk["proto"].astype(str).str.strip()
            sub_chunk["proto_number"] = proto_clean.map(lambda x: proto_map.get(x, 0)).astype(np.int32)

            flgs_clean = sub_chunk["flgs"].astype(str).str.strip()
            sub_chunk["flgs_number"] = flgs_clean.map(lambda x: flgs_map.get(x, 0)).astype(np.int32)

            state_clean = sub_chunk["state"].astype(str).str.strip()
            sub_chunk["state_number"] = state_clean.map(lambda x: state_map.get(x, 0)).astype(np.int32)

            # Ports
            sub_chunk["sport"] = sub_chunk["sport"].apply(parse_port_val).astype(np.int32)
            sub_chunk["dport"] = sub_chunk["dport"].apply(parse_port_val).astype(np.int32)

            # Numeric columns
            num_cols = ["dur", "pkts", "bytes", "spkts", "dpkts", "sbytes", "dbytes", "rate", "srate", "drate", "mean", "stddev", "sum", "min", "max"]
            for nc in num_cols:
                sub_chunk[nc] = pd.to_numeric(sub_chunk[nc], errors="coerce").fillna(0.0)
                sub_chunk[nc] = sub_chunk[nc].replace([np.inf, -np.inf], 0.0)

            # Keep only required columns
            final_cols = ["proto_number", "flgs_number", "state_number", "sport", "dport"] + num_cols + ["attack", "category", "subcategory"]
            cleaned_sample_chunks.append(sub_chunk[final_cols])

        t_file_elapsed = time.time() - t_file_0
        if (f_idx + 1) % 10 == 0 or (f_idx + 1) == len(data_files):
            print(f"  [{f_idx+1:2d}/74] {fname} ({file_rows:,} rows, {t_file_elapsed:.1f}s) | Cumulative: {total_raw_rows:,} flows")

    print("\nAggregating cleaned full-dataset representative sample...")
    df_clean_sample = pd.concat(cleaned_sample_chunks, ignore_index=True)

    # Save cleaned sample
    out_sample_path = cfg["paths"]["cleaned_sample_path"]
    df_clean_sample.to_parquet(out_sample_path, index=False)
    print(f"Saved representative sample to: {out_sample_path}")
    print(f"Total representative flows: {len(df_clean_sample):,} ({os.path.getsize(out_sample_path)/(1024*1024):.2f} MB)")

    # Save audit tables
    audit_df = pd.DataFrame([
        {"file_count": len(data_files), "total_raw_flows": total_raw_rows, "sampled_corpus_flows": len(df_clean_sample)}
    ])
    audit_path = os.path.join(IMPL_DIR, "outputs", "dataset_audit", "full_dataset_audit_74files.csv")
    audit_df.to_csv(audit_path, index=False)

    class_dist_rows = []
    sample_subcat_counts = df_clean_sample["subcategory"].value_counts().to_dict()
    for sc, raw_cnt in sorted(global_subcat_counts.items(), key=lambda x: -x[1]):
        smpl_cnt = sample_subcat_counts.get(sc, 0)
        retention_pct = (smpl_cnt / raw_cnt * 100.0) if raw_cnt > 0 else 0.0
        role = "Benign Control" if sc == "Normal" else "Attack Class"
        class_dist_rows.append({
            "subcategory": sc,
            "role": role,
            "raw_total_count": raw_cnt,
            "raw_percentage": round(raw_cnt / total_raw_rows * 100.0, 4),
            "sample_count": smpl_cnt,
            "sample_percentage": round(smpl_cnt / len(df_clean_sample) * 100.0, 4),
            "retention_rate_pct": round(retention_pct, 2)
        })

    class_dist_df = pd.DataFrame(class_dist_rows)
    class_dist_path = os.path.join(IMPL_DIR, "outputs", "dataset_audit", "class_distribution_full.csv")
    class_dist_df.to_csv(class_dist_path, index=False)
    print(f"Saved class distribution audit to: {class_dist_path}")

    print("\n--- Full Dataset Class Distribution Summary ---")
    print(f"{'Subcategory':<20} | {'Raw Total Count':<16} | {'Sample Count':<14} | {'Retention %':<12} | {'Role':<15}")
    print("-" * 85)
    for r in class_dist_rows:
        print(f"{r['subcategory']:<20} | {r['raw_total_count']:>16,d} | {r['sample_count']:>14,d} | {r['retention_rate_pct']:>11.2f}% | {r['role']:<15}")
    print("-" * 85)

    total_time = time.time() - start_time
    print(f"\nStep 1 Completed in {total_time:.2f}s ({total_time/60.0:.2f} minutes).")
    return df_clean_sample

if __name__ == "__main__":
    run_stream_cleaning()
