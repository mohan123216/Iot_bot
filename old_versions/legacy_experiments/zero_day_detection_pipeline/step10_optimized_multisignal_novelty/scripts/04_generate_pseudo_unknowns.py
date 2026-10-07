"""
04_generate_pseudo_unknowns.py: Multi-strategy pseudo-unknown sample generation
Step 10: Optimized Multi-Signal Novelty Fusion
Strict Rule: Derived strictly from Validation Known Traffic; Service_Scan is strictly absent.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from common_utils import load_step10_config, get_abs_path, load_frozen_artifacts, ensure_output_dirs, STEP10_DIR

def run_generate_pseudo_unknowns():
    print("=" * 80)
    print(">>> STEP 10: 04 - GENERATE MULTI-STRATEGY PSEUDO-UNKNOWNS <<<")
    print("=" * 80)

    ensure_output_dirs()
    artifacts = load_frozen_artifacts()
    cfg = artifacts["cfg"]
    mah_stats = artifacts["mah_stats"]
    mapping = artifacts["mapping"]
    zero_day_class = mapping["zero_day_class"]
    known_classes = mapping["known_classes"]
    class_to_idx = mapping["class_to_idx"]

    val_path = get_abs_path(cfg["paths"]["validation_path"])
    df_val = pd.read_parquet(val_path)

    # Verification: Service_Scan is strictly absent
    zd_count = int((df_val["subcategory"] == zero_day_class).sum())
    assert zd_count == 0, f"FATAL LEAKAGE: {zero_day_class} found in calibration data!"

    feats_a = cfg["features"]["representation_a_xgboost"]
    feats_b = cfg["features"]["representation_b_geometry"]
    discrete_cols = ["sport", "dport", "proto_number", "flgs_number", "state_number"]
    non_negative_cols = ["dur", "pkts", "bytes", "spkts", "dpkts", "sbytes", "dbytes", "rate", "srate", "drate"]

    rng = np.random.RandomState(cfg["experiment"]["random_seed"])
    n_per_gen = cfg["parameters"]["pseudo_unknown"]["n_samples_per_generator"]

    pseudo_records = []

    # Pre-group validation indices by known class
    class_indices = {c: df_val[df_val["subcategory"] == c].index.values for c in known_classes}

    # Strategy A: Cross-Class Interpolation
    print(f"Strategy A: Generating {n_per_gen} cross-class interpolated flows...")
    lambdas = cfg["parameters"]["pseudo_unknown"]["cross_class_lambdas"]
    count_a = 0
    while count_a < n_per_gen:
        c1, c2 = rng.choice(known_classes, size=2, replace=False)
        if len(class_indices[c1]) == 0 or len(class_indices[c2]) == 0:
            continue
        idx1 = rng.choice(class_indices[c1])
        idx2 = rng.choice(class_indices[c2])
        lam = rng.choice(lambdas)

        row1 = df_val.loc[idx1]
        row2 = df_val.loc[idx2]

        new_row = {}
        for col in feats_a:
            if col in discrete_cols:
                new_row[col] = row1[col] if rng.rand() < lam else row2[col]
            else:
                v1 = float(row1[col])
                v2 = float(row2[col])
                val_interp = lam * v1 + (1.0 - lam) * v2
                if col in non_negative_cols:
                    val_interp = max(0.0, val_interp)
                new_row[col] = val_interp

        new_row["generator_type"] = "cross_class_interpolation"
        new_row["parent_class_1"] = c1
        new_row["parent_class_2"] = c2
        new_row["param_value"] = float(lam)
        pseudo_records.append(new_row)
        count_a += 1

    # Strategy B: Controlled Covariance-Aligned Perturbation
    print(f"Strategy B: Generating {n_per_gen} covariance-perturbed flows...")
    sigmas = cfg["parameters"]["pseudo_unknown"]["perturbation_sigmas"]
    centroids = mah_stats["centroids"]
    covariances = mah_stats["covariances"]
    stds_b = mah_stats["stds_b"]
    means_b = mah_stats["means_b"]

    count_b = 0
    while count_b < n_per_gen:
        c = rng.choice(known_classes)
        if len(class_indices[c]) == 0:
            continue
        c_idx = class_to_idx[c]
        idx = rng.choice(class_indices[c])
        sigma = rng.choice(sigmas)
        base_row = df_val.loc[idx]

        cov_c = covariances[c_idx]
        delta_std = rng.multivariate_normal(mean=np.zeros(len(feats_b)), cov=cov_c) * (sigma / 2.0)
        delta_unscaled = delta_std * stds_b

        new_row = {}
        for col in feats_a:
            if col in feats_b:
                f_idx = feats_b.index(col)
                val_orig = float(base_row[col])
                new_val = val_orig + delta_unscaled[f_idx]
                if col in non_negative_cols:
                    new_val = max(0.0, new_val)
                new_row[col] = new_val
            else:
                new_row[col] = base_row[col]

        new_row["generator_type"] = "controlled_perturbation"
        new_row["parent_class_1"] = c
        new_row["parent_class_2"] = c
        new_row["param_value"] = float(sigma)
        pseudo_records.append(new_row)
        count_b += 1

    # Strategy C: Boundary / Low-Density Extrapolation
    print(f"Strategy C: Generating {n_per_gen} boundary low-density extrapolated flows...")
    gammas = cfg["parameters"]["pseudo_unknown"]["extrapolation_gammas"]

    count_c = 0
    while count_c < n_per_gen:
        c = rng.choice(known_classes)
        if len(class_indices[c]) == 0:
            continue
        c_idx = class_to_idx[c]
        idx = rng.choice(class_indices[c])
        gamma = rng.choice(gammas)
        base_row = df_val.loc[idx]

        mu_unscaled = centroids[c_idx] * stds_b + means_b

        new_row = {}
        for col in feats_a:
            if col in feats_b:
                f_idx = feats_b.index(col)
                v_orig = float(base_row[col])
                mu_val = float(mu_unscaled[f_idx])
                v_extrap = mu_val + gamma * (v_orig - mu_val)
                if col in non_negative_cols:
                    v_extrap = max(0.0, v_extrap)
                new_row[col] = v_extrap
            else:
                new_row[col] = base_row[col]

        new_row["generator_type"] = "boundary_low_density"
        new_row["parent_class_1"] = c
        new_row["parent_class_2"] = c
        new_row["param_value"] = float(gamma)
        pseudo_records.append(new_row)
        count_c += 1

    # Strategy D: Hard Pseudo-Unknowns (High Closed-Set Confidence, Divergent Representations)
    hard_cfg = cfg["parameters"]["pseudo_unknown"].get("hard_pseudo_unknown", {})
    n_hard = hard_cfg.get("n_samples", 1500)
    min_conf = float(hard_cfg.get("min_confidence", 0.85))
    noise_sigmas = hard_cfg.get("noise_sigmas", [1.2, 1.8, 2.4])
    print(f"Strategy D: Generating {n_hard} hard pseudo-unknowns (min_conf={min_conf}, sigmas={noise_sigmas})...")

    # Load normalizer to assess candidate signal divergence
    from importlib import import_module
    compute_signals_mod = import_module("02_compute_raw_signals")
    compute_all_novelty_signals = compute_signals_mod.compute_all_novelty_signals
    norm_mod = import_module("03_calibrate_signals")
    EmpiricalCDFNormalizer = norm_mod.EmpiricalCDFNormalizer

    norm_path = os.path.join(STEP10_DIR, "outputs", "calibration", "per_signal_calibration.json")
    normalizer = EmpiricalCDFNormalizer()
    normalizer.load(norm_path)

    n_pool = max(4000, n_hard * 3)
    cand_records = []
    for _ in range(n_pool):
        c = rng.choice(known_classes)
        if len(class_indices[c]) == 0:
            continue
        c_idx = class_to_idx[c]
        idx = rng.choice(class_indices[c])
        sigma = rng.choice(noise_sigmas)
        base_row = df_val.loc[idx]

        cov_c = covariances[c_idx]
        delta_std = rng.multivariate_normal(mean=np.zeros(len(feats_b)), cov=cov_c) * (sigma / 2.0)
        delta_unscaled = delta_std * stds_b

        cand_row = {}
        for col in feats_a:
            if col in feats_b:
                f_idx = feats_b.index(col)
                val_orig = float(base_row[col])
                new_val = val_orig + delta_unscaled[f_idx]
                if col in non_negative_cols:
                    new_val = max(0.0, new_val)
                cand_row[col] = new_val
            else:
                cand_row[col] = base_row[col]

        cand_row["parent_class_1"] = c
        cand_row["parent_class_2"] = c
        cand_row["param_value"] = float(sigma)
        cand_records.append(cand_row)

    df_cand_pool = pd.DataFrame(cand_records)
    cand_signals = compute_all_novelty_signals(df_cand_pool, artifacts)
    cand_z = normalizer.transform(cand_signals)
    cand_pmax = cand_signals["p_max"].values

    mask_high_conf = (cand_pmax >= min_conf)
    sub_target = n_hard // 3  # 500 per subtype

    idx_geom = np.where(mask_high_conf & (cand_z["Z_mahalanobis"].values >= 0.80))[0]
    idx_leaf = np.where(mask_high_conf & (cand_z["Z_leaf"].values >= 0.80))[0]
    idx_rel = np.where(mask_high_conf & (cand_z["Z_relative"].values >= 0.80))[0]

    chosen_indices = set()
    # 1. High conf + Geometric distance
    for i in rng.permutation(idx_geom):
        if len(chosen_indices) >= sub_target:
            break
        chosen_indices.add(int(i))

    # 2. High conf + Leaf-space divergence
    leaf_added = 0
    for i in rng.permutation(idx_leaf):
        if leaf_added >= sub_target:
            break
        if int(i) not in chosen_indices:
            chosen_indices.add(int(i))
            leaf_added += 1

    # 3. High conf + Relative class distance
    rel_added = 0
    for i in rng.permutation(idx_rel):
        if rel_added >= sub_target:
            break
        if int(i) not in chosen_indices:
            chosen_indices.add(int(i))
            rel_added += 1

    # If any remaining to fill n_hard, pick from remaining high-confidence candidates with high total non-conf novelty
    if len(chosen_indices) < n_hard:
        rem_mask = mask_high_conf.copy()
        rem_mask[list(chosen_indices)] = False
        rem_idx = np.where(rem_mask)[0]
        non_conf_score = (cand_z.loc[rem_idx, "Z_mahalanobis"].values +
                          cand_z.loc[rem_idx, "Z_leaf"].values +
                          cand_z.loc[rem_idx, "Z_relative"].values)
        sorted_rem = rem_idx[np.argsort(-non_conf_score)]
        for i in sorted_rem:
            if len(chosen_indices) >= n_hard:
                break
            chosen_indices.add(int(i))

    for idx in chosen_indices:
        r = df_cand_pool.iloc[idx].to_dict()
        r["generator_type"] = "hard_pseudo_unknown"
        pseudo_records.append(r)

    count_d = len(chosen_indices)
    print(f"Strategy D: Successfully generated {count_d} hard pseudo-unknown flows.")

    df_pseudo = pd.DataFrame(pseudo_records)
    df_pseudo["subcategory"] = "PSEUDO_UNKNOWN"

    out_parquet_path = os.path.join(STEP10_DIR, "outputs", "pseudo_unknown", "pseudo_unknown_samples.parquet")
    df_pseudo.to_parquet(out_parquet_path, index=False)

    report_dict = {
        "total_pseudo_unknowns": len(df_pseudo),
        "generator_breakdown": df_pseudo["generator_type"].value_counts().to_dict(),
        "random_seed": cfg["experiment"]["random_seed"],
        "strategies": {
            "cross_class_interpolation": {"count": count_a, "lambdas": lambdas},
            "controlled_perturbation": {"count": count_b, "sigmas": sigmas},
            "boundary_low_density": {"count": count_c, "gammas": gammas},
            "hard_pseudo_unknown": {
                "count": count_d,
                "min_confidence": min_conf,
                "noise_sigmas": noise_sigmas,
                "subcohorts": {
                    "high_conf_mahalanobis": sub_target,
                    "high_conf_leaf": leaf_added,
                    "high_conf_relative": rel_added
                }
            }
        },
        "zero_day_quarantined": True,
        "calibration_source": "validation.parquet",
        "output_path": out_parquet_path
    }

    out_json_path = os.path.join(STEP10_DIR, "outputs", "pseudo_unknown", "generation_report.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    print(f"\nSaved {len(df_pseudo):,} pseudo-unknown flows to: {out_parquet_path}")
    print(f"Saved generation report to: {out_json_path}")
    return df_pseudo

if __name__ == "__main__":
    run_generate_pseudo_unknowns()
