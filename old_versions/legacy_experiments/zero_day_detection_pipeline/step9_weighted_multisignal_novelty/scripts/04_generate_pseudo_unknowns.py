"""
04_generate_pseudo_unknowns.py: Multi-strategy pseudo-unknown sample generator
Strict rule: Uses validation known data only; Service_Scan count is verified to be 0.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from common_utils import load_step9_config, get_abs_path, load_frozen_artifacts

def run_generate_pseudo_unknowns():
    print("=" * 80)
    print(">>> STEP 9: 04 - GENERATE PSEUDO-UNKNOWN SAMPLES <<<")
    print("=" * 80)

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
    continuous_cols = [c for c in feats_a if c not in discrete_cols]

    rng = np.random.RandomState(cfg["experiment"]["random_seed"])
    n_per_gen = cfg["parameters"]["pseudo_unknown"]["n_samples_per_generator"]

    pseudo_records = []

    # Strategy A: Cross-Class Interpolation
    print(f"Strategy A: Generating {n_per_gen} cross-class interpolated samples...")
    lambdas = cfg["parameters"]["pseudo_unknown"]["cross_class_lambdas"]
    val_indices = df_val.index.values

    # Pre-group validation indices by known class
    class_indices = {c: df_val[df_val["subcategory"] == c].index.values for c in known_classes}

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
                new_row[col] = lam * v1 + (1.0 - lam) * v2

        new_row["generator_type"] = "cross_class_interpolation"
        new_row["parent_class_1"] = c1
        new_row["parent_class_2"] = c2
        new_row["lambda_param"] = float(lam)
        pseudo_records.append(new_row)
        count_a += 1

    # Strategy B: Controlled Covariance-Aligned Perturbation
    print(f"Strategy B: Generating {n_per_gen} covariance-perturbed samples...")
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

        # Draw perturbation in standardized Representation B space
        cov_c = covariances[c_idx]
        delta_std = rng.multivariate_normal(mean=np.zeros(len(feats_b)), cov=cov_c) * (sigma / 2.0)
        # Convert back to unstandardized feature scale
        delta_unscaled = delta_std * stds_b

        new_row = {}
        for col in feats_a:
            if col in feats_b:
                f_idx = feats_b.index(col)
                val_orig = float(base_row[col])
                new_val = val_orig + delta_unscaled[f_idx]
                # Enforce physical non-negativity for rates, bytes, pkts, dur
                if col in ["dur", "pkts", "bytes", "spkts", "dpkts", "sbytes", "dbytes", "rate", "srate", "drate"]:
                    new_val = max(0.0, new_val)
                new_row[col] = new_val
            else:
                new_row[col] = base_row[col]

        new_row["generator_type"] = "controlled_perturbation"
        new_row["parent_class_1"] = c
        new_row["parent_class_2"] = c
        new_row["lambda_param"] = float(sigma)
        pseudo_records.append(new_row)
        count_b += 1

    # Strategy C: Boundary / Low-Density Extrapolation
    print(f"Strategy C: Generating {n_per_gen} boundary low-density extrapolated samples...")
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
                # Extrapolate away from centroid
                v_extrap = mu_val + gamma * (v_orig - mu_val)
                if col in ["dur", "pkts", "bytes", "spkts", "dpkts", "sbytes", "dbytes", "rate", "srate", "drate"]:
                    v_extrap = max(0.0, v_extrap)
                new_row[col] = v_extrap
            else:
                new_row[col] = base_row[col]

        new_row["generator_type"] = "boundary_low_density"
        new_row["parent_class_1"] = c
        new_row["parent_class_2"] = c
        new_row["lambda_param"] = float(gamma)
        pseudo_records.append(new_row)
        count_c += 1

    df_pseudo = pd.DataFrame(pseudo_records)
    df_pseudo["subcategory"] = "PSEUDO_UNKNOWN"

    step9_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_parquet_path = os.path.join(step9_dir, "outputs", "pseudo_unknown", "pseudo_unknown_samples.parquet")
    df_pseudo.to_parquet(out_parquet_path, index=False)

    report_dict = {
        "total_pseudo_unknowns": len(df_pseudo),
        "generator_breakdown": df_pseudo["generator_type"].value_counts().to_dict(),
        "random_seed": cfg["experiment"]["random_seed"],
        "strategies": {
            "cross_class_interpolation": {"count": count_a, "lambdas": lambdas},
            "controlled_perturbation": {"count": count_b, "sigmas": sigmas},
            "boundary_low_density": {"count": count_c, "gammas": gammas}
        },
        "zero_day_quarantined": True,
        "calibration_source": "validation.parquet"
    }

    out_json_path = os.path.join(step9_dir, "outputs", "pseudo_unknown", "generation_report.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    print(f"\nSaved {len(df_pseudo):,} pseudo-unknown samples to: {out_parquet_path}")
    print(f"Saved generation report to: {out_json_path}")
    return df_pseudo

if __name__ == "__main__":
    run_generate_pseudo_unknowns()
