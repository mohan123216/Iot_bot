"""
06_optimize_weights.py: Learn multi-signal novelty fusion weights via constrained complementary optimization
Step 10: Optimized Multi-Signal Novelty Fusion
Strict Rule: Evaluated on Validation Known Traffic and Pseudo-Unknowns ONLY. Service_Scan is strictly quarantined.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from common_utils import load_step10_config, ensure_output_dirs, STEP10_DIR

def compute_redundancy_penalty(w, R_mat):
    """
    R(w) = sum_{i < j} w_i * w_j * |rho_ij|
    Measures the shared/redundant variance between weighted signals.
    """
    w = np.array(w, dtype=np.float64)
    r = 0.0
    for i in range(len(w)):
        for j in range(i + 1, len(w)):
            r += w[i] * w[j] * R_mat[i, j]
    return float(r)

def evaluate_weight_vector(w, Z_val_mat, Z_pseudo_mat, val_pred_idx, pseudo_pred_idx, val_is_normal, num_classes=7, p_percentile=95.0, compute_auc=False):
    w = np.array(w, dtype=np.float64)
    w = np.maximum(0.0, w)
    w_sum = np.sum(w)
    if w_sum > 0:
        w = w / w_sum
    else:
        w = np.ones(4) / 4.0

    score_val = Z_val_mat @ w
    score_pseudo = Z_pseudo_mat @ w

    thresholds = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        mask_c = (val_pred_idx == c)
        if np.any(mask_c):
            thresholds[c] = np.percentile(score_val[mask_c], p_percentile)
        else:
            thresholds[c] = np.percentile(score_val, p_percentile)

    tau_val = thresholds[val_pred_idx]
    is_unknown_val = (score_val > tau_val)
    fp_val = np.sum(is_unknown_val)
    n_val = len(score_val)
    known_acceptance = 1.0 - (fp_val / float(n_val))
    false_unknown_rate = fp_val / float(n_val)

    n_normal = np.sum(val_is_normal)
    benign_rejections = np.sum(is_unknown_val[val_is_normal])
    benign_rejection_rate = (benign_rejections / float(n_normal)) if n_normal > 0 else 0.0

    tau_pseudo = thresholds[pseudo_pred_idx]
    is_detected_pseudo = (score_pseudo > tau_pseudo)
    tp_pseudo = np.sum(is_detected_pseudo)
    n_pseudo = len(score_pseudo)
    recall_pseudo = tp_pseudo / float(n_pseudo)

    precision_pseudo = (tp_pseudo / float(tp_pseudo + fp_val)) if (tp_pseudo + fp_val) > 0 else 0.0
    f1_pseudo = (2.0 * precision_pseudo * recall_pseudo / (precision_pseudo + recall_pseudo)) if (precision_pseudo + recall_pseudo) > 0 else 0.0

    if compute_auc:
        y_true = np.concatenate([np.zeros(min(5000, n_val), dtype=np.int32), np.ones(n_pseudo, dtype=np.int32)])
        y_scores = np.concatenate([score_val[:min(5000, n_val)], score_pseudo])
        auc = roc_auc_score(y_true, y_scores)
    else:
        auc = 0.0

    return {
        "weights": [round(float(v), 5) for v in w],
        "known_acceptance": float(known_acceptance),
        "false_unknown_rate": float(false_unknown_rate),
        "benign_rejection_rate": float(benign_rejection_rate),
        "recall_pseudo": float(recall_pseudo),
        "precision_pseudo": float(precision_pseudo),
        "f1_pseudo": float(f1_pseudo),
        "auroc": float(auc)
    }

def evaluate_weight_vector_full(w, Z_val_mat, Z_pseudo_mat, val_pred_idx, pseudo_pred_idx, 
                               val_is_normal, pseudo_gen_types, R_mat, loss_lambdas,
                               num_classes=7, p_percentile=95.0, compute_auc=True):
    """
    Full evaluation of a candidate weight vector:
      - Class-conditional adaptive thresholds on validation known traffic
      - Operational constraints: Known Acceptance and Benign False Alarms
      - Separate evaluation on pseudo-unknown families:
          A. Cross-class interpolation
          B. Covariance perturbation
          C. Boundary low-density extrapolation
          D. Hard pseudo-unknowns
      - Balanced F1, Balanced AUROC, Redundancy Penalty R(w), and Multi-Objective J(w)
    """
    w = np.array(w, dtype=np.float64)
    w_sum = np.sum(w)
    if w_sum > 0:
        w = w / w_sum
    else:
        w = np.ones(4) / 4.0

    score_val = Z_val_mat @ w
    score_pseudo = Z_pseudo_mat @ w

    n_val = len(score_val)
    n_normal = np.sum(val_is_normal)

    # 1. Class-conditional thresholds on validation score
    thresholds = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        mask_c = (val_pred_idx == c)
        if np.any(mask_c):
            thresholds[c] = np.percentile(score_val[mask_c], p_percentile)
        else:
            thresholds[c] = np.percentile(score_val, p_percentile)

    tau_val = thresholds[val_pred_idx]
    is_unknown_val = (score_val > tau_val)
    fp_val = int(np.sum(is_unknown_val))
    known_acceptance = 1.0 - (fp_val / float(n_val))
    false_unknown_rate = fp_val / float(n_val)

    # Benign normal false alarms
    fp_normal = int(np.sum(is_unknown_val[val_is_normal]))
    benign_rejection_rate = (fp_normal / float(n_normal)) if n_normal > 0 else 0.0

    # 2. Pseudo-unknown detection
    tau_pseudo = thresholds[pseudo_pred_idx]
    is_detected_pseudo = (score_pseudo > tau_pseudo)
    tp_pseudo_total = int(np.sum(is_detected_pseudo))
    n_pseudo_total = len(score_pseudo)

    rec_agg = tp_pseudo_total / float(n_pseudo_total)
    prec_agg = (tp_pseudo_total / float(tp_pseudo_total + fp_val)) if (tp_pseudo_total + fp_val) > 0 else 0.0
    f1_agg = (2.0 * prec_agg * rec_agg / (prec_agg + rec_agg)) if (prec_agg + rec_agg) > 0 else 0.0

    # 3. Family-Specific Metrics
    families = {
        "interpolation": pseudo_gen_types == "cross_class_interpolation",
        "covariance": pseudo_gen_types == "controlled_perturbation",
        "extrapolation": pseudo_gen_types == "boundary_low_density",
        "hard": pseudo_gen_types == "hard_pseudo_unknown"
    }

    fam_metrics = {}
    for fam_name, fam_mask in families.items():
        n_fam = int(np.sum(fam_mask))
        if n_fam == 0:
            fam_metrics[fam_name] = {"recall": 0.0, "precision": 0.0, "f1": 0.0, "auroc": 0.5}
            continue
        tp_fam = int(np.sum(is_detected_pseudo[fam_mask]))
        rec_fam = tp_fam / float(n_fam)
        prec_fam = (tp_fam / float(tp_fam + fp_val)) if (tp_fam + fp_val) > 0 else 0.0
        f1_fam = (2.0 * prec_fam * rec_fam / (prec_fam + rec_fam)) if (prec_fam + rec_fam) > 0 else 0.0

        if compute_auc:
            y_true_fam = np.concatenate([np.zeros(min(5000, n_val), dtype=np.int32), np.ones(n_fam, dtype=np.int32)])
            y_scores_fam = np.concatenate([score_val[:min(5000, n_val)], score_pseudo[fam_mask]])
            auc_fam = roc_auc_score(y_true_fam, y_scores_fam)
        else:
            auc_fam = 0.5

        fam_metrics[fam_name] = {
            "recall": float(rec_fam),
            "precision": float(prec_fam),
            "f1": float(f1_fam),
            "auroc": float(auc_fam)
        }

    # 4. Balanced Objectives
    f1_bal = (fam_metrics["interpolation"]["f1"] + fam_metrics["covariance"]["f1"] + fam_metrics["extrapolation"]["f1"]) / 3.0
    f1_hard = fam_metrics["hard"]["f1"]
    auc_bal = (fam_metrics["interpolation"]["auroc"] + fam_metrics["covariance"]["auroc"] + fam_metrics["extrapolation"]["auroc"]) / 3.0

    # 5. Redundancy Penalty
    r_val = compute_redundancy_penalty(w, R_mat)

    # 6. Overall Multi-Objective J(w)
    # J(w) = F1_balanced + lambda_hard * F1_hard + lambda_auc * AUROC_balanced - lambda_red * Redundancy
    l_f1 = loss_lambdas.get("lambda_balanced_f1", 1.00)
    l_hard = loss_lambdas.get("lambda_hard", 0.35)
    l_auc = loss_lambdas.get("lambda_auc", 0.25)
    l_red = loss_lambdas.get("lambda_red", 0.15)

    j_score = (l_f1 * f1_bal) + (l_hard * f1_hard) + (l_auc * auc_bal) - (l_red * r_val)

    return {
        "weights": [round(float(v), 5) for v in w],
        "known_acceptance": float(known_acceptance),
        "false_unknown_rate": float(false_unknown_rate),
        "false_unknown_flows": fp_val,
        "normal_false_alarms": fp_normal,
        "benign_rejection_rate": float(benign_rejection_rate),
        "recall_aggregate": float(rec_agg),
        "precision_aggregate": float(prec_agg),
        "f1_aggregate": float(f1_agg),
        "f1_balanced": float(f1_bal),
        "f1_hard": float(f1_hard),
        "f1_interpolation": float(fam_metrics["interpolation"]["f1"]),
        "f1_covariance": float(fam_metrics["covariance"]["f1"]),
        "f1_extrapolation": float(fam_metrics["extrapolation"]["f1"]),
        "auroc_balanced": float(auc_bal),
        "auroc_interpolation": float(fam_metrics["interpolation"]["auroc"]),
        "auroc_covariance": float(fam_metrics["covariance"]["auroc"]),
        "auroc_extrapolation": float(fam_metrics["extrapolation"]["auroc"]),
        "auroc_hard": float(fam_metrics["hard"]["auroc"]),
        "redundancy_penalty": float(r_val),
        "j_score": float(j_score)
    }

def run_optimize_weights():
    print("=" * 80)
    print(">>> STEP 10: 06 - LEARN CONSTRAINED COMPLEMENTARY MULTI-SIGNAL WEIGHTS <<<")
    print("=" * 80)

    ensure_output_dirs()
    cfg = load_step10_config()
    weight_cfg = cfg["parameters"]["weight_optimization"]
    loss_lambdas = weight_cfg["loss_weights"]
    known_acc_min = float(weight_cfg["known_acceptance_min"])
    benign_rej_max = float(weight_cfg["benign_rejection_max"])
    grid_step = float(weight_cfg["simplex_grid_step"])
    w_floor = float(weight_cfg.get("weight_floor", 0.05))
    seed = int(cfg["experiment"]["random_seed"])

    print("Objective Formulation:")
    print(f"  Maximize J(w) = {loss_lambdas['lambda_balanced_f1']}*F1_bal + {loss_lambdas['lambda_hard']}*F1_hard + {loss_lambdas['lambda_auc']}*AUROC_bal - {loss_lambdas['lambda_red']}*Redundancy(w)")
    print(f"Constraints:")
    print(f"  Simplex: wC + wM + wL + wR = 1.0, with weight floor w_i >= {w_floor:.2f} for ALL signals")
    print(f"  Operational: Known Acceptance >= {known_acc_min*100:.1f}%, Benign False Alarms <= 2 (or Rejection <= {benign_rej_max*100:.1f}%)\n")

    # 1. Load Calibrated Validation Data & Correlation Matrix
    val_calib_path = os.path.join(STEP10_DIR, "outputs", "calibration", "calibrated_validation_signals.parquet")
    df_val = pd.read_parquet(val_calib_path)

    pseudo_calib_path = os.path.join(STEP10_DIR, "outputs", "pseudo_unknown", "pseudo_calibrated_signals.parquet")
    df_pseudo = pd.read_parquet(pseudo_calib_path)

    corr_path = os.path.join(STEP10_DIR, "outputs", "signal_analysis", "signal_correlation_matrix.csv")
    corr_df = pd.read_csv(corr_path, index_col=0)
    R_mat = np.abs(corr_df.values)

    sig_cols = ["Z_confidence", "Z_mahalanobis", "Z_leaf", "Z_relative"]
    Z_val_mat = df_val[sig_cols].values.astype(np.float64)
    Z_pseudo_mat = df_pseudo[sig_cols].values.astype(np.float64)

    val_pred_idx = df_val["pred_idx"].values.astype(np.int32)
    pseudo_pred_idx = df_pseudo["pred_idx"].values.astype(np.int32)
    val_is_normal = (df_val["true_class"].values == "Normal")
    pseudo_gen_types = df_pseudo["generator_type"].values

    # 2. Section 8: Partition Pseudo-Unknowns into Calibration & Selection Subsets (50/50 Stratified)
    rng = np.random.RandomState(seed)
    calib_indices = []
    select_indices = []
    for g_type in np.unique(pseudo_gen_types):
        g_idx = np.where(pseudo_gen_types == g_type)[0]
        perm = rng.permutation(g_idx)
        split_pt = len(perm) // 2
        calib_indices.extend(perm[:split_pt])
        select_indices.extend(perm[split_pt:])

    calib_indices = np.array(calib_indices)
    select_indices = np.array(select_indices)
    print(f"Stratified Pseudo-Unknown Partition: {len(calib_indices):,} calibration / {len(select_indices):,} selection flows.")

    # 3. Baseline Evaluations
    print("\n--- Evaluating Baseline Weight Configurations ---")
    w_equal = np.array([0.25, 0.25, 0.25, 0.25])
    res_equal = evaluate_weight_vector_full(w_equal, Z_val_mat, Z_pseudo_mat, val_pred_idx, pseudo_pred_idx, 
                                           val_is_normal, pseudo_gen_types, R_mat, loss_lambdas)

    w_step9 = np.array([0.7353, 0.2099, 0.0000, 0.0548])
    res_step9 = evaluate_weight_vector_full(w_step9, Z_val_mat, Z_pseudo_mat, val_pred_idx, pseudo_pred_idx, 
                                           val_is_normal, pseudo_gen_types, R_mat, loss_lambdas)

    w_old_step10 = np.array([1.0000, 0.0000, 0.0000, 0.0000])
    res_old_step10 = evaluate_weight_vector_full(w_old_step10, Z_val_mat, Z_pseudo_mat, val_pred_idx, pseudo_pred_idx, 
                                                val_is_normal, pseudo_gen_types, R_mat, loss_lambdas)

    print(f"Equal Weights:          J={res_equal['j_score']:.4f} | F1_bal={res_equal['f1_balanced']*100:.2f}% | F1_hard={res_equal['f1_hard']*100:.2f}% | Redundancy={res_equal['redundancy_penalty']:.4f}")
    print(f"Step 9 AUC Weights:     J={res_step9['j_score']:.4f} | F1_bal={res_step9['f1_balanced']*100:.2f}% | F1_hard={res_step9['f1_hard']*100:.2f}% | Redundancy={res_step9['redundancy_penalty']:.4f}")
    print(f"Old Step 10 (Conf Only): J={res_old_step10['j_score']:.4f} | F1_bal={res_old_step10['f1_balanced']*100:.2f}% | F1_hard={res_old_step10['f1_hard']*100:.2f}% | Redundancy={res_old_step10['redundancy_penalty']:.4f}")

    # 4. Simplex Grid Search with Strict Weight Floor w_i >= 0.05
    # Reparameterize: w_i = w_floor + u_i, sum(u_i) = 1.0 - 4*w_floor = 0.80
    rem_weight = 1.0 - 4.0 * w_floor
    ticks = np.round(np.arange(0.0, rem_weight + 1e-6, grid_step), 4)

    grid_points = []
    for i in range(len(ticks)):
        u0 = ticks[i]
        for j in range(len(ticks) - i):
            u1 = ticks[j]
            for k in range(len(ticks) - i - j):
                u2 = ticks[k]
                u3 = round(rem_weight - (u0 + u1 + u2), 4)
                if u3 >= -1e-6:
                    grid_points.append([w_floor + u0, w_floor + u1, w_floor + u2, w_floor + max(0.0, u3)])

    grid_points = np.array(grid_points)
    print(f"\nGenerated {len(grid_points):,} candidate vectors on 4-simplex with floor {w_floor:.2f} and step {grid_step:.2f}.")

    # Fast evaluation over selection split of pseudo-unknowns
    Z_pseudo_sel = Z_pseudo_mat[select_indices]
    pseudo_pred_sel = pseudo_pred_idx[select_indices]
    pseudo_gen_sel = pseudo_gen_types[select_indices]

    num_classes = 7
    class_masks_val = [val_pred_idx == c for c in range(num_classes)]
    n_val = len(df_val)

    # Subsample validation for fast AUROC during grid search
    auc_sample_idx = rng.choice(n_val, size=min(5000, n_val), replace=False)
    Z_val_sub = Z_val_mat[auc_sample_idx]

    best_j_val = -1e9
    best_w = None
    best_res_sel = None

    candidate_records = []
    t0 = time.time()

    print("Evaluating simplex candidates subject to operational acceptance and benign constraints...")
    for idx, w_cand in enumerate(grid_points):
        # 1. Validation known score and thresholds
        s_val = Z_val_mat @ w_cand
        taus = np.zeros(num_classes, dtype=np.float64)
        for c in range(num_classes):
            m = class_masks_val[c]
            taus[c] = np.percentile(s_val[m], 95.0) if np.any(m) else np.percentile(s_val, 95.0)

        tau_v = taus[val_pred_idx]
        fp_val = np.sum(s_val > tau_v)
        known_acc = 1.0 - (fp_val / float(n_val))

        # Operational filter: Known Acceptance >= 94.5%
        if known_acc < known_acc_min:
            continue

        # Operational filter: Benign False Alarms <= 2
        fp_norm = np.sum((s_val > tau_v) & val_is_normal)
        if fp_norm > 2:
            continue

        # 2. Pseudo-unknown performance on selection partition
        s_pse = Z_pseudo_sel @ w_cand
        tau_p = taus[pseudo_pred_sel]
        det_pse = (s_pse > tau_p)

        def get_fam_f1(mask_g):
            n_g = np.sum(mask_g)
            if n_g == 0:
                return 0.0
            tp_g = np.sum(det_pse[mask_g])
            prec_g = tp_g / float(tp_g + fp_val) if (tp_g + fp_val) > 0 else 0.0
            rec_g = tp_g / float(n_g)
            return (2.0 * prec_g * rec_g / (prec_g + rec_g)) if (prec_g + rec_g) > 0 else 0.0

        mask_i = (pseudo_gen_sel == "cross_class_interpolation")
        mask_c = (pseudo_gen_sel == "controlled_perturbation")
        mask_e = (pseudo_gen_sel == "boundary_low_density")
        mask_h = (pseudo_gen_sel == "hard_pseudo_unknown")

        f1_i = get_fam_f1(mask_i)
        f1_c = get_fam_f1(mask_c)
        f1_e = get_fam_f1(mask_e)
        f1_h = get_fam_f1(mask_h)
        f1_bal = (f1_i + f1_c + f1_e) / 3.0

        # Fast AUROC calculation
        s_val_sub_w = Z_val_sub @ w_cand
        y_val_sub = np.zeros(len(s_val_sub_w), dtype=np.int32)
        auc_i = roc_auc_score(np.concatenate([y_val_sub, np.ones(mask_i.sum())]), np.concatenate([s_val_sub_w, s_pse[mask_i]]))
        auc_c = roc_auc_score(np.concatenate([y_val_sub, np.ones(mask_c.sum())]), np.concatenate([s_val_sub_w, s_pse[mask_c]]))
        auc_e = roc_auc_score(np.concatenate([y_val_sub, np.ones(mask_e.sum())]), np.concatenate([s_val_sub_w, s_pse[mask_e]]))
        auc_bal = (auc_i + auc_c + auc_e) / 3.0

        r_val = compute_redundancy_penalty(w_cand, R_mat)

        j_val = (loss_lambdas["lambda_balanced_f1"] * f1_bal) + \
                (loss_lambdas["lambda_hard"] * f1_h) + \
                (loss_lambdas["lambda_auc"] * auc_bal) - \
                (loss_lambdas["lambda_red"] * r_val)

        if j_val > best_j_val:
            best_j_val = j_val
            best_w = w_cand

        candidate_records.append({
            "wC": round(float(w_cand[0]), 4),
            "wM": round(float(w_cand[1]), 4),
            "wL": round(float(w_cand[2]), 4),
            "wR": round(float(w_cand[3]), 4),
            "known_acceptance": round(known_acc * 100.0, 2),
            "normal_false_alarms": int(fp_norm),
            "f1_balanced": round(f1_bal * 100.0, 2),
            "f1_hard": round(f1_h * 100.0, 2),
            "f1_interpolation": round(f1_i * 100.0, 2),
            "f1_covariance": round(f1_c * 100.0, 2),
            "f1_extrapolation": round(f1_e * 100.0, 2),
            "auroc_balanced": round(auc_bal, 4),
            "redundancy_penalty": round(r_val, 4),
            "j_score": round(j_val, 4)
        })

    elapsed_search = time.time() - t0
    print(f"Simplex grid search finished in {elapsed_search:.2f}s. Evaluated {len(candidate_records):,} feasible candidates.")
    print(f"Optimal Grid Solution: wC={best_w[0]:.4f}, wM={best_w[1]:.4f}, wL={best_w[2]:.4f}, wR={best_w[3]:.4f} (J={best_j_val:.4f})")

    # 5. Full Evaluation of the Learned Weights on Complete Pseudo-Unknown Set
    res_learned = evaluate_weight_vector_full(best_w, Z_val_mat, Z_pseudo_mat, val_pred_idx, pseudo_pred_idx, 
                                             val_is_normal, pseudo_gen_types, R_mat, loss_lambdas)

    # 6. Extract Pareto Frontier Candidates
    df_all_cands = pd.DataFrame(candidate_records)
    df_all_cands.to_csv(os.path.join(STEP10_DIR, "outputs", "weights", "grid_search_all_candidates.csv"), index=False)

    pareto_mask = np.ones(len(df_all_cands), dtype=bool)
    f1_vals = df_all_cands["f1_balanced"].values
    j_vals = df_all_cands["j_score"].values
    acc_vals = df_all_cands["known_acceptance"].values

    for i in range(len(df_all_cands)):
        dominated = np.any((j_vals > j_vals[i]) & (f1_vals >= f1_vals[i]) & (acc_vals >= acc_vals[i]))
        if dominated:
            pareto_mask[i] = False

    df_pareto = df_all_cands[pareto_mask].sort_values(by="j_score", ascending=False).reset_index(drop=True)
    df_pareto.to_csv(os.path.join(STEP10_DIR, "outputs", "weights", "pareto_weights.csv"), index=False)
    print(f"Extracted {len(df_pareto)} Pareto-optimal candidates.")

    # 7. Build Strategy Comparison Table
    comparison_rows = [
        {
            "Strategy": "Baseline A: Equal Weights",
            "wC": 0.2500,
            "wM": 0.2500,
            "wL": 0.2500,
            "wR": 0.2500,
            "Known Acceptance (%)": round(res_equal["known_acceptance"] * 100.0, 2),
            "Benign False Alarms": res_equal["normal_false_alarms"],
            "F1 Balanced (%)": round(res_equal["f1_balanced"] * 100.0, 2),
            "F1 Hard (%)": round(res_equal["f1_hard"] * 100.0, 2),
            "AUROC Balanced": round(res_equal["auroc_balanced"], 4),
            "Redundancy Penalty": round(res_equal["redundancy_penalty"], 4),
            "Objective J": round(res_equal["j_score"], 4)
        },
        {
            "Strategy": "Baseline B: Step 9 Historical AUC Weights",
            "wC": round(float(w_step9[0]), 4),
            "wM": round(float(w_step9[1]), 4),
            "wL": round(float(w_step9[2]), 4),
            "wR": round(float(w_step9[3]), 4),
            "Known Acceptance (%)": round(res_step9["known_acceptance"] * 100.0, 2),
            "Benign False Alarms": res_step9["normal_false_alarms"],
            "F1 Balanced (%)": round(res_step9["f1_balanced"] * 100.0, 2),
            "F1 Hard (%)": round(res_step9["f1_hard"] * 100.0, 2),
            "AUROC Balanced": round(res_step9["auroc_balanced"], 4),
            "Redundancy Penalty": round(res_step9["redundancy_penalty"], 4),
            "Objective J": round(res_step9["j_score"], 4)
        },
        {
            "Strategy": "Baseline C: Old Step 10 (Confidence Only Collapse)",
            "wC": round(float(w_old_step10[0]), 4),
            "wM": round(float(w_old_step10[1]), 4),
            "wL": round(float(w_old_step10[2]), 4),
            "wR": round(float(w_old_step10[3]), 4),
            "Known Acceptance (%)": round(res_old_step10["known_acceptance"] * 100.0, 2),
            "Benign False Alarms": res_old_step10["normal_false_alarms"],
            "F1 Balanced (%)": round(res_old_step10["f1_balanced"] * 100.0, 2),
            "F1 Hard (%)": round(res_old_step10["f1_hard"] * 100.0, 2),
            "AUROC Balanced": round(res_old_step10["auroc_balanced"], 4),
            "Redundancy Penalty": round(res_old_step10["redundancy_penalty"], 4),
            "Objective J": round(res_old_step10["j_score"], 4)
        },
        {
            "Strategy": "Step 10 NEW Method: Learned Constrained Four-Signal",
            "wC": round(float(best_w[0]), 4),
            "wM": round(float(best_w[1]), 4),
            "wL": round(float(best_w[2]), 4),
            "wR": round(float(best_w[3]), 4),
            "Known Acceptance (%)": round(res_learned["known_acceptance"] * 100.0, 2),
            "Benign False Alarms": res_learned["normal_false_alarms"],
            "F1 Balanced (%)": round(res_learned["f1_balanced"] * 100.0, 2),
            "F1 Hard (%)": round(res_learned["f1_hard"] * 100.0, 2),
            "AUROC Balanced": round(res_learned["auroc_balanced"], 4),
            "Redundancy Penalty": round(res_learned["redundancy_penalty"], 4),
            "Objective J": round(res_learned["j_score"], 4)
        }
    ]

    df_comp = pd.DataFrame(comparison_rows)
    print("\n--- Final Master Weight Strategy Comparison Table ---")
    print(df_comp.to_string(index=False))

    # 8. Save Weights Configuration
    weights_summary = {
        "selected_optimal_weights": {
            "w_confidence": round(float(best_w[0]), 4),
            "w_mahalanobis": round(float(best_w[1]), 4),
            "w_leaf": round(float(best_w[2]), 4),
            "w_relative": round(float(best_w[3]), 4)
        },
        "old_step10_weights": {
            "w_confidence": 1.0000,
            "w_mahalanobis": 0.0000,
            "w_leaf": 0.0000,
            "w_relative": 0.0000
        },
        "step9_auc_weights": {
            "w_confidence": round(float(w_step9[0]), 4),
            "w_mahalanobis": round(float(w_step9[1]), 4),
            "w_leaf": round(float(w_step9[2]), 4),
            "w_relative": round(float(w_step9[3]), 4)
        },
        "equal_weights": {
            "w_confidence": 0.25,
            "w_mahalanobis": 0.25,
            "w_leaf": 0.25,
            "w_relative": 0.25
        },
        "performance_breakdown": {
            "interpolation_f1_pct": round(res_learned["f1_interpolation"] * 100.0, 2),
            "covariance_f1_pct": round(res_learned["f1_covariance"] * 100.0, 2),
            "extrapolation_f1_pct": round(res_learned["f1_extrapolation"] * 100.0, 2),
            "hard_pseudo_f1_pct": round(res_learned["f1_hard"] * 100.0, 2),
            "balanced_f1_pct": round(res_learned["f1_balanced"] * 100.0, 2),
            "aggregate_f1_pct": round(res_learned["f1_aggregate"] * 100.0, 2),
            "balanced_auroc": round(res_learned["auroc_balanced"], 4),
            "redundancy_penalty": round(res_learned["redundancy_penalty"], 4),
            "objective_j": round(res_learned["j_score"], 4),
            "known_acceptance_pct": round(res_learned["known_acceptance"] * 100.0, 2),
            "benign_false_alarms": res_learned["normal_false_alarms"]
        },
        "constraints": {
            "weight_floor": w_floor,
            "known_acceptance_min": known_acc_min,
            "benign_rejection_max": benign_rej_max
        },
        "loss_weights": loss_lambdas
    }

    opt_json_path = os.path.join(STEP10_DIR, "outputs", "weights", "optimized_weights.json")
    with open(opt_json_path, "w", encoding="utf-8") as f:
        json.dump(weights_summary, f, indent=2)

    comp_csv_path = os.path.join(STEP10_DIR, "outputs", "weights", "weight_comparison.csv")
    df_comp.to_csv(comp_csv_path, index=False)

    print(f"\nSaved learned weights to: {opt_json_path} and {comp_csv_path}")
    return weights_summary, df_comp

if __name__ == "__main__":
    run_optimize_weights()
