"""
13_generate_plots.py: Generate 16 publication-quality research figures
Step 10: Optimized Multi-Signal Novelty Fusion
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve
from common_utils import load_step10_config, ensure_output_dirs, STEP10_DIR

# Modern publication theme styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--"
})

PALETTE = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "success": "#2ca02c",
    "danger": "#d62728",
    "purple": "#9467bd",
    "brown": "#8c564b",
    "cyan": "#17becf",
    "gray": "#7f7f7f"
}

def save_fig(fig, filename):
    out_dir = os.path.join(STEP10_DIR, "outputs", "plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved plot: {out_path}")

def run_generate_plots():
    print("=" * 80)
    print(">>> STEP 10: 13 - GENERATE PUBLICATION-QUALITY FIGURES <<<")
    print("=" * 80)

    ensure_output_dirs()

    val_score_path = os.path.join(STEP10_DIR, "outputs", "calibration", "validation_unified_scores.parquet")
    df_val = pd.read_parquet(val_score_path)

    pseudo_calib_path = os.path.join(STEP10_DIR, "outputs", "pseudo_unknown", "pseudo_calibrated_signals.parquet")
    df_pseudo = pd.read_parquet(pseudo_calib_path)

    opt_weights_path = os.path.join(STEP10_DIR, "outputs", "weights", "optimized_weights.json")
    with open(opt_weights_path, "r", encoding="utf-8") as f:
        weights_info = json.load(f)
    from importlib import import_module
    score_mod = import_module("08_build_unified_score")
    builder = score_mod.UnifiedScoreBuilder(weights_info["selected_optimal_weights"])
    df_pseudo["S_unified"] = builder.compute(df_pseudo)

    kt_pred_path = os.path.join(STEP10_DIR, "outputs", "known_test", "predictions.parquet")
    df_kt = pd.read_parquet(kt_pred_path)

    zd_pred_path = os.path.join(STEP10_DIR, "outputs", "zero_day", "predictions.parquet")
    df_zd = pd.read_parquet(zd_pred_path)

    sig_metrics_path = os.path.join(STEP10_DIR, "outputs", "signal_analysis", "signal_metrics.csv")
    df_sig_metrics = pd.read_csv(sig_metrics_path)

    weight_comp_path = os.path.join(STEP10_DIR, "outputs", "weights", "weight_comparison.csv")
    df_weight_comp = pd.read_csv(weight_comp_path)

    pareto_path = os.path.join(STEP10_DIR, "outputs", "weights", "pareto_weights.csv")
    df_pareto = pd.read_csv(pareto_path)

    kt_metrics_path = os.path.join(STEP10_DIR, "outputs", "known_test", "metrics.json")
    with open(kt_metrics_path, "r", encoding="utf-8") as f:
        kt_metrics = json.load(f)

    zd_metrics_path = os.path.join(STEP10_DIR, "outputs", "zero_day", "metrics.json")
    with open(zd_metrics_path, "r", encoding="utf-8") as f:
        zd_metrics = json.load(f)

    subgroup_path = os.path.join(STEP10_DIR, "outputs", "zero_day", "subgroup_analysis.csv")
    df_subgroup = pd.read_csv(subgroup_path)

    master_comp_path = os.path.join(STEP10_DIR, "outputs", "zero_day", "master_comparison.csv")
    df_master_comp = pd.read_csv(master_comp_path)

    ablation_path = os.path.join(STEP10_DIR, "outputs", "ablation", "ablation_results.csv")
    df_ablation = pd.read_csv(ablation_path)

    thresh_path = os.path.join(STEP10_DIR, "outputs", "thresholds", "class_conditional_thresholds.csv")
    df_thresholds = pd.read_csv(thresh_path)

    # Transform df_zd into calibrated Z_i
    norm_path = os.path.join(STEP10_DIR, "outputs", "calibration", "per_signal_calibration.json")
    from importlib import import_module
    norm_mod = import_module("03_calibrate_signals")
    normalizer = norm_mod.EmpiricalCDFNormalizer()
    normalizer.load(norm_path)
    df_zd_z = normalizer.transform(df_zd)
    for col in ["Z_confidence", "Z_mahalanobis", "Z_leaf", "Z_relative"]:
        df_zd[col] = df_zd_z[col].values

    # -------------------------------------------------------------
    # Plot 1: Individual Signal Distributions (Known vs Pseudo-Unknown vs Zero-Day)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    signals = [
        ("Z_confidence", "Confidence Novelty (Z_C)"),
        ("Z_mahalanobis", "Mahalanobis Novelty (Z_M)"),
        ("Z_leaf", "Leaf-Space Novelty (Z_L)"),
        ("Z_relative", "Relative Class-Separation (Z_R)")
    ]
    for idx, (sig_k, title) in enumerate(signals):
        ax = axes[idx // 2, idx % 2]
        sns.kdeplot(df_val[sig_k], ax=ax, label="Known Traffic (Val)", color=PALETTE["primary"], fill=True, alpha=0.3, bw_adjust=1.2)
        sns.kdeplot(df_pseudo[sig_k], ax=ax, label="Pseudo-Unknown", color=PALETTE["secondary"], fill=True, alpha=0.3, bw_adjust=1.2)
        sns.kdeplot(df_zd[sig_k], ax=ax, label="Zero-Day (Service_Scan)", color=PALETTE["danger"], fill=True, alpha=0.3, bw_adjust=1.2)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Calibrated Novelty Score Z_i in [0, 1]")
        ax.set_ylabel("Density")
        ax.set_xlim(-0.02, 1.02)
        ax.legend(loc="upper left", frameon=True)
    save_fig(fig, "01_individual_signal_distributions.png")

    # -------------------------------------------------------------
    # Plot 2: Signal AUROC Comparison Across Generators
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    cohort_order = ["All Generators (Aggregate)", "Cross-Class Interpolation", "Controlled Perturbation", "Boundary Low-Density"]
    df_plot_sig = df_sig_metrics.copy()
    sns.barplot(data=df_plot_sig, x="cohort", y="auroc", hue="signal_key", ax=ax, palette="Blues_r")
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.5, label="Random Guess (0.50)")
    ax.set_ylim(0.3, 1.05)
    ax.set_title("Individual Novelty Signal AUROC Across Pseudo-Unknown Mechanisms", fontweight="bold")
    ax.set_xlabel("Pseudo-Unknown Generation Mechanism")
    ax.set_ylabel("Area Under ROC Curve (AUROC)")
    ax.legend(title="Signal", loc="lower right", frameon=True)
    plt.xticks(rotation=15, ha="right")
    save_fig(fig, "02_signal_auroc_comparison.png")

    # -------------------------------------------------------------
    # Plot 3: Signal AUPRC Comparison Across Generators
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_plot_sig, x="cohort", y="auprc", hue="signal_key", ax=ax, palette="Purples_r")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Individual Novelty Signal AUPRC Across Pseudo-Unknown Mechanisms", fontweight="bold")
    ax.set_xlabel("Pseudo-Unknown Generation Mechanism")
    ax.set_ylabel("Area Under Precision-Recall Curve (AUPRC)")
    ax.legend(title="Signal", loc="upper right", frameon=True)
    plt.xticks(rotation=15, ha="right")
    save_fig(fig, "03_signal_auprc_comparison.png")

    # -------------------------------------------------------------
    # Plot 4: Weight Comparison: Equal vs Step 9 AUC vs Optimized vs Regularized
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    strategies = df_weight_comp["Strategy"].values
    weights_matrix = df_weight_comp[["wC", "wM", "wL", "wR"]].values
    x_pos = np.arange(len(strategies))
    width = 0.2

    ax.bar(x_pos - 1.5*width, weights_matrix[:, 0], width, label="wC (Confidence)", color=PALETTE["primary"])
    ax.bar(x_pos - 0.5*width, weights_matrix[:, 1], width, label="wM (Mahalanobis)", color=PALETTE["secondary"])
    ax.bar(x_pos + 0.5*width, weights_matrix[:, 2], width, label="wL (Leaf Space)", color=PALETTE["success"])
    ax.bar(x_pos + 1.5*width, weights_matrix[:, 3], width, label="wR (Relative Dist)", color=PALETTE["purple"])

    ax.set_xticks(x_pos)
    ax.set_xticklabels(strategies, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Normalized Weight (sum = 1.0)")
    ax.set_title("Comparison of Multi-Signal Weighting Strategies", fontweight="bold")
    ax.legend(frameon=True, loc="upper right")
    save_fig(fig, "04_weight_comparison_methods.png")

    # -------------------------------------------------------------
    # Plot 5: Unified Score Distributions (Known vs Pseudo-Unknown vs Zero-Day)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(df_val["S_unified"], ax=ax, label="Validation Known Traffic", color=PALETTE["primary"], fill=True, alpha=0.35, linewidth=2)
    sns.kdeplot(df_pseudo["S_unified"], ax=ax, label="Pseudo-Unknowns (Calibration Proxy)", color=PALETTE["secondary"], fill=True, alpha=0.35, linewidth=2)
    sns.kdeplot(df_zd["S_unified"], ax=ax, label="Zero-Day (Service_Scan - Held Out)", color=PALETTE["danger"], fill=True, alpha=0.35, linewidth=2)
    p95_val = np.percentile(df_val["S_unified"], 95.0)
    ax.axvline(p95_val, color="black", linestyle="--", linewidth=2, label=f"Global P95 Cutoff ({p95_val:.3f})")
    ax.set_title("Unified Novelty Score Distributions (Learned Optimal Weights)", fontweight="bold")
    ax.set_xlabel("S_unified in [0.0, 1.0]")
    ax.set_ylabel("Probability Density")
    ax.set_xlim(-0.02, 1.02)
    ax.legend(frameon=True, loc="upper right")
    save_fig(fig, "05_unified_score_distributions.png")

    # -------------------------------------------------------------
    # Plot 6: Pseudo-Unknown Detection ROC Curves
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))
    for sig_k, sig_label in signals:
        y_true = np.concatenate([np.zeros(len(df_val)), np.ones(len(df_pseudo))])
        y_scores = np.concatenate([df_val[sig_k], df_pseudo[sig_k]])
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = roc_curve_auc = df_sig_metrics[(df_sig_metrics["cohort"] == "All Generators (Aggregate)") & (df_sig_metrics["signal_key"] == sig_k)]["auroc"].values[0]
        ax.plot(fpr, tpr, label=f"{sig_label} (AUC = {auc:.3f})", linewidth=1.8)

    # Add Unified Score ROC
    y_scores_u = np.concatenate([df_val["S_unified"], df_pseudo["S_unified"]])
    fpr_u, tpr_u, _ = roc_curve(y_true, y_scores_u)
    from sklearn.metrics import roc_auc_score
    auc_u = roc_auc_score(y_true, y_scores_u)
    ax.plot(fpr_u, tpr_u, label=f"Step 10 Unified Score (AUC = {auc_u:.3f})", color="black", linewidth=2.5, linestyle="-")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("False Positive Rate (Known Traffic Rejection)")
    ax.set_ylabel("True Positive Rate (Pseudo-Unknown Recall)")
    ax.set_title("Receiver Operating Characteristic (ROC) on Calibration Proxy", fontweight="bold")
    ax.legend(loc="lower right", frameon=True)
    save_fig(fig, "06_pseudo_unknown_roc_curves.png")

    # -------------------------------------------------------------
    # Plot 7: Pseudo-Unknown Precision-Recall Curves
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))
    for sig_k, sig_label in signals:
        y_true = np.concatenate([np.zeros(len(df_val)), np.ones(len(df_pseudo))])
        y_scores = np.concatenate([df_val[sig_k], df_pseudo[sig_k]])
        prec, rec, _ = precision_recall_curve(y_true, y_scores)
        auprc = df_sig_metrics[(df_sig_metrics["cohort"] == "All Generators (Aggregate)") & (df_sig_metrics["signal_key"] == sig_k)]["auprc"].values[0]
        ax.plot(rec, prec, label=f"{sig_label} (AUPRC = {auprc:.3f})", linewidth=1.8)

    prec_u, rec_u, _ = precision_recall_curve(y_true, y_scores_u)
    from sklearn.metrics import average_precision_score
    auprc_u = average_precision_score(y_true, y_scores_u)
    ax.plot(rec_u, prec_u, label=f"Step 10 Unified Score (AUPRC = {auprc_u:.3f})", color="black", linewidth=2.5, linestyle="-")

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("Recall (Pseudo-Unknown Detection Rate)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves on Calibration Proxy", fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    save_fig(fig, "07_pseudo_unknown_pr_curves.png")

    # -------------------------------------------------------------
    # Plot 8: Zero-Day Recall vs Known Acceptance Across Percentiles
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    pct_keys = sorted(list(zd_metrics.keys()))
    recalls = [zd_metrics[k]["zero_day_recall_pct"] for k in pct_keys]
    acceptances = [zd_metrics[k]["known_test_acceptance_pct"] for k in pct_keys]

    ax.plot(acceptances, recalls, marker="o", color=PALETTE["primary"], linewidth=2, markersize=8)
    for k, acc, rec in zip(pct_keys, acceptances, recalls):
        ax.annotate(k, (acc, rec), textcoords="offset points", xytext=(8, -4), fontsize=9)

    ax.axvline(95.0, color="gray", linestyle=":", label="Known Acceptance Constraint (>= 95%)")
    ax.set_xlabel("Known Test Traffic Acceptance Rate (%)")
    ax.set_ylabel("Held-Out Zero-Day Recall (%)")
    ax.set_title("Zero-Day Recall vs Known Acceptance Operational Frontier", fontweight="bold")
    ax.legend(frameon=True)
    save_fig(fig, "08_zeroday_recall_vs_known_acceptance.png")

    # -------------------------------------------------------------
    # Plot 9: Zero-Day Recall vs Benign Rejection Across Percentiles
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    benign_rejs = [zd_metrics[k]["benign_rejection_pct"] for k in pct_keys]
    ax.plot(benign_rejs, recalls, marker="s", color=PALETTE["danger"], linewidth=2, markersize=8)
    for k, ben, rec in zip(pct_keys, benign_rejs, recalls):
        ax.annotate(k, (ben, rec), textcoords="offset points", xytext=(8, 2), fontsize=9)

    ax.axvline(2.0, color="gray", linestyle=":", label="Benign Rejection Constraint (<= 2%)")
    ax.set_xlabel("Benign ('Normal') Traffic False Alarm / Rejection Rate (%)")
    ax.set_ylabel("Held-Out Zero-Day Recall (%)")
    ax.set_title("Zero-Day Recall vs Benign False Alarm Rate", fontweight="bold")
    ax.legend(frameon=True)
    save_fig(fig, "09_zeroday_recall_vs_benign_rejection.png")

    # -------------------------------------------------------------
    # Plot 10: Unknown Precision vs Zero-Day Recall Across Percentiles
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    precisions = [zd_metrics[k]["unknown_precision_pct"] for k in pct_keys]
    f1s = [zd_metrics[k]["unknown_f1_score_pct"] for k in pct_keys]

    ax.plot(recalls, precisions, marker="^", color=PALETTE["purple"], linewidth=2, markersize=8, label="Precision vs Recall")
    for k, rec, prec in zip(pct_keys, recalls, precisions):
        ax.annotate(k, (rec, prec), textcoords="offset points", xytext=(-5, 8), fontsize=9)

    ax.set_xlabel("Zero-Day Recall (%)")
    ax.set_ylabel("Unknown Precision (%)")
    ax.set_title("Unknown Class Precision vs Recall Operational Curve", fontweight="bold")
    ax.legend(frameon=True)
    save_fig(fig, "10_precision_vs_recall_tradeoff.png")

    # -------------------------------------------------------------
    # Plot 11: Master Comparison: Step 6 vs Step 8 vs Step 9 vs Step 10
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    methods = df_master_comp["Method"].values
    x_m = np.arange(len(methods))
    w_m = 0.18

    ax.bar(x_m - 1.5*w_m, df_master_comp["Zero-Day Recall (%)"], w_m, label="Zero-Day Recall (%)", color=PALETTE["danger"])
    ax.bar(x_m - 0.5*w_m, df_master_comp["Unknown Precision (%)"], w_m, label="Unknown Precision (%)", color=PALETTE["primary"])
    ax.bar(x_m + 0.5*w_m, df_master_comp["Unknown F1 (%)"], w_m, label="Unknown F1 (%)", color=PALETTE["success"])
    ax.bar(x_m + 1.5*w_m, df_master_comp["Known Acceptance (%)"], w_m, label="Known Acceptance (%)", color=PALETTE["purple"])

    ax.set_xticks(x_m)
    ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=9.5)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Performance (%)")
    ax.set_title("Master Performance Comparison Across Experimental Pipeline Steps", fontweight="bold")
    ax.legend(frameon=True, loc="lower right")
    save_fig(fig, "11_historical_step_comparison.png")

    # -------------------------------------------------------------
    # Plot 12: Service_Scan Subgroup Recall by XGBoost Predicted Class
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_subgroup, x="predicted_class", y="subgroup_recall_pct", ax=ax, hue="predicted_class", legend=False, palette="coolwarm_r")
    for i, row in df_subgroup.iterrows():
        ax.text(i, row["subgroup_recall_pct"] + 1.5, f"{row['detected_flows']:,} / {row['total_flows']:,}\n({row['subgroup_recall_pct']:.1f}%)", ha="center", fontsize=8.5)
    ax.set_ylim(0, 115)
    ax.set_title("Service_Scan Zero-Day Detection Rate by XGBoost Predicted Class", fontweight="bold")
    ax.set_xlabel("Closed-Set XGBoost Predicted Class")
    ax.set_ylabel("Subgroup Detection Recall (%)")
    plt.xticks(rotation=20, ha="right")
    save_fig(fig, "12_servicescan_subgroup_recall.png")

    # -------------------------------------------------------------
    # Plot 13: OS_Fingerprint Score Distribution (Known vs Service_Scan)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    val_os_scores = df_val[df_val["pred_class"] == "OS_Fingerprint"]["S_unified"]
    zd_os_scores = df_zd[df_zd["pred_class"] == "OS_Fingerprint"]["S_unified"]

    sns.kdeplot(val_os_scores, ax=ax, label=f"Known Traffic -> OS_Fingerprint (N={len(val_os_scores):,})", color=PALETTE["primary"], fill=True, alpha=0.35, linewidth=2)
    sns.kdeplot(zd_os_scores, ax=ax, label=f"Service_Scan -> OS_Fingerprint (N={len(zd_os_scores):,})", color=PALETTE["danger"], fill=True, alpha=0.35, linewidth=2)

    os_row = df_thresholds[df_thresholds["class_name"] == "OS_Fingerprint"]
    if len(os_row) > 0 and "P95.0" in os_row.columns:
        tau_os_p95 = float(os_row["P95.0"].values[0])
        ax.axvline(tau_os_p95, color="black", linestyle="--", linewidth=2, label=f"Adaptive Tau_OS(P95) = {tau_os_p95:.4f}")

    ax.set_title("Novelty Score Distribution for OS_Fingerprint Predicted Subgroup", fontweight="bold")
    ax.set_xlabel("Unified Novelty Score S_unified")
    ax.set_ylabel("Density")
    ax.set_xlim(-0.02, 1.02)
    ax.legend(frameon=True, loc="upper right")
    save_fig(fig, "13_os_fingerprint_score_distribution.png")

    # -------------------------------------------------------------
    # Plot 14: OS_Fingerprint Threshold Boundary & Separation Margin
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    if len(os_row) > 0:
        p_cols = [c for c in df_thresholds.columns if c.startswith("P")]
        p_vals = [float(c.replace("P", "")) for c in p_cols]
        tau_vals = [float(os_row[c].values[0]) for c in p_cols]
        ax.plot(p_vals, tau_vals, marker="o", color=PALETTE["purple"], linewidth=2, label="Adaptive Threshold Tau_OS(p)")

        # Median and P75 of Service_Scan -> OS_Fingerprint
        zd_os_med = np.median(zd_os_scores)
        zd_os_q25 = np.percentile(zd_os_scores, 25.0)
        ax.axhline(zd_os_med, color=PALETTE["danger"], linestyle="--", linewidth=1.5, label=f"Service_Scan OS Median ({zd_os_med:.3f})")
        ax.axhline(zd_os_q25, color=PALETTE["danger"], linestyle=":", linewidth=1.5, label=f"Service_Scan OS Q25 ({zd_os_q25:.3f})")

        ax.set_xlabel("Calibration Percentile p (%)")
        ax.set_ylabel("Novelty Score Threshold Tau_OS")
        ax.set_title("OS_Fingerprint Adaptive Threshold Boundary & Separation Boundary", fontweight="bold")
        ax.legend(frameon=True)
    save_fig(fig, "14_os_fingerprint_threshold_boundary.png")

    # -------------------------------------------------------------
    # Plot 15: Weight Sensitivity / Pareto Frontier
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    if len(df_pareto) > 0:
        f1_col = "f1_balanced" if "f1_balanced" in df_pareto.columns else "pseudo_f1"
        c_col = "j_score" if "j_score" in df_pareto.columns else "benign_rejection"
        sc = ax.scatter(
            df_pareto["known_acceptance"],
            df_pareto[f1_col],
            c=df_pareto[c_col],
            cmap="viridis",
            s=80,
            edgecolors="black",
            alpha=0.85
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Multi-Objective Score J(w)" if c_col == "j_score" else "Benign Rejection Rate (%)")
        ax.set_xlabel("Known Traffic Acceptance Rate (%)")
        ax.set_ylabel("Balanced Pseudo-Unknown Detection F1 (%)")
        ax.set_title("Pareto Frontier of Multi-Signal Weight Candidates", fontweight="bold")
    save_fig(fig, "15_weight_sensitivity_pareto.png")

    # -------------------------------------------------------------
    # Plot 16: Threshold Sensitivity Across Candidate Percentiles
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    p_num = [float(k.replace("P", "")) for k in pct_keys]
    ax.plot(p_num, recalls, marker="o", label="Zero-Day Recall (%)", color=PALETTE["danger"], linewidth=2)
    ax.plot(p_num, precisions, marker="s", label="Unknown Precision (%)", color=PALETTE["primary"], linewidth=2)
    ax.plot(p_num, f1s, marker="^", label="Unknown F1 (%)", color=PALETTE["success"], linewidth=2)
    ax.plot(p_num, acceptances, marker="d", label="Known Acceptance (%)", color=PALETTE["purple"], linewidth=2)

    ax.set_xlabel("Class-Conditional Percentile Cutoff p (%)")
    ax.set_ylabel("Metric Value (%)")
    ax.set_title("Threshold Sensitivity & Performance Trade-Off Curves", fontweight="bold")
    ax.set_ylim(20, 102)
    ax.legend(frameon=True, loc="lower left")
    save_fig(fig, "16_threshold_sensitivity_percentiles.png")

    print("\nAll 16 publication-quality figures successfully generated!")

if __name__ == "__main__":
    run_generate_plots()
