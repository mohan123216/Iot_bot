#!/usr/bin/env python3
"""
03_generate_comprehensive_plots.py: Generate publication-quality visualizations for full dataset LOAO
Produces 6 high-resolution (300 DPI) figures summarizing zero-day detection, false alarms, and weights.
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common_utils import IMPL_DIR, load_config

def run_generate_plots():
    print("=" * 80)
    print(">>> FULL DATASET IMPLEMENTATION: STEP 3 - GENERATE PUBLICATION PLOTS <<<")
    print("=" * 80)

    plots_dir = os.path.join(IMPL_DIR, "outputs", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    master_csv_path = os.path.join(IMPL_DIR, "outputs", "loao_evaluations", "loao_master_results.csv")
    weights_csv_path = os.path.join(IMPL_DIR, "outputs", "weights_and_thresholds", "learned_weights_per_attack.csv")

    if not os.path.exists(master_csv_path):
        raise FileNotFoundError(f"Master results CSV not found at {master_csv_path}")

    df_master = pd.read_csv(master_csv_path)
    df_weights = pd.read_csv(weights_csv_path)

    # Style settings
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.autolayout": True
    })

    attacks = df_master["heldout_attack"].tolist()
    x = np.arange(len(attacks))

    # --- Plot 1: Zero-Day Detection Rate (Recall) by Attack ---
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    recalls = df_master["zero_day_detection_rate_pct"].values
    bars = ax.bar(x, recalls, color="#1f77b4", edgecolor="#0b3954", width=0.6, alpha=0.9, zorder=3)
    macro_mean = np.mean(recalls)
    ax.axhline(macro_mean, color="#d62728", linestyle="--", linewidth=2, label=f"Macro Average: {macro_mean:.2f}%", zorder=4)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(attacks, rotation=25, ha="right")
    ax.set_ylabel("Zero-Day Detection Rate / Recall (%)")
    ax.set_title("Zero-Day Attack Detection Rate Across All 7 Attack Families\n(Evaluated on Entire 74-File UNSW Bot-IoT Dataset)")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)
    ax.legend(loc="lower right")
    p1 = os.path.join(plots_dir, "zero_day_detection_rate_by_attack.png")
    plt.savefig(p1, bbox_inches="tight")
    plt.close()
    print(f"  [1/6] Saved: {p1}")

    # --- Plot 2: Benign False Alarm Rate (FAR) by Held-Out Scenario ---
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    fars = df_master["false_alarm_rate_pct"].values
    bars = ax.bar(x, fars, color="#2ca02c", edgecolor="#144d14", width=0.6, alpha=0.9, zorder=3)
    macro_far = np.mean(fars)
    ax.axhline(macro_far, color="#d62728", linestyle="--", linewidth=2, label=f"Macro Mean FAR: {macro_far:.2f}%", zorder=4)
    ax.axhline(2.0, color="#ff7f0e", linestyle=":", linewidth=2, label="Operational Target Budget (<= 2.0%)", zorder=4)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, h + 0.1, f"{h:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(attacks, rotation=25, ha="right")
    ax.set_ylabel("Benign False Alarm Rate on Normal Traffic (%)")
    ax.set_title("False Alarm Rate (FAR) on Benign Normal IoT Flows\nAcross All Zero-Day Holdout Experiments")
    ax.set_ylim(0, max(5.0, np.max(fars) * 1.3))
    ax.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)
    ax.legend(loc="upper right")
    p2 = os.path.join(plots_dir, "false_alarm_rate_by_attack.png")
    plt.savefig(p2, bbox_inches="tight")
    plt.close()
    print(f"  [2/6] Saved: {p2}")

    # --- Plot 3: Known Attack Acceptance & Classification Accuracy ---
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    width = 0.35
    k_acc = df_master["known_attack_acceptance_rate_pct"].values
    k_cls = df_master["known_attack_classification_acc_pct"].values

    b1 = ax.bar(x - width/2, k_acc, width, label="Known Attack Acceptance Rate", color="#3b528b", edgecolor="#1a253f", alpha=0.9, zorder=3)
    b2 = ax.bar(x + width/2, k_cls, width, label="Closed-Set Classification Accuracy", color="#5dc863", edgecolor="#234c26", alpha=0.9, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(attacks, rotation=25, ha="right")
    ax.set_ylabel("Performance (%)")
    ax.set_title("Known Attack Retention & Classification Fidelity Under Zero-Day Deployment")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)
    ax.legend(loc="lower right")
    p3 = os.path.join(plots_dir, "known_attack_accuracy_by_attack.png")
    plt.savefig(p3, bbox_inches="tight")
    plt.close()
    print(f"  [3/6] Saved: {p3}")

    # --- Plot 4: Optimal Learned Multi-Signal Weights ---
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    w_conf = df_weights["w_confidence"].values
    w_mah = df_weights["w_mahalanobis"].values
    w_leaf = df_weights["w_leaf"].values
    w_rel = df_weights["w_relative"].values

    p_conf = ax.bar(x, w_conf, label="Confidence (S_C)", color="#2b5c8f", edgecolor="#173453", width=0.6, zorder=3)
    p_mah = ax.bar(x, w_mah, bottom=w_conf, label="Mahalanobis (S_M)", color="#e07a5f", edgecolor="#8f3a22", width=0.6, zorder=3)
    p_leaf = ax.bar(x, w_leaf, bottom=w_conf + w_mah, label="Leaf-Space (S_L)", color="#81b29a", edgecolor="#3f6653", width=0.6, zorder=3)
    p_rel = ax.bar(x, w_rel, bottom=w_conf + w_mah + w_leaf, label="Relative Dist (S_R)", color="#f2cc8f", edgecolor="#997538", width=0.6, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(attacks, rotation=25, ha="right")
    ax.set_ylabel("Signal Weight (Sum = 1.0)")
    ax.set_title("Optimal Constrained Multi-Signal Fusion Weights on the 4-Simplex\nLearned per Zero-Day Attack Experiment")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)
    ax.legend(loc="upper right", ncol=2)
    p4 = os.path.join(plots_dir, "learned_weights_distribution.png")
    plt.savefig(p4, bbox_inches="tight")
    plt.close()
    print(f"  [4/6] Saved: {p4}")

    # --- Plot 5: Trade-off Scatter (Zero-Day Detection vs False Alarm Rate) ---
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)
    colors = plt.cm.tab10(np.linspace(0, 1, len(attacks)))

    for i, att in enumerate(attacks):
        r = df_master.iloc[i]
        ax.scatter(r["false_alarm_rate_pct"], r["zero_day_detection_rate_pct"], color=colors[i], s=180, edgecolors="black", linewidths=1.5, zorder=4, label=att)
        ax.annotate(att, (r["false_alarm_rate_pct"], r["zero_day_detection_rate_pct"]),
                    xytext=(6, -2), textcoords="offset points", fontsize=9, fontweight="bold")

    ax.axvline(2.0, color="#d62728", linestyle="--", linewidth=1.5, label="2.0% FAR Budget", zorder=2)
    ax.set_xlabel("Benign False Alarm Rate (%) [Lower is Better]")
    ax.set_ylabel("Zero-Day Detection Rate / Recall (%) [Higher is Better]")
    ax.set_title("Operational Operating Point: Zero-Day Recall vs Benign FAR")
    ax.set_xlim(-0.5, max(5.0, np.max(fars) * 1.2))
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle=":", alpha=0.6)
    p5 = os.path.join(plots_dir, "open_set_tradeoff_scatter.png")
    plt.savefig(p5, bbox_inches="tight")
    plt.close()
    print(f"  [5/6] Saved: {p5}")

    # --- Plot 6: Open-Set AUROC & AUPRC Comparison ---
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    aurocs = df_master["open_set_auroc_pct"].values
    auprcs = df_master["open_set_auprc_pct"].values

    b1 = ax.bar(x - width/2, aurocs, width, label="Open-Set AUROC", color="#440154", edgecolor="#1a0022", alpha=0.9, zorder=3)
    b2 = ax.bar(x + width/2, auprcs, width, label="Open-Set AUPRC", color="#21918c", edgecolor="#0e4340", alpha=0.9, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(attacks, rotation=25, ha="right")
    ax.set_ylabel("Metric (%)")
    ax.set_title("Open-Set Discrimination Fidelity: AUROC & AUPRC Across All Attack Classes")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)
    ax.legend(loc="lower right")
    p6 = os.path.join(plots_dir, "open_set_auroc_auprc_by_attack.png")
    plt.savefig(p6, bbox_inches="tight")
    plt.close()
    print(f"  [6/6] Saved: {p6}")

    print("\nAll 6 publication figures generated successfully in:", plots_dir)

if __name__ == "__main__":
    run_generate_plots()
