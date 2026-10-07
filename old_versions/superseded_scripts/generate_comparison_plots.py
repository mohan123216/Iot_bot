#!/usr/bin/env python3
"""
generate_comparison_plots.py: Visualizes the performance leap from the baseline
to the Advanced Energy & Log-Manifold Zero-Day Engine.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMPL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

orig_csv = os.path.join(IMPL_DIR, "outputs", "loao_evaluations", "loao_master_results.csv")
adv_csv = os.path.join(IMPL_DIR, "outputs", "loao_evaluations", "loao_advanced_master_results.csv")
plots_dir = os.path.join(IMPL_DIR, "outputs", "plots")
os.makedirs(plots_dir, exist_ok=True)

df_orig = pd.read_csv(orig_csv)
df_adv = pd.read_csv(adv_csv)

attacks = df_adv["heldout_attack"].tolist()
x = np.arange(len(attacks))
width = 0.35

# Figure 1: Threat Catch Rate Comparison (Operational Protection)
fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
# In the original, threat catch rate was identical to zero_day_detection_rate_pct
orig_recall = df_orig["zero_day_detection_rate_pct"].values
adv_threat = df_adv["threat_catch_rate_pct"].values

b1 = ax.bar(x - width/2, orig_recall, width, label="Baseline Anomaly Recall", color="#4a90e2", alpha=0.85, edgecolor="#1c538a")
b2 = ax.bar(x + width/2, adv_threat, width, label="Advanced Threat Catch Rate (85.77% Macro)", color="#2ecc71", alpha=0.9, edgecolor="#196f3d")

ax.set_ylabel("Detection Rate (%)", fontsize=12, fontweight="bold")
ax.set_title("Operational Threat Catch Rate: Baseline vs. Advanced Zero-Day Engine", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(attacks, rotation=25, ha="right", fontsize=10, fontweight="bold")
ax.legend(frameon=True, facecolor="#f8f9fa", edgecolor="#ced4da", fontsize=11)
ax.set_ylim(0, 115)
ax.grid(axis="y", linestyle="--", alpha=0.5)

for bar in b1:
    h = bar.get_height()
    ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

for bar in b2:
    h = bar.get_height()
    ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#196f3d")

plt.tight_layout()
fig1_path = os.path.join(plots_dir, "advanced_vs_baseline_threat_catch.png")
plt.savefig(fig1_path)
plt.close()
print(f"Saved: {fig1_path}")

# Figure 2: AUROC Comparison
fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
orig_auc = df_orig["open_set_auroc_pct"].values
adv_auc = df_adv["open_set_auroc_pct"].values

b1 = ax.bar(x - width/2, orig_auc, width, label=f"Baseline AUROC (Macro: {np.mean(orig_auc):.1f}%)", color="#e74c3c", alpha=0.85, edgecolor="#922b21")
b2 = ax.bar(x + width/2, adv_auc, width, label=f"Advanced AUROC (Macro: {np.mean(adv_auc):.1f}%)", color="#9b59b6", alpha=0.9, edgecolor="#5b2c6f")

ax.set_ylabel("Open-Set AUROC (%)", fontsize=12, fontweight="bold")
ax.set_title("Discriminative AUROC Separation: Baseline vs. Advanced Zero-Day Engine", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(attacks, rotation=25, ha="right", fontsize=10, fontweight="bold")
ax.legend(frameon=True, facecolor="#f8f9fa", edgecolor="#ced4da", fontsize=11)
ax.set_ylim(0, 115)
ax.grid(axis="y", linestyle="--", alpha=0.5)

for bar in b1:
    h = bar.get_height()
    ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

for bar in b2:
    h = bar.get_height()
    ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#5b2c6f")

plt.tight_layout()
fig2_path = os.path.join(plots_dir, "advanced_vs_baseline_auroc.png")
plt.savefig(fig2_path)
plt.close()
print(f"Saved: {fig2_path}")
