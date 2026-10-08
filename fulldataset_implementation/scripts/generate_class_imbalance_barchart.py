"""
generate_class_imbalance_barchart.py
Publication-grade visual comparison showing:
  1. Extreme Class Imbalance in Raw UNSW Bot-IoT Dataset (73,370,443 flows)
  2. Balanced Multi-Class Representation After Streaming Stratified Sampling (61,130 flows)
Zero text overlap, breathable layout, and clear typography.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

OUTPUT_DIR = r"d:\p01\Iot_bot\outputs\plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dataset Statistics across all 74 files
classes = [
    "UDP Flood",
    "TCP Flood",
    "Service_Scan",
    "OS_Fingerprint",
    "HTTP Flood",
    "Normal (Benign)",
    "Keylogging",
    "Data_Exfiltration"
]

raw_counts = [39624597, 31863600, 1463364, 358275, 49477, 9543, 1469, 118]
raw_pcts = [54.01, 43.43, 1.99, 0.49, 0.07, 0.013, 0.002, 0.00016]

balanced_counts = [10000, 10000, 10000, 10000, 10000, 9543, 1469, 118]
balanced_pcts = [16.36, 16.36, 16.36, 16.36, 16.36, 15.61, 2.40, 0.19]
retention_notes = ["0.03% sampled", "0.03% sampled", "0.68% sampled", "2.79% sampled", "20.21% sampled", "100% RETAINED", "100% RETAINED", "100% RETAINED"]

# Color palette
C_DARK = "#0F172A"
C_BLUE = "#2563EB"
C_TEAL = "#0D9488"
C_GREEN = "#16A34A"
C_RED = "#DC2626"
C_AMBER = "#D97706"
C_PURPLE = "#7C3AED"
C_GRAY_LIGHT = "#CBD5E1"
C_GRAY_TEXT = "#64748B"
C_CARD_BG = "#F8FAFC"

bar_colors = [
    "#3B82F6",  # UDP
    "#2563EB",  # TCP
    "#6366F1",  # Service Scan
    "#8B5CF6",  # OS Fingerprint
    "#0EA5E9",  # HTTP
    C_GREEN,    # Normal
    C_PURPLE,   # Keylogging
    C_AMBER     # Data Exfil
]

fig = plt.figure(figsize=(20, 11.2), facecolor="#FFFFFF")
gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 0.22], width_ratios=[1.0, 1.0],
                       left=0.08, right=0.96, top=0.85, bottom=0.08, wspace=0.28, hspace=0.35)

# Main Titles
fig.text(0.08, 0.960, "UNSW BOT-IOT DATASET: CLASS DISTRIBUTION BEFORE VS. AFTER BALANCING",
         fontsize=17, fontweight='bold', color=C_DARK)
fig.text(0.08, 0.925, "Streaming Stratified Ingestion: Resolving the 335,000 : 1 Imbalance Across 73,370,443 Flows",
         fontsize=11.5, color=C_GRAY_TEXT)

y_pos = np.arange(len(classes))

# ------------------------------------------------------------------------------
# PANEL 1: RAW DATASET (SEVERE IMBALANCE - LOG SCALE)
# ------------------------------------------------------------------------------
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor("#FFFFFF")
for spine in ax1.spines.values():
    spine.set_color(C_GRAY_LIGHT)

bars1 = ax1.barh(y_pos, raw_counts, color=bar_colors, height=0.62, edgecolor=C_DARK, linewidth=0.6)
ax1.set_xscale("log")
ax1.set_yticks(y_pos)
ax1.set_yticklabels(classes, fontsize=10.5, fontweight='bold', color=C_DARK)
ax1.invert_yaxis()
ax1.set_xlim(10, 2e8)
ax1.set_xlabel("Flow Count (Logarithmic Scale)", fontsize=10.5, fontweight='bold', color=C_DARK)
ax1.set_title("BEFORE: Raw Dataset (73,370,443 Total Flows across 74 CSVs)\nExtreme Flood Domination (Floods = 99.9% | Normal = 0.013%)",
              fontsize=11.5, fontweight='bold', color=C_DARK, pad=12)
ax1.grid(axis='x', linestyle='--', alpha=0.5)

for i, bar in enumerate(bars1):
    w = bar.get_width()
    pct = raw_pcts[i]
    if pct >= 1.0:
        pct_str = f"{pct:.1f}%"
    elif pct >= 0.01:
        pct_str = f"{pct:.3f}%"
    else:
        pct_str = f"{pct:.5f}%"
    
    txt = f"  {w:,}  ({pct_str})"
    ax1.text(w, bar.get_y() + bar.get_height()/2, txt,
             va='center', ha='left', fontsize=9.0, fontweight='bold', color=C_DARK)

# ------------------------------------------------------------------------------
# PANEL 2: BALANCED DATASET (STRATIFIED CURATED SAMPLE - LINEAR SCALE)
# ------------------------------------------------------------------------------
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor("#FFFFFF")
for spine in ax2.spines.values():
    spine.set_color(C_GRAY_LIGHT)

bars2 = ax2.barh(y_pos, balanced_counts, color=bar_colors, height=0.62, edgecolor=C_DARK, linewidth=0.6)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(classes, fontsize=10.5, fontweight='bold', color=C_DARK)
ax2.invert_yaxis()
ax2.set_xlim(0, 14200)
ax2.set_xlabel("Curated Flow Count (Linear Scale)", fontsize=10.5, fontweight='bold', color=C_DARK)
ax2.set_title("AFTER: Curated Balanced Dataset (61,130 Total Flows)\nCapped High-Volume Floods + 100% Minority Retention",
              fontsize=11.5, fontweight='bold', color=C_DARK, pad=12)
ax2.grid(axis='x', linestyle='--', alpha=0.5)

for i, bar in enumerate(bars2):
    w = bar.get_width()
    pct = balanced_pcts[i]
    note = retention_notes[i]
    txt = f"  {w:,} ({pct:.1f}%) — {note}"
    color_txt = C_GREEN if "RETAINED" in note else C_DARK
    ax2.text(w, bar.get_y() + bar.get_height()/2, txt,
             va='center', ha='left', fontsize=9.0, fontweight='bold', color=color_txt)

# ------------------------------------------------------------------------------
# PANEL 3: SUMMARY INFORMATION CARD
# ------------------------------------------------------------------------------
ax_card = fig.add_subplot(gs[1, :])
ax_card.axis('off')

card_box = FancyBboxPatch((0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0.02",
                          facecolor=C_CARD_BG, edgecolor=C_GRAY_LIGHT, linewidth=1.2,
                          transform=ax_card.transAxes)
ax_card.add_patch(card_box)

key_takeaways = (
    r"$\bf{Key\ Stratified\ Balancing\ Invariants:}$" "\n"
    "•  1. Massive Flood Capping: Raw UDP (39.6M) and TCP (31.8M) accounted for 97.4% of all packets; both capped to exactly 10,000 flows using dynamic streaming sampling.\n"
    "•  2. 100% Rare Minority Preservation: Zero sampling applied to Normal (9,543), Keylogging (1,469), and Data Exfiltration (118) to preserve scarce zero-day patterns.\n"
    "•  3. Imbalance Ratio Reduction: The extreme raw ratio of 335,801 : 1 (UDP vs Exfiltration) is safely rebalanced to 84 : 1 without any synthetic sample distortion."
)

ax_card.text(0.03, 0.50, key_takeaways, va='center', ha='left', fontsize=10.2, color=C_DARK,
             transform=ax_card.transAxes, linespacing=1.6)

out_file = os.path.join(OUTPUT_DIR, "class_imbalance_before_and_after.png")
plt.savefig(out_file, dpi=300)
plt.close(fig)
print(f"Generated class imbalance comparison barchart: {out_file}")
