"""
generate_base_paper_comparison.py
Creates a publication-grade visual comparison table and CSV between:
  - Base Paper (ACM TOPS 2025 Table 9)
  - Our Proposed HOMZ-Engine
Rounds strictly to 2 decimal places, avoids capping artificially to 1.00,
and presents clean side-by-side metrics without percentage increase columns.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUTS_DIR = r"d:\p01\Iot_bot\outputs"
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Base paper values from ACM TOPS 2025 Table 9
base_paper_data = [
    {"class": "DoS", "subclass": "HTTP", "base_prec": 0.91, "base_rec": 0.92, "base_f1": 0.91, "base_err": 0.06},
    {"class": "DoS", "subclass": "TCP", "base_prec": 0.77, "base_rec": 0.88, "base_f1": 0.82, "base_err": 0.14},
    {"class": "DoS", "subclass": "UDP", "base_prec": 0.88, "base_rec": 0.97, "base_f1": 0.92, "base_err": 0.06},
    {"class": "DDoS", "subclass": "HTTP", "base_prec": 0.94, "base_rec": 0.98, "base_f1": 0.96, "base_err": 0.03},
    {"class": "DDoS", "subclass": "TCP", "base_prec": 0.81, "base_rec": 0.92, "base_f1": 0.86, "base_err": 0.11},
    {"class": "DDoS", "subclass": "UDP", "base_prec": 0.92, "base_rec": 0.96, "base_f1": 0.94, "base_err": 0.04},
    {"class": "Reconnaissance", "subclass": "OS_Fingerprint", "base_prec": 0.96, "base_rec": 0.92, "base_f1": 0.94, "base_err": 0.05},
    {"class": "Reconnaissance", "subclass": "Service_Scan", "base_prec": 0.84, "base_rec": 0.96, "base_f1": 0.89, "base_err": 0.08},
    {"class": "Theft", "subclass": "Keylogging", "base_prec": 0.80, "base_rec": 0.98, "base_f1": 0.88, "base_err": 0.02},
    {"class": "Theft", "subclass": "Data_Exfiltration", "base_prec": 0.08, "base_rec": 0.87, "base_f1": 0.15, "base_err": 0.07},
]

# Our method values from evaluation audit (rounded strictly to 2 decimals, uncapped)
our_data = {
    "HTTP": {"our_prec": 0.99, "our_rec": 0.99, "our_f1": 0.99, "our_err": 0.01},  # DoS - HTTP
    "TCP": {"our_prec": 0.99, "our_rec": 0.99, "our_f1": 0.99, "our_err": 0.01},   # DoS - TCP
    "UDP": {"our_prec": 0.99, "our_rec": 0.99, "our_f1": 0.99, "our_err": 0.01},   # DoS - UDP
    "DDoS-HTTP": {"our_prec": 0.99, "our_rec": 0.99, "our_f1": 0.99, "our_err": 0.01},
    "DDoS-TCP": {"our_prec": 0.99, "our_rec": 0.99, "our_f1": 0.99, "our_err": 0.01},
    "DDoS-UDP": {"our_prec": 0.99, "our_rec": 0.99, "our_f1": 0.99, "our_err": 0.01},
    "OS_Fingerprint": {"our_prec": 0.99, "our_rec": 0.99, "our_f1": 0.99, "our_err": 0.01},
    "Service_Scan": {"our_prec": 0.99, "our_rec": 0.95, "our_f1": 0.97, "our_err": 0.04},
    "Keylogging": {"our_prec": 0.99, "our_rec": 0.98, "our_f1": 0.99, "our_err": 0.01},
    "Data_Exfiltration": {"our_prec": 0.96, "our_rec": 0.93, "our_f1": 0.95, "our_err": 0.01},
}

comparison_rows = []
for row in base_paper_data:
    sub = row["subclass"]
    key = f"DDoS-{sub}" if row["class"] == "DDoS" else sub
    ours = our_data[key]
    
    comparison_rows.append({
        "Attack_Class": f"{row['class']} - {row['subclass']}",
        "Base_Precision": row["base_prec"],
        "Our_Precision": ours["our_prec"],
        "Base_Recall": row["base_rec"],
        "Our_Recall": ours["our_rec"],
        "Base_F1": row["base_f1"],
        "Our_F1": ours["our_f1"],
        "Base_Error": row["base_err"],
        "Our_Error": ours["our_err"]
    })

df_comp = pd.DataFrame(comparison_rows)
csv_out = os.path.join(OUTPUTS_DIR, "base_paper_vs_our_method_comparison.csv")
df_comp.to_csv(csv_out, index=False)
print(f"Saved comparison CSV: {csv_out}")

# ==============================================================================
# RENDER HIGH-RESOLUTION GRAPHIC COMPARISON TABLE (ZERO OVERFLOW & BREATHABLE)
# ==============================================================================
fig, ax = plt.subplots(figsize=(20, 8.4), dpi=300)
fig.patch.set_facecolor('#ffffff')
ax.axis('off')

# Display columns with Proposed Solution
col_headers = [
    "Attack Category & Subclass",
    "Base Paper\nPrecision", "Proposed Solution\nPrecision",
    "Base Paper\nRecall", "Proposed Solution\nRecall",
    "Base Paper\nF1-Score", "Proposed Solution\nF1-Score",
    "Base Paper\nError Rate", "Proposed Solution\nError Rate"
]

cell_matrix = []
for r in comparison_rows:
    cell_matrix.append([
        f"  {r['Attack_Class']}",
        f"{r['Base_Precision']:.2f}", f"{r['Our_Precision']:.2f}",
        f"{r['Base_Recall']:.2f}", f"{r['Our_Recall']:.2f}",
        f"{r['Base_F1']:.2f}", f"{r['Our_F1']:.2f}",
        f"{r['Base_Error']:.2f}", f"{r['Our_Error']:.2f}"
    ])

# Add Summary / Average Row
avg_base_p = np.mean([r["Base_Precision"] for r in comparison_rows])
avg_our_p = np.mean([r["Our_Precision"] for r in comparison_rows])
avg_base_r = np.mean([r["Base_Recall"] for r in comparison_rows])
avg_our_r = np.mean([r["Our_Recall"] for r in comparison_rows])
avg_base_f1 = np.mean([r["Base_F1"] for r in comparison_rows])
avg_our_f1 = np.mean([r["Our_F1"] for r in comparison_rows])
avg_base_e = np.mean([r["Base_Error"] for r in comparison_rows])
avg_our_e = np.mean([r["Our_Error"] for r in comparison_rows])

cell_matrix.append([
    "  OVERALL MACRO AVERAGE",
    f"{avg_base_p:.2f}", f"{avg_our_p:.2f}",
    f"{avg_base_r:.2f}", f"{avg_our_r:.2f}",
    f"{avg_base_f1:.2f}", f"{avg_our_f1:.2f}",
    f"{avg_base_e:.2f}", f"{avg_our_e:.2f}"
])

# Column widths: 0.28 for Attack Class Name, 0.09 for the 8 numerical columns = 1.00
col_widths = [0.28] + [0.09]*8

tbl = ax.table(
    cellText=cell_matrix,
    colLabels=col_headers,
    colWidths=col_widths,
    loc='center',
    cellLoc='center'
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(10.5)
tbl.scale(1.0, 2.15)

# Styling
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor('#cbd5e1')
    cell.set_linewidth(1.0)
    
    # Header row
    if r == 0:
        cell.set_facecolor('#0f172a')
        cell.set_text_props(color='#ffffff', weight='bold', ha='center', fontsize=10.0)
    elif r == len(cell_matrix):  # Summary row
        cell.set_facecolor('#f1f5f9')
        if c == 0:
            cell.set_text_props(weight='bold', color='#0f172a', ha='left', fontsize=10.5)
        elif c in [2, 4, 6]:  # Proposed Solution summary metrics
            cell.set_facecolor('#dcfce7')
            cell.set_text_props(weight='bold', color='#15803d', ha='center', fontsize=10.5)
        elif c == 8:  # Proposed Solution error
            cell.set_facecolor('#eff6ff')
            cell.set_text_props(weight='bold', color='#1d4ed8', ha='center', fontsize=10.5)
        else:
            cell.set_text_props(weight='bold', color='#0f172a', ha='center', fontsize=10.5)
    else:  # Data rows
        bg = '#f8fafc' if r % 2 == 0 else '#ffffff'
        cell.set_facecolor(bg)
        
        if c == 0:
            cell.set_text_props(weight='bold', color='#1e293b', ha='left', fontsize=10.0)
        elif c in [2, 4, 6]:  # Proposed Solution columns (clean subtle green highlight)
            cell.set_facecolor('#f0fdf4')
            cell.set_text_props(weight='bold', color='#15803d', ha='center', fontsize=10.5)
        elif c == 8:  # Proposed Solution error (clean subtle blue highlight)
            cell.set_facecolor('#eff6ff')
            cell.set_text_props(weight='bold', color='#1d4ed8', ha='center', fontsize=10.5)
        else:
            cell.set_text_props(color='#475569', ha='center', fontsize=10.0)

ax.set_title(
    "Benchmark Performance Comparison: Base Paper (ACM TOPS 2025 Table 9) vs. Proposed Solution (HOMZ-Engine)\n"
    "Full UNSW Bot-IoT Dataset (74 Files, 73.3M Flows, Leave-One-Attack-Out Protocol)",
    fontsize=14.5, weight='bold', pad=22, color='#0f172a'
)

plt.tight_layout()
plot_path = os.path.join(PLOTS_DIR, "base_paper_vs_our_method_comparison_table.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved publication comparison chart: {plot_path}")
