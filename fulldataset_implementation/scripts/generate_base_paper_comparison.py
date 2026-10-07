"""
generate_base_paper_comparison.py
Creates a publication-grade visual comparison table and CSV between:
  - Base Paper (ACM TOPS 2025 Table 9)
  - Our Proposed HOMZ-Engine
Calculates the exact increase and improvement across all 10 attack classes.
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

# Data definition
# Base paper values from ACM TOPS 2025 Table 9
base_paper_data = [
    {"class": "DoS", "subclass": "HTTP", "base_prec": 0.91, "base_rec": 0.92, "base_acc": 0.94, "base_f1": 0.91, "base_err": 0.06},
    {"class": "DoS", "subclass": "TCP", "base_prec": 0.77, "base_rec": 0.88, "base_acc": 0.86, "base_f1": 0.82, "base_err": 0.14},
    {"class": "DoS", "subclass": "UDP", "base_prec": 0.88, "base_rec": 0.97, "base_acc": 0.94, "base_f1": 0.92, "base_err": 0.06},
    {"class": "DDoS", "subclass": "HTTP", "base_prec": 0.94, "base_rec": 0.98, "base_acc": 0.97, "base_f1": 0.96, "base_err": 0.03},
    {"class": "DDoS", "subclass": "TCP", "base_prec": 0.81, "base_rec": 0.92, "base_acc": 0.89, "base_f1": 0.86, "base_err": 0.11},
    {"class": "DDoS", "subclass": "UDP", "base_prec": 0.92, "base_rec": 0.96, "base_acc": 0.96, "base_f1": 0.94, "base_err": 0.04},
    {"class": "Reconnaissance", "subclass": "OS_Fingerprint", "base_prec": 0.96, "base_rec": 0.92, "base_acc": 0.95, "base_f1": 0.94, "base_err": 0.05},
    {"class": "Reconnaissance", "subclass": "Service_Scan", "base_prec": 0.84, "base_rec": 0.96, "base_acc": 0.92, "base_f1": 0.89, "base_err": 0.08},
    {"class": "Theft", "subclass": "Keylogging", "base_prec": 0.80, "base_rec": 0.98, "base_acc": 0.98, "base_f1": 0.88, "base_err": 0.02},
    {"class": "Theft", "subclass": "Data_Exfiltration", "base_prec": 0.08, "base_rec": 0.87, "base_acc": 0.93, "base_f1": 0.15, "base_err": 0.07},
]

# Our method values from paper_table9_reproduction_metrics.csv
our_data = {
    "HTTP": {"our_prec": 1.00, "our_rec": 1.00, "our_acc": 1.00, "our_f1": 1.00, "our_err": 0.00},  # DoS
    "TCP": {"our_prec": 1.00, "our_rec": 1.00, "our_acc": 1.00, "our_f1": 1.00, "our_err": 0.00},   # DoS
    "UDP": {"our_prec": 1.00, "our_rec": 1.00, "our_acc": 1.00, "our_f1": 1.00, "our_err": 0.00},   # DoS
    "DDoS-HTTP": {"our_prec": 1.00, "our_rec": 1.00, "our_acc": 1.00, "our_f1": 1.00, "our_err": 0.00},
    "DDoS-TCP": {"our_prec": 1.00, "our_rec": 1.00, "our_acc": 1.00, "our_f1": 1.00, "our_err": 0.00},
    "DDoS-UDP": {"our_prec": 1.00, "our_rec": 1.00, "our_acc": 1.00, "our_f1": 1.00, "our_err": 0.00},
    "OS_Fingerprint": {"our_prec": 1.00, "our_rec": 0.99, "our_acc": 0.99, "our_f1": 1.00, "our_err": 0.01},
    "Service_Scan": {"our_prec": 1.00, "our_rec": 0.95, "our_acc": 0.96, "our_f1": 0.97, "our_err": 0.04},
    "Keylogging": {"our_prec": 1.00, "our_rec": 0.99, "our_acc": 0.99, "our_f1": 0.99, "our_err": 0.01},
    "Data_Exfiltration": {"our_prec": 0.97, "our_rec": 0.93, "our_acc": 1.00, "our_f1": 0.95, "our_err": 0.00},
}

comparison_rows = []
for row in base_paper_data:
    sub = row["subclass"]
    key = f"DDoS-{sub}" if row["class"] == "DDoS" else sub
    ours = our_data[key]
    
    diff_f1 = ours["our_f1"] - row["base_f1"]
    diff_prec = ours["our_prec"] - row["base_prec"]
    diff_rec = ours["our_rec"] - row["base_rec"]
    diff_acc = ours["our_acc"] - row["base_acc"]
    err_reduct = row["base_err"] - ours["our_err"]
    
    comparison_rows.append({
        "Attack_Class": f"{row['class']} - {row['subclass']}",
        "Base_Precision": row["base_prec"],
        "Our_Precision": ours["our_prec"],
        "Diff_Precision": diff_prec,
        "Base_Recall": row["base_rec"],
        "Our_Recall": ours["our_rec"],
        "Diff_Recall": diff_rec,
        "Base_Accuracy": row["base_acc"],
        "Our_Accuracy": ours["our_acc"],
        "Diff_Accuracy": diff_acc,
        "Base_F1": row["base_f1"],
        "Our_F1": ours["our_f1"],
        "F1_Increase": diff_f1,
        "Base_Error": row["base_err"],
        "Our_Error": ours["our_err"],
        "Error_Reduction": err_reduct
    })

df_comp = pd.DataFrame(comparison_rows)
csv_out = os.path.join(OUTPUTS_DIR, "base_paper_vs_our_method_comparison.csv")
df_comp.to_csv(csv_out, index=False)
print(f"Saved comparison CSV: {csv_out}")

# ==============================================================================
# RENDER HIGH-RESOLUTION GRAPHIC COMPARISON TABLE (ZERO OVERFLOW)
# ==============================================================================
fig, ax = plt.subplots(figsize=(22, 7.8), dpi=300)
fig.patch.set_facecolor('#ffffff')
ax.axis('off')

# Build display matrix
col_headers = [
    "Attack Category & Subclass",
    "Base Paper\nPrecision", "Our Method\nPrecision", "Precision\nIncrease (Δ)",
    "Base Paper\nRecall", "Our Method\nRecall", "Recall\nIncrease (Δ)",
    "Base Paper\nF1-Score", "Our Method\nF1-Score", "F1-Score\nIncrease (Δ)",
    "Base Paper\nError", "Our Method\nError", "Error\nReduction"
]

cell_matrix = []
for r in comparison_rows:
    p_inc = f"+{r['Diff_Precision']*100:.1f}%" if r['Diff_Precision'] >= 0 else f"{r['Diff_Precision']*100:.1f}%"
    r_inc = f"+{r['Diff_Recall']*100:.1f}%" if r['Diff_Recall'] >= 0 else f"{r['Diff_Recall']*100:.1f}%"
    f_inc = f"+{r['F1_Increase']*100:.1f}%" if r['F1_Increase'] >= 0 else f"{r['F1_Increase']*100:.1f}%"
    e_red = f"-{r['Error_Reduction']*100:.1f}%" if r['Error_Reduction'] >= 0 else f"+{abs(r['Error_Reduction'])*100:.1f}%"
    
    cell_matrix.append([
        f"  {r['Attack_Class']}",
        f"{r['Base_Precision']:.2f}", f"{r['Our_Precision']:.2f}", p_inc,
        f"{r['Base_Recall']:.2f}", f"{r['Our_Recall']:.2f}", r_inc,
        f"{r['Base_F1']:.2f}", f"{r['Our_F1']:.2f}", f_inc,
        f"{r['Base_Error']:.2f}", f"{r['Our_Error']:.2f}", e_red
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
    f"{avg_base_p:.2f}", f"{avg_our_p:.2f}", f"+{(avg_our_p - avg_base_p)*100:.1f}%",
    f"{avg_base_r:.2f}", f"{avg_our_r:.2f}", f"+{(avg_our_r - avg_base_r)*100:.1f}%",
    f"{avg_base_f1:.2f}", f"{avg_our_f1:.2f}", f"+{(avg_our_f1 - avg_base_f1)*100:.1f}%",
    f"{avg_base_e:.2f}", f"{avg_our_e:.2f}", f"-{(avg_base_e - avg_our_e)*100:.1f}%"
])

# Column widths: 0.20 for Attack Class Name, 0.066 for the 12 numerical columns = 1.00
col_widths = [0.20] + [0.066]*12

tbl = ax.table(
    cellText=cell_matrix,
    colLabels=col_headers,
    colWidths=col_widths,
    loc='center',
    cellLoc='center'
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1.0, 2.05)

# Styling
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor('#cbd5e1')
    cell.set_linewidth(1.0)
    
    # Header row
    if r == 0:
        cell.set_facecolor('#0f172a')
        cell.set_text_props(color='#ffffff', weight='bold', ha='center')
    elif r == len(cell_matrix):  # Summary row
        cell.set_facecolor('#f1f5f9')
        if c == 0:
            cell.set_text_props(weight='bold', color='#0f172a', ha='left')
        elif c in [3, 6, 9, 12]:
            cell.set_facecolor('#dcfce7')
            cell.set_text_props(weight='bold', color='#15803d', ha='center')
        else:
            cell.set_text_props(weight='bold', color='#0f172a', ha='center')
    else:  # Data rows
        bg = '#f8fafc' if r % 2 == 0 else '#ffffff'
        cell.set_facecolor(bg)
        
        if c == 0:
            cell.set_text_props(weight='bold', color='#1e293b', ha='left')
        elif c in [2, 5, 8]:  # Our method columns
            cell.set_text_props(weight='bold', color='#0f172a', ha='center')
        elif c in [3, 6, 9]:  # Increase columns (green highlight)
            val = cell.get_text().get_text()
            if val.startswith("+") and val != "+0.0%":
                cell.set_facecolor('#dcfce7')
                cell.set_text_props(weight='bold', color='#15803d', ha='center')
            else:
                cell.set_text_props(weight='bold', color='#047857', ha='center')
        elif c == 12:  # Error reduction column
            cell.set_facecolor('#eff6ff')
            cell.set_text_props(weight='bold', color='#1d4ed8', ha='center')
        else:
            cell.set_text_props(color='#475569', ha='center')

ax.set_title(
    "Benchmark Performance Comparison: Base Paper (ACM TOPS 2025 Table 9) vs. Proposed HOMZ-Engine\n"
    "Full UNSW Bot-IoT Dataset (74 Files, 73.3M Flows, Leave-One-Attack-Out Protocol)",
    fontsize=14, weight='bold', pad=22, color='#0f172a'
)

plt.tight_layout()
plot_path = os.path.join(PLOTS_DIR, "base_paper_vs_our_method_comparison_table.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved publication comparison chart: {plot_path}")
