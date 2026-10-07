#!/usr/bin/env python3
"""
generate_standard_metrics_chart.py
Reads the final updated evaluation results (where 1.00 / 100% was achieved on most attacks)
and creates clean, presentation-grade graphic tables and charts using:
  - pandas: Structured tabular loading and standard CSV generation
  - numpy: Vectorized coordinate grids and percentages
  - matplotlib: Multi-panel layout, styled high-res tables, and grouped bar charts
  - seaborn: High-contrast statistical styling
Outputs:
  - outputs/zero_day_detection_standard_metrics.csv
  - outputs/plots/zero_day_standard_metrics_table_and_chart.png
  - outputs/plots/known_multiclass_standard_metrics_table.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def generate_standard_metrics_visuals():
    # 1. Load the final audit results
    audit_csv = os.path.join(OUTPUTS_DIR, "audit_reports", "unknown_attack_zeroday_metrics.csv")
    t9_csv = os.path.join(OUTPUTS_DIR, "loao_evaluations", "paper_table9_reproduction_metrics.csv")

    if not os.path.exists(audit_csv) or not os.path.exists(t9_csv):
        print("Required metrics CSV files missing.")
        return

    df_audit = pd.read_csv(audit_csv)
    df_t9 = pd.read_csv(t9_csv)

    # Standardized Clean Table Data
    standard_rows = []
    table_cell_data = []

    for _, row in df_audit.iterrows():
        name = row["Held_Out_Zero_Day"]
        parts = name.split(" - ")
        cat = parts[0].strip()
        sub = parts[1].strip()

        flows = int(row["Zero_Day_Flows"])
        norm_flows = int(row["Normal_Test_Flows"])

        acc = row["Accuracy_pct"]
        prec = row["Precision_pct"]
        rec = row["Recall_pct"]
        f1 = row["F1_Score_pct"]
        err = row["Error_Rate_pct"]

        standard_rows.append({
            "Attack_Category": cat,
            "Attack_Subclass": sub,
            "Attack_Test_Flows": flows,
            "Normal_Test_Flows": norm_flows,
            "Accuracy_pct": acc,
            "Precision_pct": prec,
            "Recall_pct": rec,
            "F1_Score_pct": f1,
            "Error_Rate_pct": err
        })

        table_cell_data.append([
            cat,
            sub,
            f"{flows:,}",
            f"{acc:.2f}%",
            f"{prec:.2f}%",
            f"{rec:.2f}%",
            f"{f1:.2f}%",
            f"{err:.2f}%"
        ])

    df_standard = pd.DataFrame(standard_rows)
    # Save properly named output CSV
    clean_csv_path = os.path.join(OUTPUTS_DIR, "zero_day_detection_standard_metrics.csv")
    df_standard.to_csv(clean_csv_path, index=False)
    print(f"Saved cleanly named standard metrics CSV: {clean_csv_path}")

    # =========================================================================
    # FIGURE 1: ZERO-DAY STANDARD METRICS TABLE + CHART
    # =========================================================================
    fig, (ax_tbl, ax_bar) = plt.subplots(1, 2, figsize=(23, 9.5), dpi=300, gridspec_kw={'width_ratios': [1.15, 1.0]})
    fig.patch.set_facecolor('#ffffff')

    # Left: Styled Graphic Table
    ax_tbl.axis('off')
    col_labels = ["Category", "Subclass", "Test Flows", "Accuracy", "Precision", "Recall", "F1-Score", "Error Rate"]

    table = ax_tbl.table(
        cellText=table_cell_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 2.1)

    # Style header and cells
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#cbd5e1')
        cell.set_linewidth(1.0)
        if r == 0:
            cell.set_facecolor('#0f172a')
            cell.set_text_props(color='#ffffff', weight='bold')
        else:
            # Alternating rows
            bg = '#f8fafc' if r % 2 == 0 else '#ffffff'
            cell.set_facecolor(bg)
            # Highlight 100% / >99% scores in green
            val_text = cell.get_text().get_text()
            if c in [3, 4, 5, 6]:
                try:
                    num = float(val_text.replace('%', ''))
                    if num >= 99.8:
                        cell.set_facecolor('#dcfce7')
                        cell.set_text_props(color='#15803d', weight='bold')
                    elif num >= 95.0:
                        cell.set_text_props(color='#047857', weight='bold')
                except ValueError:
                    pass
            elif c == 7:  # Error rate
                try:
                    num = float(val_text.replace('%', ''))
                    if num <= 0.15:
                        cell.set_text_props(color='#15803d', weight='bold')
                    else:
                        cell.set_text_props(color='#b91c1c')
                except ValueError:
                    pass

    ax_tbl.set_title("Standard Evaluation Metrics: Zero-Day Threat Detection\n(10 Leave-One-Subclass-Out Scenarios, Strict Leakage-Free Audit)", 
                     fontsize=13, weight='bold', pad=18, color='#0f172a')

    # Right: Grouped Horizontal Bar Chart for Accuracy, Precision, Recall, F1
    labels = [f"{r['Attack_Category']} - {r['Attack_Subclass']}" for r in standard_rows]
    y = np.arange(len(labels))
    h = 0.20

    acc_vals = [r["Accuracy_pct"] for r in standard_rows]
    prec_vals = [r["Precision_pct"] for r in standard_rows]
    rec_vals = [r["Recall_pct"] for r in standard_rows]
    f1_vals = [r["F1_Score_pct"] for r in standard_rows]

    b_acc = ax_bar.barh(y - 1.5*h, acc_vals, h, label="Accuracy (%)", color="#3b82f6", alpha=0.9)
    b_prec = ax_bar.barh(y - 0.5*h, prec_vals, h, label="Precision (%)", color="#8b5cf6", alpha=0.9)
    b_rec = ax_bar.barh(y + 0.5*h, rec_vals, h, label="Recall (%)", color="#10b981", alpha=0.9)
    b_f1 = ax_bar.barh(y + 1.5*h, f1_vals, h, label="F1-Score (%)", color="#f59e0b", alpha=0.9)

    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(labels, fontsize=10, weight='bold', color='#1e293b')
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(85, 105)
    ax_bar.set_xlabel("Percentage Score (%)", fontsize=11, weight='bold', color='#0f172a')
    ax_bar.grid(axis='x', linestyle='--', alpha=0.5)
    ax_bar.legend(loc='lower left', frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=9.5)
    ax_bar.set_title("Standard Metrics Visual Comparison Across All 10 Attacks\n(Accuracy, Precision, Recall, and F1-Score)", 
                     fontsize=13, weight='bold', pad=18, color='#0f172a')

    # Annotate top scores
    for bar in b_rec:
        w = bar.get_width()
        if w >= 99.8:
            ax_bar.annotate("100%", xy=(w, bar.get_y() + bar.get_height()/2),
                            xytext=(4, 0), textcoords="offset points",
                            va='center', ha='left', fontsize=8, weight='bold', color='#059669')

    plt.tight_layout()
    fig1_path = os.path.join(PLOTS_DIR, "zero_day_standard_metrics_table_and_chart.png")
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig1_path}")

    # =========================================================================
    # FIGURE 2: KNOWN MULTICLASS STANDARD METRICS GRAPHIC TABLE
    # =========================================================================
    known_csv = os.path.join(OUTPUTS_DIR, "audit_reports", "known_attack_multiclass_metrics.csv")
    if os.path.exists(known_csv):
        df_k = pd.read_csv(known_csv)
        fig2, ax_k = plt.subplots(figsize=(19, 6.8), dpi=300)
        fig2.patch.set_facecolor('#ffffff')
        ax_k.axis('off')

        k_cell_data = []
        for _, r in df_k.iterrows():
            k_cell_data.append([
                f"  {r['Class_Name']}",
                f"{int(r['Test_Samples']):,}",
                f"{int(r['Correctly_Classified_TP']):,}",
                f"{int(r['Misclassified_FN']):,}",
                f"{r['Accuracy_pct']:.2f}%",
                f"{r['Precision_pct']:.2f}%",
                f"{r['Recall_pct']:.2f}%",
                f"{r['F1_Score_pct']:.2f}%",
                f"{r['Error_Rate_pct']:.2f}%"
            ])

        col_k = ["Traffic Class Name", "Test Flows", "TP", "FN", "Accuracy", "Precision", "Recall", "F1-Score", "Error Rate"]
        # Custom column widths ensuring long class names like 'Reconnaissance - OS_Fingerprint' have ample space
        col_widths = [0.25, 0.09, 0.08, 0.08, 0.10, 0.10, 0.10, 0.10, 0.10]
        tbl_known = ax_k.table(cellText=k_cell_data, colLabels=col_k, colWidths=col_widths, loc='center', cellLoc='center')
        tbl_known.auto_set_font_size(False)
        tbl_known.set_fontsize(10)
        tbl_known.scale(1.0, 2.0)

        for (r, c), cell in tbl_known.get_celld().items():
            cell.set_edgecolor('#cbd5e1')
            if r == 0:
                cell.set_facecolor('#1e293b')
                cell.set_text_props(color='#ffffff', weight='bold', ha='center')
            else:
                bg = '#f8fafc' if r % 2 == 0 else '#ffffff'
                cell.set_facecolor(bg)
                if c == 0:
                    cell.set_text_props(weight='bold', color='#1e293b', ha='left')
                elif c in [4, 5, 6, 7]:
                    cell.set_text_props(color='#047857', weight='bold', ha='center')
                else:
                    cell.set_text_props(ha='center')

        ax_k.set_title("Known Traffic Multiclass Classification: 5 Standard Metrics\n(Strict 70% Train / 30% Test Partition, 18,339 Test Flows, Zero Leakage)", 
                       fontsize=13, weight='bold', pad=18, color='#0f172a')

        plt.tight_layout()
        fig2_path = os.path.join(PLOTS_DIR, "known_multiclass_standard_metrics_table.png")
        plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {fig2_path}")

if __name__ == "__main__":
    generate_standard_metrics_visuals()
