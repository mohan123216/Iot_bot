# Step 6: Publication Dashboards & Visual Presentation Figures

## 📌 Executive Summary
In Step 6, we provide **all publication-grade visual dashboards and presentation charts**. Every single figure is rendered in high resolution (300 DPI) with clean typography, tailored palettes, and zero text collisions. They are ready to be dragged directly into PowerPoint/Google Slides or an academic report.

---

## 🖼️ Catalog of Presentation Figures & Slide Narration

### Figure 1: `section1_data_processing_dashboard.png`
- **What it shows:** 4-panel master dashboard illustrating end-to-end dataset ingestion, the raw 73.3M flow distribution, the before-vs-after log scale imbalance resolution, and the curated 61,130-flow composition.
- **Presentation Narration:**
  > *"This dashboard explains our data pipeline. On the top right, notice the stark contrast between the red bars (raw 73.3M flows dominated by DoS/DDoS) and the teal bars (our balanced 61,130-sample corpus). We maintained 100% of all rare normal and theft flows while capping floods at 10,000 flows each."*

### Figure 2: `section2_xgboost_classification_dashboard.png`
- **What it shows:** 4-panel master dashboard illustrating Tier 1 balanced multi-class classification: feature engineering contributions, training convergence, per-class F1-scores, and the multi-class confusion matrix.
- **Presentation Narration:**
  > *"Here we present Tier 1 of our IDS. Our cost-sensitive XGBoost classifier achieves 98.97% overall multi-class accuracy. Looking at the confusion matrix, benign normal traffic achieves 99.8% precision, and stealthy Data Exfiltration achieves 93.6% F1-score despite having only 24 test instances."*

### Figure 3: `section3_zeroday_detection_dashboard.png`
- **What it shows:** 4-panel master dashboard illustrating Tier 2 open-set zero-day detection: the 5-signal fusion pipeline, the Leave-One-Subclass-Out detection rates, and the head-to-head outperformance vs the 2025 ACM TOPS paper.
- **Presentation Narration:**
  > *"This dashboard demonstrates our Tier 2 Zero-Day Engine. By combining Free-Energy margins with Log-Manifold Mahalanobis distance, we eliminate softmax overconfidence. As shown in the comparison panel, we achieve 100% detection rate on volumetric attacks, solve sister-class shadowing on scans, and rescue Data Exfiltration precision from 0.08 up to 0.97."*

### Figure 4: `base_paper_vs_our_method_comparison_table.png`
- **What it shows:** A high-contrast graphical table directly contrasting our Precision, Recall, F1, and Error Rates against Table 9 of the ACM TOPS 2025 paper across all 10 attack subclasses.
- **Presentation Narration:**
  > *"This figure is our direct paper-to-paper benchmark. Every green highlight indicates where our solution surpasses the ACM TOPS 2025 baseline. Notably, on Data Exfiltration, the baseline suffered an error rate of 0.07 and precision of 0.08, whereas our method achieves 0.97 precision and 0.00 error rate."*

### Figure 5: `feature_importance_top20.png`
- **What it shows:** Top 20 most predictive engineered features ranked by Gain.
- **Key Insight:** `sbytes_per_spkt` (source payload asymmetry), `is_dport_http`, and `log_rate` dominate as the most discriminative signals.

### Figure 6: `balanced_multiclass_confusion_matrix.png`
- **What it shows:** Normalized confusion matrix heatmap across all 8 classes.

### Figure 7: `zero_day_standard_metrics_table_and_chart.png`
- **What it shows:** Standard metrics bar chart across all 10 held-out zero-day attacks.

---

## 🚀 How to Re-generate All Dashboards
From this folder, run:
```powershell
python generate_presentation_dashboards.py
```
*(All 10 figures will be verified and refreshed in `plots/`)*
