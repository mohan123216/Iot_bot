# Centralized Experiment Outputs & Visualizations

This directory contains strictly the active evaluation metrics, academic benchmarks, dataset audits, and publication figures for the final solution.

---

## Directory Organization

```text
outputs/
├── audit_reports/
│   ├── known_attack_multiclass_metrics.csv         # Strict 70/30 train/test split multiclass accuracy & per-class metrics
│   └── unknown_attack_zeroday_metrics.csv          # 10 Leave-One-Subclass-Out zero-day threat detection results
│
├── dataset_audit/
│   ├── full_dataset_audit_74files.csv              # Row counts, sizes, and integrity across all 74 raw files (73.3M flows)
│   └── class_distribution_full.csv                 # Detailed traffic frequency breakdown and retention quotas
│
├── loao_evaluations/
│   ├── paper_table9_reproduction_metrics.csv       # Direct replication and outperformance benchmark vs ACM TOPS 2025 Table 9
│   └── loao_ultimate_standard_metrics.csv          # 5 standard metrics (Accuracy, Precision, Recall, F1, Error) per held-out attack
│
├── plots/
│   ├── advanced_vs_baseline_threat_catch.png       # Bar chart: Baseline Recall vs Advanced Threat Catch Rate (85.77% Macro)
│   ├── advanced_vs_baseline_auroc.png              # Bar chart: Open-Set AUROC comparison
│   ├── balanced_multiclass_confusion_matrix.png    # Normalized multiclass confusion matrix heatmap
│   ├── feature_importance_top20.png                # Top 20 most predictive networking features
│   └── standard_metrics_3way_barchart.png          # Side-by-side comparison across all 10 attack classes
│
├── reports/
│   └── FULL_DATASET_ZERO_DAY_IMPLEMENTATION_REPORT.md # Full Markdown research report
│
└── balanced_multiclass_classification_report.csv   # Comprehensive multiclass classification report
```
