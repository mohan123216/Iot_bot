# IoT Botnet Zero-Day Attack Detection with Open-Set Recognition

A research and experimental framework for detecting zero-day (novel/unseen) cyber attacks in IoT networks using open-set recognition, geometric distance models, tree leaf-space novelty profiling, and multi-signal fusion.

Developed on the **UNSW Bot-IoT** dataset benchmark.

---

## 📌 Project Overview

Traditional intrusion detection systems (IDS) operate under a **closed-set assumption**—assuming all traffic belongs to previously observed attack categories. This pipeline implements an **open-set recognition framework** that:
1. Classifies known IoT traffic and attack classes with high fidelity.
2. Identifies previously unobserved, zero-day attack flows (such as unseen reconnaissance/scanning behaviors) as **Novel / Zero-Day**.
3. Combines class-conditional Mahalanobis distance geometry, XGBoost tree leaf-space traversal statistics, and classifier confidence.
4. Evaluates leave-one-attack-out generalization across multiple attack families (DDoS HTTP/TCP/UDP, DoS HTTP/TCP/UDP, OS Fingerprinting, Service Scanning).
5. Calibrates adaptive and optimized multi-signal fusion weights (Steps 9 & 10) to maximize zero-day recall while maintaining low false alarm rates on known benign traffic.

---

## 🔬 Experimental Pipeline Structure

All code, configurations, models, figures, and results are located in `experiments/zero_day_detection_pipeline/`:

```text
experiments/zero_day_detection_pipeline/
├── configs/                              # Experiment configuration files (YAML)
├── data/                                 # Dataset splits directory (ignored from Git due to file sizes)
│   ├── cleaned/                          # Preprocessed flows
│   └── splits/                           # Leakage-free train / validation / test splits
├── models/                               # Trained baseline & open-set models
├── outputs/                              # Aggregate pipeline outputs & evaluation metrics
├── predictions/                          # Pipeline prediction files
├── reports/                              # Detailed evaluation reports & findings
├── scripts/                              # Core pipeline execution scripts (Steps 1–8)
├── step4/                                # Distance-based novelty detection (Mahalanobis / Euclidean)
├── step5/                                # Leaf-space novelty detection (tree traversal profiles)
├── step6/                                # Multi-modal hybrid fusion (distance + leaf + confidence)
├── step7/                                # Statistical validation, ablation, & bootstrap testing
├── step8/                                # Class-conditional local density (kNN) & rare-tree novelty
├── step8_attack_generalization/          # Leave-one-attack-out cross-attack generalization
├── step9_weighted_multisignal_novelty/   # Weighted multi-signal novelty fusion
└── step10_optimized_multisignal_novelty/ # Joint optimization & Pareto frontier analysis
```

---

## 📊 Key Results & Artifacts Included

All experimental outputs, metric tables, prediction summaries, and visual plots are tracked in this repository:
- **Metrics & Reports**: Located in each step's `outputs/` and `reports/` directories (`.csv`, `.json`, `.md`).
- **Visualizations**: High-resolution performance curves, ROC/PR plots, detector complementarity, and confusion matrices (`.png`).
- **Prediction Files**: Evaluated prediction tables across test sets and zero-day partitions (`.parquet`, `.csv`).
- **Trained Artifacts**: Class centroids, covariance matrices, and decision tree structures (`.json`, `.pkl`).

---

## 💾 Dataset Information

Due to file size constraints (files exceeding 100 MB), the raw UNSW Bot-IoT dataset files (`UNSW_2018_IoT_Botnet_Full5pc_*.csv`) and generated split parquets in `data/` are excluded from version control via `.gitignore`.

To reproduce the data splits:
1. Download the 5% subset of the UNSW Bot-IoT dataset (`UNSW_2018_IoT_Botnet_Full5pc_*.csv`) from the official UNSW repository.
2. Place the CSV files in the project root directory.
3. Run Step 1 (`scripts/01_dataset_discovery_and_clean.py`) and Step 2 (`scripts/02_create_leakage_free_splits.py`).
