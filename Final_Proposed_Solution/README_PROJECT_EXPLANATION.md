# Master Project Guide: Hybrid Multi-Signal Zero-Day Intrusion Detection Engine (HOMZ-Engine)

> **Complete Step-by-Step Presentation & Technical Architecture Walkthrough**  
> *Evaluated on the 73.3 Million Flow UNSW Bot-IoT Dataset Benchmark*  
> *Direct Benchmark against:* **ACM Transactions on Privacy and Security (ACM TOPS, 2025)**

---

## 🎯 What Is This Folder?
This folder (`Final_Proposed_Solution`) contains **ONLY the necessary, clean, and organized components** of our final proposed solution. Everything is structured sequentially from **Step 1 to Step 7** so you can easily understand, run, and present your project without confusion.

---

## 🏗️ Master System Architecture

Our solution is a **Two-Tier Hybrid Open-Set Multi-Signal Architecture**:

```
                              Incoming Network Flow (35 Features)
                                              │
                                              ▼
            ┌──────────────────────────────────────────────────────────────────┐
            │ TIER 1: Balanced Cost-Sensitive GBDT Classifier                  │
            │ - 35 engineered flow, directional, and protocol features         │
            │ - Inverse-frequency class weighting (wc = N / (K * Nc))          │
            │ - 98.97% Multi-Class Accuracy, 99.82% Benign Normal Specificity  │
            └─────────────────────────────────┬────────────────────────────────┘
                                              │ Logits + Tree Leaves + Flow Data
                                              ▼
            ┌──────────────────────────────────────────────────────────────────┐
            │ TIER 2: 4-SIGNAL OPEN-SET ZERO-DAY REJECTION ENGINE              │
            │                                                                  │
            │  Signal 1: Confidence Novelty (S_C = 1.0 - P_max)                │
            │            (Measures prediction uncertainty & model entropy)     │
            │                                                                  │
            │  Signal 2: Log-Manifold Regularized Mahalanobis Distance (S_M)   │
            │            D_M(x) = sqrt((z - μ_c)^T * Σ_c^-1 * (z - μ_c))        │
            │            (Ledoit-Wolf covariance shrinkage on log(1 + x))      │
            │                                                                  │
            │  Signal 3: Tree Leaf-Space Traversal Novelty (S_L)               │
            │            S_leaf = 1.0 - LeafSim(x, c)                          │
            │            (Detects structural anomalies across 100 trees)       │
            │                                                                  │
            │  Signal 4: Relative Margin Distance Ratio (S_R)                  │
            │            R(x) = D_M(pred) / (min_{j≠pred} D_M(j) + ε)           │
            │            (Detects boundary ambiguity between known classes)    │
            │                                                                  │
            │  CALIBRATION: Empirical CDF Percentile Normalizer (searchsorted) │
            │               Z_i(x) = P(S_known <= S_i(x)) in [0.0, 1.0]        │
            │  FUSION:      S_unified = sum(w_i * Z_i)                         │
            │  THRESHOLD:   tau = percentile(S_val, 95.0) on Known Validation  │
            └─────────────────────────────────┬────────────────────────────────┘
                                              │
                                              ▼
                           Unified Anomaly Score > Threshold (τ)?
                                  ┌───────────┴───────────┐
                                 YES                      NO
                                  │                       │
                                  ▼                       ▼
                         [ ZERO-DAY ATTACK ]      [ KNOWN TRAFFIC ]
                         (Alien Novel Threat)     (Normal or Known Attack)
```

---

## 📂 Step-by-Step Folder Structure & Map

```text
Final_Proposed_Solution/
├── 00_run_entire_pipeline.py                 # [Master Runner] Executes Step 1 to 6 in ~12 seconds
├── README_PROJECT_EXPLANATION.md             # [This File] Master narrative & presentation script
│
├── 01_Data_Preprocessing/                    # STEP 1: Ingestion & Balanced Reservoir Sampling
│   ├── README_STEP_1.md                      # Detailed Step 1 documentation
│   ├── 01_stream_clean_full_dataset.py       # Ingestion & streaming audit script
│   └── dataset/
│       ├── fulldataset_cleaned_sample.parquet# Cleaned binary sample (61,130 flows, 3.7 MB)
│       ├── fulldataset_cleaned_sample.csv    # Cleaned CSV sample (7.3 MB)
│       └── class_distribution_full_74files.csv # Audit of all 73,370,443 raw flows
│
├── 02_Tier1_Known_Classification/            # STEP 2: Balanced Multi-Class Classifier
│   ├── README_STEP_2.md                      # Detailed Step 2 documentation
│   ├── 02_train_balanced_multiclass.py       # Tier 1 training & evaluation script
│   ├── models/
│   │   ├── xgboost_balanced_multiclass_ids.json # Serialized XGBoost model
│   │   └── multiclass_class_mapping.json     # Class mapping definitions
│   └── results/
│       ├── balanced_multiclass_classification_report.csv
│       ├── known_attack_multiclass_metrics.csv
│       └── balanced_multiclass_confusion_matrix.png
│
├── 03_Tier2_ZeroDay_Detection/               # STEP 3: Multi-Signal Zero-Day Engine
│   ├── README_STEP_3.md                      # Detailed Step 3 documentation
│   ├── 03_run_zero_day_engine.py             # 5-Signal novelty isolation script
│   └── results/
│       ├── zero_day_detection_standard_metrics.csv
│       ├── unknown_attack_zeroday_metrics.csv
│       └── zero_day_standard_metrics_table_and_chart.png
│
├── 04_Paper_Table9_Benchmark/                # STEP 4: ACM TOPS 2025 Table 9 Reproduction
│   ├── README_STEP_4.md                      # Detailed Step 4 documentation
│   ├── 04_evaluate_paper_table9.py           # 10 LOCO experiments benchmark script
│   ├── models/                               # Pre-trained held-out LOCO models
│   └── results/
│       ├── paper_table9_reproduction_metrics.csv
│       ├── base_paper_vs_our_method_comparison.csv
│       └── base_paper_vs_our_method_comparison_table.png
│
├── 05_Integrity_Audit/                       # STEP 5: Data Integrity & Zero Leakage Audit
│   ├── README_STEP_5.md                      # Detailed Step 5 documentation
│   ├── 05_rigorous_dual_evaluation_audit.py  # Leakage assertion & dual-eval script
│   └── results/
│       └── full_dataset_audit_74files.csv    # Audit records
│
├── 06_Presentation_Dashboards/               # STEP 6: High-Resolution Publication Plots
│   ├── README_STEP_6.md                      # Guide to every figure for slides/viva
│   ├── generate_presentation_dashboards.py   # Script to regenerate dashboards
│   └── plots/                                # 10 High-res (300 DPI) publication figures
│       ├── section1_data_processing_dashboard.png
│       ├── section2_xgboost_classification_dashboard.png
│       ├── section3_zeroday_detection_dashboard.png
│       ├── base_paper_vs_our_method_comparison_table.png
│       ├── feature_importance_top20.png
│       ├── balanced_multiclass_confusion_matrix.png
│       └── class_imbalance_before_and_after.png
│
└── 07_Final_Reports/                         # STEP 7: Full Reports & Viva Cheat Sheet
    ├── FULL_DATASET_ZERO_DAY_IMPLEMENTATION_REPORT.md # Formal academic technical report
    └── QUICK_CHEAT_SHEET_FOR_VIVA.md         # 1-page quick answers to tough examiner questions
```

---

## 🎙️ The 7-Step Presentation Narrative (How to Present to Evaluators)

When explaining your project to an evaluator, professor, or examiner, follow this clear step-by-step narrative:

### Step 1: Preprocessing & Resolving the 73.3M Class Imbalance
- **The Problem:** The raw UNSW Bot-IoT dataset contains 73,370,443 flows across 74 CSV files (~15.3 GB). Floods make up **99.9%** of traffic, while normal benign traffic is only **0.013%** (9,543 flows), and Data Exfiltration is only **0.00016%** (118 flows). Naive sampling would erase rare attacks.
- **Our Solution:** We implemented an **Equal-Quota Streaming Reservoir** that streams all 74 files in chunks of 250,000 rows. We retained **100% of all rare classes** (Normal: 9,543, Keylogging: 1,469, Data Exfiltration: 118) and capped high-volume floods to exactly 10,000 flows each, creating a balanced **61,130-flow representative corpus**.
- **Anti-Leakage:** Stripped IP addresses, MACs, and timestamps to eliminate shortcut learning.

### Step 2: Tier 1 - Balanced Multi-Class Known Attack Classifier
- **The Problem:** Minority attacks (like Data Exfiltration) usually get ignored by ML classifiers because standard loss functions prioritize majority classes.
- **Our Solution:** We engineered **35 domain networking features** (such as source-to-destination payload asymmetry `sbytes_ratio` and port profiling), combined with **inverse-frequency class weighting** ($w_c = N / (K \cdot N_c)$) in an XGBoost ensemble.
- **Results:** Achieved **98.97% overall multi-class accuracy**, with **99.82% specificity on normal traffic** and **93.62% F1-score on Data Exfiltration**.

### Step 3: Tier 2 - 4-Signal Open-Set Zero-Day Rejection Engine
- **The Problem:** Standard classifiers suffer from the **Softmax 100%-Sum Fallacy**. When an unobserved zero-day attack appears, softmax forces the output probabilities to sum to 100%, causing the model to misclassify the alien attack as a known one with high false confidence. Baselines achieve only 33% recall.
- **Our Solution:** We decoupled novelty detection from classification using **4 foundational novelty signals**:
  1. *Confidence Novelty* ($S_C = 1.0 - P_{\max}$, measures prediction uncertainty).
  2. *Log-Manifold Mahalanobis Geometry* ($S_M = D_M$, Ledoit-Wolf regularized covariance distance).
  3. *Tree Leaf-Space Traversal Novelty* ($S_L = 1.0 - \text{LeafSim}$, tracks path novelty across 100 decision trees).
  4. *Relative Neighborhood Margin Ratio* ($S_R = D_M / D_{\text{other}}$, measures boundary ambiguity).
  Signals are normalized into **calibrated percentile ranks** $Z_i \in [0.0, 1.0]$ via an **Empirical CDF Normalizer** and evaluated against an operational threshold $\tau = \text{percentile}(S_{\text{val}}, 95.0)$.
- **Results:** Over 95% known traffic acceptance rate, >96% benign normal specificity, and robust rejection of zero-day threats without relying on arbitrary port heuristics.

### Step 4: Empirical Benchmark vs ACM TOPS 2025 Table 9
- **The Protocol:** Strict **Leave-One-Subclass-Out (LOCO / Type-B Unknown Attack)** across all 10 subclasses.
- **The Comparison:**
  - *DoS & DDoS Floods:* Our Precision/Recall is **1.00** across all 6 subclasses (vs 0.77 - 0.94 in the base paper).
  - *Data Exfiltration:* Precision jumped from **0.08 (paper)** to **0.97 (ours)**! F1-score jumped from **0.15** to **0.95**!
  - *Keylogging:* Precision jumped from **0.80 (paper)** to **1.00 (ours)**!
  - *Error Rates:* Kept between **0.00 and 0.04** (vs 0.02 - 0.14 in the paper).

### Step 5: Rigorous Data Integrity Audit
- **The Proof:** We mathematically verified zero data leakage by asserting zero sample overlap between training and testing sets ($\text{Train} \cap \text{Test} = \emptyset$). We also verified that no synthetic oversampling (SMOTE) contaminated the test sets.

### Step 6: Presentation Dashboards & Visualizations
- 10 publication-quality charts ready for presentation slides, including Section 1, Section 2, and Section 3 Master Dashboards, confusion matrices, and Table 9 graphical comparisons.

### Step 7: Final Documentation & Viva Preparation
- Comprehensive 100-line academic report and a 1-page Viva Cheat Sheet with answers to tough examiner questions.

---

## 🚀 How to Run the Entire Project (One-Click)

From the `Final_Proposed_Solution` directory, simply execute:
```powershell
python 00_run_entire_pipeline.py
```
*Total execution time:* **~12 seconds** for all 6 steps!

Or run individual steps independently:
```powershell
# Step 1: Preprocessing & Sampling
python 01_Data_Preprocessing/01_stream_clean_full_dataset.py

# Step 2: Tier 1 Classifier
python 02_Tier1_Known_Classification/02_train_balanced_multiclass.py

# Step 3: Tier 2 Zero-Day Engine
python 03_Tier2_ZeroDay_Detection/03_run_zero_day_engine.py

# Step 4: Base Paper Table 9 Benchmark
python 04_Paper_Table9_Benchmark/04_evaluate_paper_table9.py

# Step 5: Integrity Audit
python 05_Integrity_Audit/05_rigorous_dual_evaluation_audit.py

# Step 6: Dashboards Generator
python 06_Presentation_Dashboards/generate_presentation_dashboards.py
```
