# Step 2 Report: Leakage-Free Train/Validation/Test Splitting and Zero-Day Holdout Design

**Experiment**: Zero-Day Attack Detection Pipeline (Step 2)  
**Directory**: `experiments/zero_day_detection_pipeline/`  
**Configuration**: [`configs/experiment_config.yaml`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/configs/experiment_config.yaml)  
**Date**: 2026-10-03  
**Status**: Completed (Programmatic Leakage Audit: PASS)  

---

## 1. Executive Summary

This report establishes the experimental partitioning protocol for the **Zero-Day Attack Detection Pipeline**.
The objective of this stage is to simulate a scientifically valid, open-set operational environment where the intrusion detection system must evaluate an attack class that was **100% unseen** during model training, calibration, threshold tuning, and feature preprocessing.

### Key Milestones Accomplished:
- **Cleaned Dataset Partitioned**: **367,585 flow records** partitioned into four mutually exclusive, non-overlapping subsets.
- **Designated Zero-Day Class**: **`Service_Scan`** (7,302 flows, 1.99% of cleaned corpus) held out strictly into the Zero-Day Test set.
- **Stratified Known Partitioning**: 360,283 known flows partitioned into **70% Training (252,198 flows)**, **15% Validation (54,042 flows)**, and **15% Known Test (54,043 flows)** with all 7 known subcategories preserved across all three splits.
- **Strict Leakage Prevention**: Programmatic audit verified **zero samples** of `Service_Scan` in Train, Validation, or Known Test, and **zero index overlap** across splits.
- **Zero Preprocessing Leakage**: Feature scaling statistics (means, standard deviations, medians, IQRs) were computed strictly on the **Training set** and frozen.

---

## 2. Zero-Day Class Selection & Candidate Analysis

To model realistic open-set conditions, the fine-grained `subcategory` label was selected as the operational target. The table below details all candidate zero-day classes available in the cleaned BoT-IoT corpus:

| candidate_class   |   sample_count |   percentage_of_total | is_selected_zero_day   |
|:------------------|---------------:|----------------------:|:-----------------------|
| Service_Scan      |           7302 |                1.9865 | True                   |
| OS_Fingerprint    |           1777 |                0.4834 | False                  |
| HTTP              |            266 |                0.0724 | False                  |
| Keylogging        |             73 |                0.0199 | False                  |
| Data_Exfiltration |              6 |                0.0016 | False                  |

### Explicit Decision Rule for Zero-Day Selection:
1. **Statistical Significance Requirement**: Standalone evaluation of an unseen attack requires a sample size large enough ($N \ge 1,000$) to calculate robust receiver operating characteristics (AUROC, AUPRC) and reliable confidence intervals.
2. **Behavioral Divergence**: The held-out attack should exhibit network characteristics distinct from background volumetric floods (`UDP`, `TCP`) to test true out-of-distribution detection.
3. **Selection**: **`Service_Scan`** (7,302 flows) was selected because it represents active network probing and reconnaissance (port scanning, service discovery), providing an independent evaluation cohort that never contaminates training or threshold calibration.

---

## 3. Strict Data Partition Structure

```text
                               CLEANED DATASET (N=367,585)
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     │                                             │
             ZERO-DAY CLASS                                  KNOWN CLASSES
         (Service_Scan: 7,302)                           (7 Classes: 360,283)
                     │                                             │
                     │                           ┌─────────────────┼─────────────────┐
                     │                           │                 │                 │
                     ▼                           ▼                 ▼                 ▼
             ZERODAY TEST SET                 TRAIN           VALIDATION        KNOWN TEST
                (7,302 flows)            (252,198 flows)   (54,042 flows)   (54,043 flows)
                 [Final Eval]              [70% Stratified]  [15% Stratified]  [15% Stratified]
```

### Partition Role Specifications:
1. **Training Set (`train.parquet`)**:
   - Contains ONLY known attack classes (`UDP`, `TCP`, `OS_Fingerprint`, `HTTP`, `Keylogging`, `Data_Exfiltration`) and benign `Normal` traffic.
   - Purpose: Classifier training, training-only sample weighting, feature representation fitting.
2. **Validation Set (`validation.parquet`)**:
   - Contains ONLY known classes.
   - Purpose: Hyperparameter optimization, probability calibration, novelty score distribution estimation, adaptive threshold tuning.
3. **Known Test Set (`known_test.parquet`)**:
   - Contains ONLY known classes.
   - Purpose: Final closed-set classification accuracy, known-class retention rate, benign false alarm evaluation ($	ext{FPR}_{	ext{Normal}}$).
4. **Zero-Day Test Set (`zeroday_test.parquet`)**:
   - Contains ONLY `Service_Scan` flows.
   - Purpose: Final unbiased evaluation of open-set zero-day detection and wrong-known mapping rate.

---

## 4. Class Distribution Across Splits

The table below demonstrates the exact stratification across partitions:

| class             |   total_cleaned_count |   train_count |   train_pct_within_split |   validation_count |   val_pct_within_split |   known_test_count |   test_pct_within_split |   zeroday_test_count | is_zero_day   |
|:------------------|----------------------:|--------------:|-------------------------:|-------------------:|-----------------------:|-------------------:|------------------------:|---------------------:|:--------------|
| Data_Exfiltration |                     6 |             4 |                   0.0016 |                  1 |                 0.0019 |                  1 |                  0.0019 |                    0 | False         |
| HTTP              |                   266 |           186 |                   0.0738 |                 40 |                 0.074  |                 40 |                  0.074  |                    0 | False         |
| Keylogging        |                    73 |            51 |                   0.0202 |                 11 |                 0.0204 |                 11 |                  0.0204 |                    0 | False         |
| Normal            |                   477 |           334 |                   0.1324 |                 72 |                 0.1332 |                 71 |                  0.1314 |                    0 | False         |
| OS_Fingerprint    |                  1777 |          1244 |                   0.4933 |                266 |                 0.4922 |                267 |                  0.4941 |                    0 | False         |
| Service_Scan      |                  7302 |             0 |                   0      |                  0 |                 0      |                  0 |                  0      |                 7302 | True          |
| TCP               |                159340 |        111538 |                  44.2264 |              23901 |                44.2267 |              23901 |                 44.2259 |                    0 | False         |
| UDP               |                198344 |        138841 |                  55.0524 |              29751 |                55.0516 |              29752 |                 55.0525 |                    0 | False         |

---

## 5. Training Set Class Imbalance Analysis

In accordance with strict experimental rules, **no class balancing (SMOTE, oversampling, or undersampling) was performed in Step 2.**

The empirical distribution of the Training set is detailed below:

| class             |   train_count |   train_percentage |   majority_to_class_ratio | imbalance_severity      |
|:------------------|--------------:|-------------------:|--------------------------:|:------------------------|
| UDP               |        138841 |            55.0524 |                      1    | Baseline Majority       |
| TCP               |        111538 |            44.2264 |                      1.24 | Balanced / High Support |
| OS_Fingerprint    |          1244 |             0.4933 |                    111.61 | Moderate Imbalance      |
| Normal            |           334 |             0.1324 |                    415.69 | Severe Imbalance        |
| HTTP              |           186 |             0.0738 |                    746.46 | Severe Imbalance        |
| Keylogging        |            51 |             0.0202 |                   2722.37 | Extreme Imbalance       |
| Data_Exfiltration |             4 |             0.0016 |                  34710.2  | Extreme Imbalance       |

### Key Imbalance Observations:
- **Volumetric Dominance**: `UDP` (55.05%) and `TCP` (44.23%) account for **99.28%** of the entire training partition.
- **Benign Scarcity**: Benign `Normal` traffic accounts for only **334 flows (0.132%)**, yielding an imbalance ratio of **415.7 : 1**.
- **Theft Scarcity**: `Data_Exfiltration` has only 4 training flows, and `Keylogging` has 51 flows.

### Planned Imbalance Handling for Step 3:
1. **Cost-Sensitive Class Weights**: Assign inverse-frequency weights to the XGBoost multiclass objective:
   $$w_c = \frac{N_{\text{train}}}{C \cdot N_c}$$
2. **Stratified Mini-Batch Sampling**: Ensure minority classes are sampled proportionally during model optimization.
3. **Benign Protection Boundary**: Treat `Normal` traffic as a protected class with dedicated acceptance thresholds to guarantee $\text{FPR}_{\text{Normal}} \le 3.0\%$.

---

## 6. Programmatic Leakage Audit Results

The automated leakage audit executed with **`STATUS: PASS`**:

| Verification Check | Target Condition | Measured Value | Result |
| :--- | :---: | :---: | :---: |
| Zero-Day samples in Training set | 0 | 0 | **PASS** |
| Zero-Day samples in Validation set | 0 | 0 | **PASS** |
| Zero-Day samples in Known Test set | 0 | 0 | **PASS** |
| Zero-Day samples in Zero-Day Test set | 7302 | 7302 | **PASS** |
| Train & Validation index overlap | False | False | **PASS** |
| Train & Known Test index overlap | False | False | **PASS** |
| Validation & Known Test index overlap | False | False | **PASS** |
| Known classes match across splits | True | True | **PASS** |
| Feature preprocessing leakage | False | False | **PASS** |

---

## 7. Dual Feature Representation Architecture

Two separate feature subsets were prepared and configured in [`configs/experiment_config.yaml`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/configs/experiment_config.yaml):

1. **Representation A (XGBoost Predictive Features, 34 Features)**:
   Includes flow rates, packet sizes, durations, window statistics, port fields (`sport`, `dport`), and protocol states. Excludes IP addresses, timestamps, sequence numbers, and ground-truth label columns.
2. **Representation B (Geometry & Novelty Distance Features, 29 Continuous Features)**:
   Focuses strictly on continuous behavioral flow dynamics (duration, packet counts, byte counts, flow rates, arrival rates, window metrics). Excludes discrete port numbers to ensure robust Mahalanobis and Euclidean distance metric spaces.

---

## 8. Saved Artifacts

### Partition Datasets:
- Training: [`data/splits/train.parquet`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/data/splits/train.parquet) (252,198 flows)
- Validation: [`data/splits/validation.parquet`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/data/splits/validation.parquet) (54,042 flows)
- Known Test: [`data/splits/known_test.parquet`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/data/splits/known_test.parquet) (54,043 flows)
- Zero-Day Test: [`data/splits/zeroday_test.parquet`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/data/splits/zeroday_test.parquet) (7,302 flows)

### Summary & Audit Tables:
- [`outputs/split_summary.csv`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/outputs/split_summary.csv)
- [`outputs/split_class_distribution.csv`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/outputs/split_class_distribution.csv)
- [`outputs/zero_day_holdout_summary.csv`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/outputs/zero_day_holdout_summary.csv)
- [`outputs/imbalance_analysis.csv`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/outputs/imbalance_analysis.csv)
- [`outputs/leakage_audit.json`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/outputs/leakage_audit.json)
- [`outputs/training_feature_statistics.json`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/outputs/training_feature_statistics.json)
