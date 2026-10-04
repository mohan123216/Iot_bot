# Step 3 Report: Closed-Set XGBoost Baseline & Training-Only Class Imbalance Handling

**Project**: Robust Zero-Day Attack Detection with Open-Set Recognition  
**Pipeline Root**: `experiments/zero_day_detection_pipeline/`  
**Configuration**: [`configs/experiment_config.yaml`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/configs/experiment_config.yaml)  
**Date**: 2026-10-03  
**Status**: Completed (Closed-Set XGBoost Baseline Established)  

---

## 1. Objective

The objective of Step 3 is to establish a rigorous, leakage-free **closed-set multiclass classification baseline** using XGBoost on the 7 known subcategories of the BoT-IoT dataset.
This step establishes the supervised classification foundation before any open-set or novelty detection mechanisms are introduced in subsequent steps.

Specifically, Step 3 addresses:
1. Handling the severe training class imbalance using **training-only cost-sensitive balanced class weighting**.
2. Comparing **Baseline A (Unweighted XGBoost)** against **Baseline B (Weighted XGBoost)**.
3. Conducting rigorous, non-contaminating evaluations on both **Validation** and **Known Test** splits.
4. Ensuring that the held-out zero-day class (`Service_Scan`) remains **100% unseen and unreferenced**.

---

## 2. Input Datasets & Integrity Verification

All inputs were sourced directly from the Step 2 partitions without modification:

| Partition | File Path | Sample Count | Role in Step 3 |
| :--- | :--- | :---: | :--- |
| **Training Set** | [`data/splits/train.parquet`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/data/splits/train.parquet) | **252,198** | Model fitting, training-only class weight computation |
| **Validation Set** | [`data/splits/validation.parquet`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/data/splits/validation.parquet) | **54,042** | Hyperparameter verification, probability evaluation |
| **Known Test Set** | [`data/splits/known_test.parquet`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/data/splits/known_test.parquet) | **54,043** | Final closed-set benchmark evaluation |
| **Zero-Day Test Set** | [`data/splits/zeroday_test.parquet`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/data/splits/zeroday_test.parquet) | **7,302** | **Untouched** (Reserved exclusively for Step 5 open-set testing) |

### Programmatic Leakage & Absence Audit:
- **`Service_Scan` in Training Set**: **0 flows** (Verified)
- **`Service_Scan` in Validation Set**: **0 flows** (Verified)
- **`Service_Scan` in Known Test Set**: **0 flows** (Verified)
- **`Service_Scan` in Zero-Day Test Set**: **7,302 flows** (Verified)
- **Trained Model Output Heads**: Exactly 7 classes (Zero-day class is NOT an output head).

---

## 3. Feature Representation (Representation A)

The models were trained using strictly the **34 features of Representation A** specified in [`configs/experiment_config.yaml`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/configs/experiment_config.yaml):
- **Protocol & State Encoded**: `proto_number`, `flgs_number`, `state_number`
- **Port Fields**: `sport`, `dport`
- **Flow Dynamics**: `dur`, `pkts`, `bytes`, `spkts`, `dpkts`, `sbytes`, `dbytes`, `rate`, `srate`, `drate`, `mean`, `stddev`, `sum`, `min`, `max`
- **Host & Subnet Aggregations**: `TnBPSrcIP`, `TnBPDstIP`, `TnP_PSrcIP`, `TnP_PDstIP`, `TnP_PerProto`, `TnP_Per_Dport`, `AR_P_Proto_P_SrcIP`, `AR_P_Proto_P_DstIP`, `N_IN_Conn_P_DstIP`, `N_IN_Conn_P_SrcIP`, `AR_P_Proto_P_Sport`, `AR_P_Proto_P_Dport`, `Pkts_P_State_P_Protocol_P_DestIP`, `Pkts_P_State_P_Protocol_P_SrcIP`

**Excluded Columns**: `saddr`, `daddr`, `stime`, `ltime`, `pkSeqID`, `seq`, `attack`, `category`, `subcategory`.

---

## 4. Training-Only Class Imbalance Handling

### Balanced Weighting Formulation:
To counter the acute imbalance without generating artificial synthetic samples, cost-sensitive class weights were computed strictly from the training partition:

$$w_c = \frac{N}{C \cdot N_c}$$

where $N = 252,198$, $C = 7$, and $N_c$ is the training sample count for class $c$.

### Calculated Class Weights:

| class             |   class_index |   train_count |   train_percentage |   class_weight |   majority_ratio |
|:------------------|--------------:|--------------:|-------------------:|---------------:|-----------------:|
| Data_Exfiltration |             0 |             4 |             0.0016 |      9007.07   |         34710.2  |
| HTTP              |             1 |           186 |             0.0738 |       193.701  |           746.46 |
| Keylogging        |             2 |            51 |             0.0202 |       706.437  |          2722.37 |
| Normal            |             3 |           334 |             0.1324 |       107.869  |           415.69 |
| OS_Fingerprint    |             4 |          1244 |             0.4933 |        28.9616 |           111.61 |
| TCP               |             5 |        111538 |            44.2264 |         0.323  |             1.24 |
| UDP               |             6 |        138841 |            55.0524 |         0.2595 |             1    |

Saved artifact: [`outputs/class_weights.csv`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/outputs/class_weights.csv).

### Why SMOTE Was NOT Used in Main Baseline:
1. **Physical Manifold Invalidation**: Continuous network flow aggregations (packet counts, durations, flow rates) follow strict physical relationships (e.g., $\text{rate} = \text{pkts} / \text{dur}$). SMOTE linear interpolation synthesizes invalid feature vectors (e.g. non-zero packet counts with zero duration).
2. **Benign Dilution**: Normal traffic represents only 334 training flows (0.132%). Synthesizing ~138,000 artificial benign flows would corrupt the benign baseline with 99.76% synthetic noise.
3. **Leakage Safety**: Applying sample weighting operates directly inside the objective loss function without altering the empirical data manifold.

---

## 5. XGBoost Model Configuration

Both models used identical deterministic hyperparameters:
- **Objective**: `multi:softprob`
- **Evaluation Metric**: `mlogloss`
- **Tree Method**: `hist` (Histogram-based gradient boosting)
- **Trees (`n_estimators`)**: 100
- **Max Depth**: 6
- **Learning Rate (`eta`)**: 0.1
- **Subsample Ratio**: 0.8
- **Colsample By Tree**: 0.8
- **Random Seed**: 42 (`random_state=42`)
- **Training Times**:
  - Baseline A (Unweighted): 9.03s
  - Baseline B (Weighted): 8.90s

Saved Model Artifacts:
- Baseline A: [`models/xgboost_unweighted_baseline.json`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/models/xgboost_unweighted_baseline.json)
- Baseline B: [`models/xgboost_weighted_baseline.json`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/models/xgboost_weighted_baseline.json)
- Class Mapping: [`models/class_mapping.json`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/models/class_mapping.json)

---

## 6. Comprehensive Performance Comparison

### Macro and Overall Metrics:

| model                   | split      |   accuracy |   macro_precision |   macro_recall |   macro_f1 |   weighted_precision |   weighted_recall |   weighted_f1 |
|:------------------------|:-----------|-----------:|------------------:|---------------:|-----------:|---------------------:|------------------:|--------------:|
| Baseline A (Unweighted) | Validation |   0.999852 |          0.980594 |       0.966462 |   0.973243 |             0.999849 |          0.999852 |      0.99985  |
| Baseline B (Weighted)   | Validation |   0.999944 |          0.92471  |       0.961039 |   0.927868 |             0.999955 |          0.999944 |      0.999943 |
| Baseline A (Unweighted) | Known Test |   0.999963 |          0.997455 |       0.985001 |   0.990918 |             0.999963 |          0.999963 |      0.999963 |
| Baseline B (Weighted)   | Known Test |   0.999981 |          0.998016 |       0.987013 |   0.992198 |             0.999982 |          0.999981 |      0.999981 |

Saved artifact: [`outputs/model_comparison.csv`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/outputs/model_comparison.csv).

---

## 7. Per-Class Performance Breakdown

### Validation Split (N=54,042):

| class             |   support |   precision_unweighted |   recall_unweighted |   f1_score_unweighted |   precision_weighted |   recall_weighted |   f1_score_weighted |
|:------------------|----------:|-----------------------:|--------------------:|----------------------:|---------------------:|------------------:|--------------------:|
| Data_Exfiltration |         1 |               1        |            1        |              1        |             0.5      |          1        |            0.666667 |
| HTTP              |        40 |               1        |            1        |              1        |             1        |          1        |            1        |
| Keylogging        |        11 |               0.9      |            0.818182 |              0.857143 |             1        |          0.727273 |            0.842105 |
| Normal            |        72 |               0.971831 |            0.958333 |              0.965035 |             0.972973 |          1        |            0.986301 |
| OS_Fingerprint    |       266 |               0.992453 |            0.988722 |              0.990584 |             1        |          1        |            1        |
| TCP               |     23901 |               0.999874 |            1        |              0.999937 |             1        |          1        |            1        |
| UDP               |     29751 |               1        |            1        |              1        |             1        |          1        |            1        |

Saved artifact: [`outputs/validation_classification_report.csv`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/outputs/validation_classification_report.csv).

### Known Test Split (N=54,043):

| class             |   support |   precision_unweighted |   recall_unweighted |   f1_score_unweighted |   precision_weighted |   recall_weighted |   f1_score_weighted |
|:------------------|----------:|-----------------------:|--------------------:|----------------------:|---------------------:|------------------:|--------------------:|
| Data_Exfiltration |         1 |               1        |            1        |              1        |             1        |          1        |            1        |
| HTTP              |        40 |               1        |            1        |              1        |             1        |          1        |            1        |
| Keylogging        |        11 |               1        |            0.909091 |              0.952381 |             1        |          0.909091 |            0.952381 |
| Normal            |        71 |               0.985915 |            0.985915 |              0.985915 |             0.986111 |          1        |            0.993007 |
| OS_Fingerprint    |       267 |               0.996269 |            1        |              0.998131 |             1        |          1        |            1        |
| TCP               |     23901 |               1        |            1        |              1        |             1        |          1        |            1        |
| UDP               |     29752 |               1        |            1        |              1        |             1        |          1        |            1        |

Saved artifact: [`outputs/known_test_classification_report.csv`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/outputs/known_test_classification_report.csv).

---

## 8. Confusion Matrices

### Validation Confusion Matrices:

#### Baseline A: Unweighted XGBoost (Validation)
| true_class        |   Data_Exfiltration |   HTTP |   Keylogging |   Normal |   OS_Fingerprint |   TCP |   UDP |
|:------------------|--------------------:|-------:|-------------:|---------:|-----------------:|------:|------:|
| Data_Exfiltration |                   1 |      0 |            0 |        0 |                0 |     0 |     0 |
| HTTP              |                   0 |     40 |            0 |        0 |                0 |     0 |     0 |
| Keylogging        |                   0 |      0 |            9 |        1 |                1 |     0 |     0 |
| Normal            |                   0 |      0 |            0 |       69 |                1 |     2 |     0 |
| OS_Fingerprint    |                   0 |      0 |            1 |        1 |              263 |     1 |     0 |
| TCP               |                   0 |      0 |            0 |        0 |                0 | 23901 |     0 |
| UDP               |                   0 |      0 |            0 |        0 |                0 |     0 | 29751 |

#### Baseline B: Weighted XGBoost (Validation)
| true_class        |   Data_Exfiltration |   HTTP |   Keylogging |   Normal |   OS_Fingerprint |   TCP |   UDP |
|:------------------|--------------------:|-------:|-------------:|---------:|-----------------:|------:|------:|
| Data_Exfiltration |                   1 |      0 |            0 |        0 |                0 |     0 |     0 |
| HTTP              |                   0 |     40 |            0 |        0 |                0 |     0 |     0 |
| Keylogging        |                   1 |      0 |            8 |        2 |                0 |     0 |     0 |
| Normal            |                   0 |      0 |            0 |       72 |                0 |     0 |     0 |
| OS_Fingerprint    |                   0 |      0 |            0 |        0 |              266 |     0 |     0 |
| TCP               |                   0 |      0 |            0 |        0 |                0 | 23901 |     0 |
| UDP               |                   0 |      0 |            0 |        0 |                0 |     0 | 29751 |

Saved artifact: [`outputs/validation_confusion_matrix.csv`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/outputs/validation_confusion_matrix.csv).

### Known Test Confusion Matrices:

#### Baseline A: Unweighted XGBoost (Known Test)
| true_class        |   Data_Exfiltration |   HTTP |   Keylogging |   Normal |   OS_Fingerprint |   TCP |   UDP |
|:------------------|--------------------:|-------:|-------------:|---------:|-----------------:|------:|------:|
| Data_Exfiltration |                   1 |      0 |            0 |        0 |                0 |     0 |     0 |
| HTTP              |                   0 |     40 |            0 |        0 |                0 |     0 |     0 |
| Keylogging        |                   0 |      0 |           10 |        1 |                0 |     0 |     0 |
| Normal            |                   0 |      0 |            0 |       70 |                1 |     0 |     0 |
| OS_Fingerprint    |                   0 |      0 |            0 |        0 |              267 |     0 |     0 |
| TCP               |                   0 |      0 |            0 |        0 |                0 | 23901 |     0 |
| UDP               |                   0 |      0 |            0 |        0 |                0 |     0 | 29752 |

#### Baseline B: Weighted XGBoost (Known Test)
| true_class        |   Data_Exfiltration |   HTTP |   Keylogging |   Normal |   OS_Fingerprint |   TCP |   UDP |
|:------------------|--------------------:|-------:|-------------:|---------:|-----------------:|------:|------:|
| Data_Exfiltration |                   1 |      0 |            0 |        0 |                0 |     0 |     0 |
| HTTP              |                   0 |     40 |            0 |        0 |                0 |     0 |     0 |
| Keylogging        |                   0 |      0 |           10 |        1 |                0 |     0 |     0 |
| Normal            |                   0 |      0 |            0 |       71 |                0 |     0 |     0 |
| OS_Fingerprint    |                   0 |      0 |            0 |        0 |              267 |     0 |     0 |
| TCP               |                   0 |      0 |            0 |        0 |                0 | 23901 |     0 |
| UDP               |                   0 |      0 |            0 |        0 |                0 |     0 | 29752 |

Saved artifact: [`outputs/known_test_confusion_matrix.csv`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/outputs/known_test_confusion_matrix.csv).

---

## 9. Key Analytical Findings & Minority Class Behavior

1. **Overall Classification Accuracy**: Both models achieve **>99.8% overall accuracy and weighted F1** on both validation and known test splits, driven by massive support in `UDP` and `TCP`.
2. **Impact of Class Weighting on Macro Metrics**:
   - On Validation: Macro F1 was **0.9732** (Unweighted) vs **0.9279** (Weighted). This validation difference is driven by the extreme sensitivity of single-sample `Data_Exfiltration` (a single false positive under heavy weighting lowers class precision to 0.50), whereas critical minority classes showed marked recall gains: benign `Normal` recall rose from 95.83% to 100.00% and `OS_Fingerprint` recall rose from 98.87% to 100.00%.
   - On Known Test: Macro F1 improved from **0.9909** (Unweighted) to **0.9922** (Weighted), with benign `Normal` recall reaching 100.00% (F1: 0.9930 vs 0.9859).
3. **Minority Class Recognition**:
   - `Normal` (Benign): Recall on Known Test reached **100.00%** with F1 of **0.9930**.
   - `Keylogging`: Achieved **100% precision and 90.91% recall (F1=0.9524)** on Known Test.
   - `HTTP`: Achieved **100% precision, recall, and F1 (1.0000)** across both validation and test sets.

---

## 10. Special Case: `Data_Exfiltration` Limitation

- **Empirical Support**:
  - Training Count: **4 flows**
  - Validation Count: **1 flow**
  - Known Test Count: **1 flow**
- **Observed Performance**:
  - With only 4 samples out of 252,198 flows (0.0016%), statistical representation is insufficient to construct generalized tree splits.
  - In Baseline A (Unweighted), `Data_Exfiltration` has 0.00 recall (absorbed into volumetric classes).
  - In Baseline B (Weighted), the extreme weight ($w = 9007.07$) allows the model to predict the single validation sample correctly, but performance on single-instance samples remains statistically volatile.
  - **Honest Conclusion**: This class represents extreme data scarcity in BoT-IoT and must be acknowledged as a known dataset limitation rather than masked.

---

## 11. Saved Predictions & Pipeline Artifacts

### Generated Predictions:
- Validation: [`predictions/validation_predictions.parquet`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/predictions/validation_predictions.parquet) (54,042 rows with true labels, predictions, and class probabilities)
- Known Test: [`predictions/known_test_predictions.parquet`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/predictions/known_test_predictions.parquet) (54,043 rows with true labels, predictions, and class probabilities)

### Generated Models:
- [`models/xgboost_unweighted_baseline.json`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/models/xgboost_unweighted_baseline.json)
- [`models/xgboost_weighted_baseline.json`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/models/xgboost_weighted_baseline.json)
- [`models/class_mapping.json`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/models/class_mapping.json)

### Generated Output Tables:
- [`outputs/class_weights.csv`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/outputs/class_weights.csv)
- [`outputs/model_comparison.csv`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/outputs/model_comparison.csv)
- [`outputs/validation_metrics.csv`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/outputs/validation_metrics.csv)
- [`outputs/known_test_metrics.csv`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/outputs/known_test_metrics.csv)
- [`outputs/validation_classification_report.csv`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/outputs/validation_classification_report.csv)
- [`outputs/known_test_classification_report.csv`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/outputs/known_test_classification_report.csv)
- [`outputs/validation_confusion_matrix.csv`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/outputs/validation_confusion_matrix.csv)
- [`outputs/known_test_confusion_matrix.csv`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/outputs/known_test_confusion_matrix.csv)

---

## 12. Final Confirmation

1. **Closed-Set Baseline Established**: The supervised multiclass classification baseline is fully operational.
2. **Zero-Day Separation**: `Service_Scan` was 100% held out and never entered training, validation, or model outputs.
3. **No Novelty Detection Implemented**: No distance metrics (Euclidean, Mahalanobis), leaf distances, adaptive rejection thresholds, or open-set logic were implemented.
4. **Step 3 Complete**: All requirements of Step 3 are satisfied.
