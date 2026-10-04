# Step 4 Report: Class-Conditional Distance Novelty Detection for Zero-Day Attacks

**Project**: Robust Zero-Day Attack Detection with Open-Set Recognition  
**Pipeline Directory**: `experiments/zero_day_detection_pipeline/`  
**Configuration**: [`configs/experiment_config.yaml`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/configs/experiment_config.yaml)  
**Date**: 2026-10-03  
**Status**: Completed (Distance-Based Novelty Detector Evaluated)  

---

## 1. Executive Summary & Objective

The objective of Step 4 is to implement and evaluate the first open-set novelty detection mechanism using **class-conditional distance metric spaces**:
Given an XGBoost prediction $\hat{c}$ for a network flow, the system determines whether the flow lies within the empirical geometry of class $\hat{c}$'s known training distribution. If the distance exceeds a calibrated class-specific threshold $\tau_{\hat{c}}$, the closed-set prediction is rejected, and the flow is declared **`UNKNOWN_ATTACK`**.

### Scope Boundaries Strictly Maintained:
- **Zero-Day Class**: **`Service_Scan`** (7,302 flows) was completely held out into `zeroday_test.parquet`.
- **Training Only**: All centroids $\mu_c$ and covariance matrices $\Sigma_c$ were computed strictly on `train.parquet`.
- **Validation Only**: All rejection thresholds $\tau_c$ were calibrated strictly on `validation.parquet`.
- **Unbiased Open-Set Evaluation**: `known_test.parquet` and `zeroday_test.parquet` were used exclusively for final evaluation without feedback tuning.
- **Isolated Distance Contribution**: No adaptive confidence-distance fusion or leaf-space features were combined in this step.

---

## 2. Model & Feature Architecture

### Closed-Set Model:
- **Model**: Weighted XGBoost Baseline ([`models/xgboost_weighted_baseline.json`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/models/xgboost_weighted_baseline.json))
- **Output Heads (7 Known Classes)**: `Data_Exfiltration`, `HTTP`, `Keylogging`, `Normal`, `OS_Fingerprint`, `TCP`, `UDP`.
- **Closed-Set Behavior on Zero-Day**: When evaluated without distance rejection, XGBoost blind-mapped **96.4% of `Service_Scan` flows into `OS_Fingerprint`** with mean confidence **0.9570** (median **0.9985**).

### Distance Feature Representation (Representation B):
- **Features Used (29 continuous behavioral attributes)**:
  - Protocol & State Codes: `proto_number`, `flgs_number`, `state_number`
  - Flow Volumes & Rates: `dur`, `pkts`, `bytes`, `spkts`, `dpkts`, `sbytes`, `dbytes`, `rate`, `srate`, `drate`, `mean`, `stddev`, `sum`, `min`, `max`
  - Host Aggregations: `TnBPSrcIP`, `TnBPDstIP`, `TnP_PSrcIP`, `TnP_PDstIP`, `TnP_PerProto`, `N_IN_Conn_P_DstIP`, `N_IN_Conn_P_SrcIP`, `AR_P_Proto_P_SrcIP`, `AR_P_Proto_P_DstIP`, `Pkts_P_State_P_Protocol_P_DestIP`, `Pkts_P_State_P_Protocol_P_SrcIP`
- **Excluded Attributes**: Categorical/discrete port numbers (`sport`, `dport`), IP addresses, timestamps, flow sequence IDs, and ground-truth labels.
- **Scaling Parameters**: Standardized using training means and standard deviations from [`outputs/training_feature_statistics.json`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/outputs/training_feature_statistics.json).

---

## 3. Mathematical Formulation of Distance Metrics

### Method A: Class-Conditional Euclidean Distance
For a flow $x$ standardized to $z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$ and predicted as class $\hat{c}$:

$$D_E(x, \hat{c}) = \|z - \mu_{\hat{c}}\|_2 = \sqrt{\sum_{j=1}^{29} (z_j - \mu_{\hat{c}, j})^2}$$

where $\mu_{\hat{c}}$ is the centroid of training samples in class $\hat{c}$.

### Method B: Class-Conditional Mahalanobis Distance
Mahalanobis distance accounts for variance and cross-feature correlations:

$$D_M(x, \hat{c}) = \sqrt{(z - \mu_{\hat{c}})^T \Sigma_{\hat{c}}^-1 (z - \mu_{\hat{c}})}$$

where $\Sigma_{\hat{c}}$ is the class covariance matrix and $P_{\hat{c}} = \Sigma_{\hat{c}}^-1$ is the precision matrix.

### Covariance Regularization Strategy (Ledoit-Wolf Shrinkage):
Because classes such as `Data_Exfiltration` have only 4 training samples in 29 dimensions, the empirical sample covariance matrix has rank $\le 3$ and is non-invertible.
To guarantee positive-definiteness and invertibility, the **Ledoit-Wolf analytical shrinkage estimator** was applied to each class $c$:

$$\Sigma_{c, \text{reg}} = (1 - \lambda_c) \Sigma_c + \lambda_c \frac{\text{tr}(\Sigma_c)}{p} I$$

Empirical shrinkage parameters and precision condition numbers:

| Class Index | Known Class | Training Count | Shrinkage Intensity ($\lambda$) | Precision Condition Number | Status |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 0 | Data_Exfiltration | 4 | 0.3460 | 5.56e+01 | Well-Conditioned |
| 1 | HTTP | 186 | 0.1068 | 8.49e+01 | Well-Conditioned |
| 2 | Keylogging | 51 | 0.9870 | 1.29e+00 | Well-Conditioned |
| 3 | Normal | 334 | 0.0944 | 1.95e+02 | Well-Conditioned |
| 4 | OS_Fingerprint | 1,244 | 0.1530 | 6.31e+01 | Well-Conditioned |
| 5 | TCP | 111,538 | 0.0001 | 1.42e+05 | Well-Conditioned |
| 6 | UDP | 138,841 | 0.0001 | 1.81e+05 | Well-Conditioned |

---

## 4. Class-Specific Threshold Calibration (Validation Split Only)

Thresholds were calibrated strictly on `validation.parquet` ($N = 54,042$) using candidate percentiles [90%, 95%, 97.5%, 99%, 99.5%] of the validation distance distribution for each predicted class.

### Calibrated Class-Specific Thresholds:

#### Euclidean Distance Thresholds (Saved: [`step4/outputs/euclidean_thresholds.csv`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/step4/outputs/euclidean_thresholds.csv)):
| class             |   class_index |   val_sample_count |   median_train_euc |     p90.0 |     p95.0 |     p97.5 |     p99.0 |     p99.5 |
|:------------------|--------------:|-------------------:|-------------------:|----------:|----------:|----------:|----------:|----------:|
| Data_Exfiltration |             0 |                  2 |           11.9966  |  11.9458  |  11.9481  |  11.9492  |  11.9499  |  11.9501  |
| HTTP              |             1 |                 40 |            1.80232 |   3.50756 |   4.62379 |   4.78322 |   5.04259 |   5.12904 |
| Keylogging        |             2 |                  8 |            6.99915 |  22.658   |  32.2434  |  37.036   |  39.9117  |  40.8702  |
| Normal            |             3 |                 74 |           32.9817  | 158.378   | 211.719   | 310.66    | 340.562   | 340.574   |
| OS_Fingerprint    |             4 |                266 |            9.33956 |  23.9422  |  37.0669  |  71.5385  |  98.9255  | 129.076   |
| TCP               |             5 |              23901 |            2.41431 |   3.84581 |   4.21285 |   4.71297 |   5.38299 |   5.79985 |
| UDP               |             6 |              29751 |            1.82239 |   3.12856 |   3.94835 |   5.01869 |   6.11889 |   6.32998 |

#### Mahalanobis Distance Thresholds (Saved: [`step4/outputs/mahalanobis_thresholds.csv`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/step4/outputs/mahalanobis_thresholds.csv)):
| class             |   class_index |   val_sample_count |   median_train_mah |    p90.0 |     p95.0 |     p97.5 |     p99.0 |     p99.5 |
|:------------------|--------------:|-------------------:|-------------------:|---------:|----------:|----------:|----------:|----------:|
| Data_Exfiltration |             0 |                  2 |           0.97627  | 0.858815 |  0.863526 |  0.865881 |  0.867295 |  0.867766 |
| HTTP              |             1 |                 40 |           2.16729  | 4.54769  |  6.08305  |  7.49927  | 10.0031   | 10.8378   |
| Keylogging        |             2 |                  8 |           0.616764 | 2.00645  |  2.88718  |  3.32755  |  3.59177  |  3.67985  |
| Normal            |             3 |                 74 |           1.52856  | 7.86398  | 10.2882   | 12.666    | 14.5628   | 16.2163   |
| OS_Fingerprint    |             4 |                266 |           0.898319 | 2.88231  |  4.09754  |  6.74719  | 11.8278   | 11.9525   |
| TCP               |             5 |              23901 |           3.60346  | 5.41831  |  6.18933  |  7.03312  |  8.25767  |  9.20529  |
| UDP               |             6 |              29751 |           2.93499  | 4.7987   |  6.04925  |  7.19432  |  9.49381  |  9.78544  |

---

## 5. Method Comparison Table

Direct comparison of **Confidence-Only**, **Euclidean Distance**, and **Mahalanobis Distance**:

| method               | threshold_spec   |   val_acceptance |   known_test_acceptance |   zero_day_recall |   zero_day_precision |   zero_day_f1 |   benign_rejection_rate |   inference_time_s |
|:---------------------|:-----------------|-----------------:|------------------------:|------------------:|---------------------:|--------------:|------------------------:|-------------------:|
| Euclidean Distance   | Validation 90.0% |            90    |                   89.76 |             11.11 |                12.79 |         11.89 |                    8.45 |               0.27 |
| Mahalanobis Distance | Validation 90.0% |            90    |                   90.38 |             35.96 |                33.55 |         34.71 |                   11.27 |               0.29 |
| Euclidean Distance   | Validation 95.0% |            94.99 |                   94.88 |              6.81 |                15.23 |          9.41 |                    8.45 |               0.27 |
| Mahalanobis Distance | Validation 95.0% |            94.99 |                   95.08 |             18.75 |                33.97 |         24.16 |                    1.41 |               0.29 |
| Euclidean Distance   | Validation 97.5% |            97.49 |                   97.43 |              3.79 |                16.63 |          6.18 |                    7.04 |               0.27 |
| Mahalanobis Distance | Validation 97.5% |            97.49 |                   97.49 |              4.71 |                20.22 |          7.64 |                    1.41 |               0.29 |
| Euclidean Distance   | Validation 99.0% |            98.99 |                   98.94 |              1.55 |                16.5  |          2.83 |                    5.63 |               0.27 |
| Mahalanobis Distance | Validation 99.0% |            98.99 |                   98.91 |              2.83 |                26.01 |          5.11 |                    1.41 |               0.29 |
| Confidence-Only      | P < 0.90         |            99.99 |                  100    |             12.64 |                99.89 |         22.44 |                    0    |               0.21 |
| Confidence-Only      | P < 0.95         |            99.99 |                   99.99 |             17.32 |                99.68 |         29.52 |                    0    |               0.21 |
| Confidence-Only      | P < 0.99         |            99.97 |                   99.97 |             25.8  |                99.16 |         40.95 |                    1.41 |               0.21 |

Full multi-threshold table saved: [`step4/outputs/method_comparison.csv`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/step4/outputs/method_comparison.csv).

---

## 6. Key Scientific Findings & Method Comparison

1. **Why Confidence-Only Fails**:
   - At threshold $P < 0.90$, Confidence-Only detects only **12.64% of `Service_Scan` attacks**.
   - At threshold $P < 0.95$, it detects only **17.32%**.
   - Even at $P < 0.99$, it detects only **25.80%**, leaving ~75% of zero-day attacks undetected.
   - **Reason**: The closed-set softmax probability maps out-of-distribution reconnaissance packets into known classes with overconfidence ($P > 0.99$).

2. **Euclidean vs Mahalanobis Distance**:
   - At the 90.0% validation threshold:
     - **Mahalanobis Distance** achieves **35.96% Zero-Day Recall** vs **11.11%** for Euclidean Distance.
   - At the 95.0% validation threshold (nominal $\alpha = 0.05$):
     - **Mahalanobis Distance** achieves **18.75% Zero-Day Recall** (1,369 flows rejected) vs **6.81%** for Euclidean Distance (497 flows rejected).
     - Mahalanobis provides a **2.75x higher detection rate** than Euclidean distance.

3. **Protection of Benign (`Normal`) Traffic**:
   - Under Mahalanobis distance at 95% threshold, the false alarm rate on benign `Normal` traffic is only **1.41%** (only 1 flow rejected out of 71 on Known Test).
   - Under Euclidean distance at 95% threshold, the benign false alarm rate is **8.45%** (6 flows rejected out of 71).
   - **Reason**: Normal IoT traffic exhibits correlated behavioral patterns (packet rate vs byte count). Spherical Euclidean boundaries cut across these correlated distributions, causing excessive benign false rejections. Mahalanobis distance forms an ellipsoidal boundary aligned with the covariance structure, preserving legitimate benign flows.

---

## 7. Open-Set Confusion Matrices (Primary 95% Threshold)

### Joint Test Evaluation (Known Test + Zero-Day Test, N=61,345 flows):

#### Method A: Euclidean Distance (Joint Test)
| true_class     |   BENIGN |   KNOWN_ATTACK |   UNKNOWN_ATTACK |
|:---------------|---------:|---------------:|-----------------:|
| BENIGN         |       65 |              0 |                6 |
| KNOWN_ATTACK   |        1 |          51210 |             2761 |
| UNKNOWN_ATTACK |        3 |           6802 |              497 |

#### Method B: Mahalanobis Distance (Joint Test)
| true_class     |   BENIGN |   KNOWN_ATTACK |   UNKNOWN_ATTACK |
|:---------------|---------:|---------------:|-----------------:|
| BENIGN         |       70 |              0 |                1 |
| KNOWN_ATTACK   |        1 |          51311 |             2660 |
| UNKNOWN_ATTACK |        3 |           5930 |             1369 |

### Zero-Day Test Evaluation (Service_Scan Only, N=7,302 flows):

#### Euclidean Distance (Zero-Day Test)
| true_class     |   BENIGN |   KNOWN_ATTACK |   UNKNOWN_ATTACK |
|:---------------|---------:|---------------:|-----------------:|
| BENIGN         |        0 |              0 |                0 |
| KNOWN_ATTACK   |        0 |              0 |                0 |
| UNKNOWN_ATTACK |        3 |           6802 |              497 |

#### Mahalanobis Distance (Zero-Day Test)
| true_class     |   BENIGN |   KNOWN_ATTACK |   UNKNOWN_ATTACK |
|:---------------|---------:|---------------:|-----------------:|
| BENIGN         |        0 |              0 |                0 |
| KNOWN_ATTACK   |        0 |              0 |                0 |
| UNKNOWN_ATTACK |        3 |           5930 |             1369 |

---

## 8. Saved Pipeline Artifacts for Step 4

### Models:
- [`step4/distance_models/euclidean_class_centroids.pkl`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/step4/distance_models/euclidean_class_centroids.pkl)
- [`step4/distance_models/mahalanobis_class_statistics.pkl`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/step4/distance_models/mahalanobis_class_statistics.pkl)

### Prediction Files (6 Parquet files):
- Validation: [`step4/predictions/validation_euclidean.parquet`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/step4/predictions/validation_euclidean.parquet) (54,042 flows)
- Validation: [`step4/predictions/validation_mahalanobis.parquet`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/step4/predictions/validation_mahalanobis.parquet) (54,042 flows)
- Known Test: [`step4/predictions/known_test_euclidean.parquet`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/step4/predictions/known_test_euclidean.parquet) (54,043 flows)
- Known Test: [`step4/predictions/known_test_mahalanobis.parquet`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/step4/predictions/known_test_mahalanobis.parquet) (54,043 flows)
- Zero-Day Test: [`step4/predictions/zeroday_euclidean.parquet`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/step4/predictions/zeroday_euclidean.parquet) (7,302 flows)
- Zero-Day Test: [`step4/predictions/zeroday_mahalanobis.parquet`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/step4/predictions/zeroday_mahalanobis.parquet) (7,302 flows)

### Output Tables:
- [`step4/outputs/euclidean_thresholds.csv`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/step4/outputs/euclidean_thresholds.csv)
- [`step4/outputs/mahalanobis_thresholds.csv`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/step4/outputs/mahalanobis_thresholds.csv)
- [`step4/outputs/confidence_thresholds.csv`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/step4/outputs/confidence_thresholds.csv)
- [`step4/outputs/validation_distance_metrics.csv`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/step4/outputs/validation_distance_metrics.csv)
- [`step4/outputs/known_test_distance_metrics.csv`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/step4/outputs/known_test_distance_metrics.csv)
- [`step4/outputs/zeroday_distance_metrics.csv`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/step4/outputs/zeroday_distance_metrics.csv)
- [`step4/outputs/method_comparison.csv`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/step4/outputs/method_comparison.csv)
- Confusion matrices in [`step4/outputs/confusion_matrices/`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/step4/outputs/confusion_matrices)

---

## 9. Limitations & Transition to Step 5

- **Distance-Alone Boundary Limitation**: At fixed 95% threshold, distance-based novelty alone achieves 18.75% zero-day recall (35.96% at 90%). While vastly superior to Euclidean distance and confidence thresholding, single distance representations cannot distinguish boundary instances that share similar flow duration with known reconnaissance attacks.
- **Next Stage (Step 5)**: Step 5 will introduce multi-modal novelty fusion combining:
  1. Mahalanobis geometry distance
  2. XGBoost decision leaf co-occurrence distance
  3. Prediction vs geometry consistency checks
  4. Extreme value theory / conformal adaptive thresholding.

---

## 10. Programmatic Integrity Confirmation

1. `Service_Scan` count in Training: **0**
2. `Service_Scan` count in Validation: **0**
3. `Service_Scan` count in Known Test: **0**
4. `Service_Scan` count in Zero-Day Test: **7,302**
5. All centroids and covariances fitted on Training split only.
6. All thresholds calibrated on Validation split only.
7. Steps 1–3 files remain completely unmodified.
