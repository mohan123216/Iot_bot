# Dataset Discovery and Audit Report: BoT-IoT Network Traffic Analysis

---

## 1. Executive Summary

This report documents the rigorous discovery, data quality assessment, and statistical audit of the raw network flow datasets present in the repository.

### Key Audit Findings:
- **Total Flow Records**: **3,668,522 network flows** across 4 raw extraction files totaling **970.01 MB (0.947 GB)**.
- **Extreme Class Imbalance**: The dataset displays an extreme imbalance ratio of **24,387.7 : 1** between the majority attack category (`DDoS`) and the minority category (`Theft`).
- **Benign Traffic Scarcity**: Legitimate benign network traffic (`Normal`) accounts for **only 477 flows (0.013%)** of the entire 3.66M flow corpus, with attack traffic comprising **99.987%** of records.
- **Missing Value Profile**: **0 missing values (0 nulls)** across all columns. However, non-numeric hexadecimal port encodings (e.g. `0x0303`, representing ICMP Type 3 / Code 3) were detected in `sport` and `dport`.
- **Duplicate Analysis**: **0 duplicate records** across full feature vectors due to sub-second microsecond timestamps and dynamic packet rate calculations.
- **Leakage Vulnerabilities Identified**: Raw IP addresses (`saddr`, `daddr`), temporal timestamps (`stime`, `ltime`), and extraction sequence identifiers (`pkSeqID`, `seq`) represent severe testbed environment leakage and must be strictly excluded from behavioral detection models.

---

## 2. Dataset Files and Physical Metadata

| File Name | File Size (MB) | Row Count | Column Count | Storage Path |
| :--- | :---: | :---: | :---: | :--- |
| `UNSW_2018_IoT_Botnet_Full5pc_1.csv` | 259.72 MB | 1,000,000 | 46 | `D:\temporary\project-word1-iot-bot-ds\UNSW_2018_IoT_Botnet_Full5pc_1.csv` |
| `UNSW_2018_IoT_Botnet_Full5pc_2.csv` | 260.43 MB | 1,000,000 | 46 | `D:\temporary\project-word1-iot-bot-ds\UNSW_2018_IoT_Botnet_Full5pc_2.csv` |
| `UNSW_2018_IoT_Botnet_Full5pc_3.csv` | 269.06 MB | 1,000,000 | 46 | `D:\temporary\project-word1-iot-bot-ds\UNSW_2018_IoT_Botnet_Full5pc_3.csv` |
| `UNSW_2018_IoT_Botnet_Full5pc_4.csv` | 180.8 MB | 668,522 | 46 | `D:\temporary\project-word1-iot-bot-ds\UNSW_2018_IoT_Botnet_Full5pc_4.csv` |

**Total Disk Footprint**: 970.01 MB  
**Total Records**: 3,668,522  
**Data Format**: Comma-Separated Values (CSV), RFC 4180 standard.

---

## 3. Class Distribution & Imbalance Analysis

### A. Binary Attack vs Benign Level (`attack`)

| class_name   |   sample_count |   percentage |   imbalance_ratio_vs_minority |
|:-------------|---------------:|-------------:|------------------------------:|
| Attack (1)   |        3668045 |       99.987 |                       7689.82 |
| Normal (0)   |            477 |        0.013 |                       7689.82 |

### B. High-Level Attack Category Level (`category`)

| class_name     |   sample_count |   percentage |   imbalance_ratio_vs_minority |
|:---------------|---------------:|-------------:|------------------------------:|
| DDoS           |        1926624 |      52.5177 |                       24387.7 |
| DoS            |        1650260 |      44.9843 |                       24387.7 |
| Reconnaissance |          91082 |       2.4828 |                       24387.7 |
| Normal         |            477 |       0.013  |                       24387.7 |
| Theft          |             79 |       0.0022 |                       24387.7 |

- **Majority Class**: `DDoS` (1,926,624 flows, 52.52%)
- **Minority Class**: `Theft` (79 flows, 0.002%)
- **Category Imbalance Ratio**: **24,387.7 : 1**
- **DDoS to Normal Imbalance Ratio**: **4,039.0 : 1**

### C. Fine-Grained Subcategory Level (`subcategory`)

| class_name        |   sample_count |   percentage |   imbalance_ratio_vs_minority |
|:------------------|---------------:|-------------:|------------------------------:|
| UDP               |        1981230 |      54.0062 |                        330205 |
| TCP               |        1593180 |      43.4284 |                        330205 |
| Service_Scan      |          73168 |       1.9945 |                        330205 |
| OS_Fingerprint    |          17914 |       0.4883 |                        330205 |
| HTTP              |           2474 |       0.0674 |                        330205 |
| Normal            |            477 |       0.013  |                        330205 |
| Keylogging        |             73 |       0.002  |                        330205 |
| Data_Exfiltration |              6 |       0.0002 |                        330205 |

---

## 4. Missing Values and Format Anomalies

Detailed inspection across all 3,668,522 rows and 46 columns revealed:

| column_name   |   missing_null_count |   missing_null_pct |   pos_infinity_count |   neg_infinity_count |   hex_string_in_numeric_count |   total_problematic_count |   total_problematic_pct | status                      |
|:--------------|---------------------:|-------------------:|---------------------:|---------------------:|------------------------------:|--------------------------:|------------------------:|:----------------------------|
| sport         |                    0 |                  0 |                    0 |                    0 |                          9052 |                      9052 |                  0.2467 | CONTAINS_FORMATTING_ANOMALY |
| dport         |                    0 |                  0 |                    0 |                    0 |                          9052 |                      9052 |                  0.2467 | CONTAINS_FORMATTING_ANOMALY |

### Port Field Hexadecimal Anomaly:
- While standard TCP and UDP flows record numeric port values (e.g., 80, 443, 53), ICMP packets do not possess layer-4 port numbers.
- In the BoT-IoT packet aggregation process, the Argus flow collector recorded ICMP control codes into the `sport` and `dport` fields using hexadecimal strings (such as `0x0303` for Destination Unreachable and `0x5000` for Echo Request).
- **Required Safe Handling**: These values must be mapped to valid integer codes or properly treated as protocol-specific indicators rather than crashing numeric parsers.

---

## 5. Duplicate Record Analysis

| duplicate_scope                                               |   duplicate_count |   duplicate_percentage | explanation                                                                                                                               |
|:--------------------------------------------------------------|------------------:|-----------------------:|:------------------------------------------------------------------------------------------------------------------------------------------|
| Full Row Duplicates (including pkSeqID)                       |                 0 |                      0 | pkSeqID is a strictly unique autoincrement primary key across all 3,668,522 rows.                                                         |
| Flow Records excluding pkSeqID                                |                 0 |                      0 | High-precision floating point epoch timestamps (stime, ltime) ensure every flow tuple is uniquely timestamped.                            |
| Behavioral Flow Tuples (excluding pkSeqID, stime, ltime, seq) |                 0 |                      0 | Continuous packet arrival rates, sliding-window statistical aggregations, and durations prevent identical duplicate flow feature vectors. |

Network flow duplicates can artificially inflate accuracy metrics if identical flows appear in both training and evaluation splits. In BoT-IoT:
- High-resolution flow start times (`stime`) and finish times (`ltime`) guarantee full row uniqueness.
- When ignoring sequence counters and timestamps, sliding-window statistical features (`AR_P_Proto_P_SrcIP`, `TnBPSrcIP`, `rate`) maintain genuine continuous variance, yielding **0 exact duplicate feature vectors**.

---

## 6. Feature Taxonomy & Potential Leakage Assessment

The 46 columns in the BoT-IoT dataset were classified into five distinct functional categories:

### A. Potentially Dangerous / Leakage Features (MUST NOT be used as predictive features)
1. **Source & Destination IP Addresses (`saddr`, `daddr`)**:
   - *Rationale*: In the BoT-IoT experimental testbed, attack traffic originated from a fixed set of compromised IP addresses (`192.168.100.147-150`), while victim servers were located at `192.168.100.3`. A classifier given raw IP addresses will trivially memorize the IP subnet rather than learning generalized malicious flow dynamics.
2. **Timestamps (`stime`, `ltime`)**:
   - *Rationale*: Attack campaigns were launched in distinct temporal windows (e.g. DoS on June 4, DDoS on June 5, Reconnaissance on June 9). Raw timestamps allow models to overfit to the testbed timeline.
3. **Primary Key / Sequence IDs (`pkSeqID`, `seq`)**:
   - *Rationale*: Arbitrary row indexes generated by dataset extractors.
4. **Target Labels (`attack`, `category`, `subcategory`)**:
   - *Rationale*: Ground-truth targets.

### B. Behavioral Features (Legitimate for Model Training)
- **Flow Characteristics**: `dur`, `pkts`, `bytes`, `spkts`, `dpkts`, `sbytes`, `dbytes`, `rate`, `srate`, `drate`
- **Statistical Aggregations**: `mean`, `stddev`, `sum`, `min`, `max`
- **State & Protocol Codes**: `flgs_number`, `proto_number`, `state_number`
- **Sliding-Window Behavioral Metrics**: `TnBPSrcIP`, `TnBPDstIP`, `TnP_PSrcIP`, `TnP_PDstIP`, `TnP_PerProto`, `TnP_Per_Dport`, `AR_P_Proto_P_SrcIP`, `AR_P_Proto_P_DstIP`, `N_IN_Conn_P_DstIP`, `N_IN_Conn_P_SrcIP`, `AR_P_Proto_P_Sport`, `AR_P_Proto_P_Dport`, `Pkts_P_State_P_Protocol_P_DestIP`, `Pkts_P_State_P_Protocol_P_SrcIP`

---

## 7. Constant & Near-Constant Features

Inspection across all columns confirmed:
- **Strictly Constant Features**: None across the global dataset (every column exhibits variance across the 4 files).
- **Partition-Specific Near-Constants**: In File 1, `attack` is 100% constant (`1`), and `category` is 100% constant (`DoS`). This demonstrates that raw files are partition-clustered and must be shuffled/stratified during train/validation splitting.

---

## 8. Data Cleaning Performed (Safe & Non-Destructive)

To prepare the dataset for subsequent modeling without introducing data leakage, the following safe operations were established in `preprocessing/clean_dataset.py`:
1. **Label Whitespace Normalization**: Stripped leading/trailing whitespace from `category`, `subcategory`, `proto`, and `state`.
2. **Standardized Categorical Casing**: Ensured consistent title casing across all labels.
3. **Hex Port Resolution**: Safely parsed hexadecimal ICMP port notations (`0x0303` -> integer `771`) to prevent numeric conversion exceptions.
4. **Preservation of Raw Files**: Original files (`UNSW_2018_IoT_Botnet_Full5pc_*.csv`) were kept 100% untouched. Cleaned data is saved strictly in `data/cleaned/`.

---

## 9. Cleaning NOT Performed (Intentionally Postponed)

In strict accordance with scientific integrity rules:
- **NO SMOTE or Synthetic Oversampling**: Applying SMOTE on unpartitioned data causes severe data leakage between synthetic neighbors and future test sets.
- **NO Undersampling**: Premature downsampling discards critical minority tail behaviors before establishing evaluation protocols.
- **NO Train/Test Splitting**: Stratified Leave-One-Category-Out (LOCO) partitioning is reserved for Step 2.
- **NO Feature Normalization / Scaling**: Scalers (e.g. RobustScaler, StandardScaler) must be fitted strictly on training data only.

---

## 10. Technical Recommendation for Handling Class Imbalance

Based strictly on the observed distribution (477 Normal vs 1,926,624 DDoS):

### Why Naive SMOTE is NOT Recommended:
1. **Continuous Manifold Distortion**: Network flow features (packet counts, durations, flow rates) follow heavy-tailed, non-Gaussian, multi-modal distributions. Standard SMOTE interpolates linearly between nearest neighbors in Euclidean space, generating unrealistic synthetic flows that do not correspond to legitimate network protocol behavior.
2. **Extreme Minority Expansion**: Expanding Normal traffic from 477 flows to ~1.9 million via SMOTE would create over 1.89 million artificial synthetic flows, overwhelming genuine benign behavioral characteristics.

### Recommended Strategy for Step 2:
1. **Cost-Sensitive Class Weighting**: Apply inverse-frequency class weights during loss computation:
   $$w_c = \frac{N}{C \cdot N_c}$$
2. **Stratified Mini-Batch Sampling**: Ensure every training batch contains representative samples of minority benign traffic.
3. **Dedicated Benign Protection Boundaries**: In open-set novelty detection, treat Normal traffic as a protected class with dedicated acceptance thresholds calibrated exclusively on empirical calibration distributions.

---

## 11. Generated Artifacts

The following reproducible outputs were generated in `outputs/`:
- `outputs/dataset_summary.csv`
- `outputs/class_distribution.csv`
- `outputs/missing_value_report.csv`
- `outputs/duplicate_report.csv`
- `outputs/feature_type_report.csv`
- `outputs/constant_features.csv`
- `reports/dataset_audit_report.md`
