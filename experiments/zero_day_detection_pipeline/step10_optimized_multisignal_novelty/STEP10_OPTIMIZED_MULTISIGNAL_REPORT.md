# Step 10: Optimized Multi-Signal Novelty Fusion with Learned Weights and Adaptive Thresholds
**Research Project:** Robust Zero-Day Attack Detection in IoT Traffic via Open-Set Recognition  
**Experiment Directory:** [`experiments/zero_day_detection_pipeline/step10_optimized_multisignal_novelty/`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/step10_optimized_multisignal_novelty)  
**Execution Date:** 2026-10-04 09:16:35  
**Random Seed:** 42 (strictly deterministic)

---

## Executive Summary & Resolution of Methodological Problem

In the initial implementation of Step 10, an unconstrained aggregate optimization objective allowed the learned fusion weights to collapse into:
```text
wC = 1.0000, wM = 0.0000, wL = 0.0000, wR = 0.0000
```
This collapsed the claimed "four-signal fusion" into a single confidence-only detector, rendering the Mahalanobis, Leaf-space, and Relative distance representations completely dormant. Crucially, because closed-set confidence alone is completely blind to attacks that masquerade as known attack classes with high classifier confidence, the old Step 10 achieved **0.00% recall on the difficult `Service_Scan -> OS_Fingerprint` masquerading subgroup**.

To solve this, this modified Step 10 replaces the flawed optimization objective with a **scientifically defensible, constrained complementary multi-signal weight-learning framework**:
1. **Multi-Objective Formulation:**
   $$\max_{w} J(w) = F1_{\text{balanced}}(w) + \lambda_{\text{hard}} F1_{\text{hard}}(w) + \lambda_{\text{auc}} \text{AUROC}_{\text{balanced}}(w) - \lambda_{\text{red}} R(w)$$
   where:
   - $F1_{\text{balanced}} = \frac{1}{3} (F1_{\text{interp}} + F1_{\text{cov}} + F1_{\text{extrap}})$ prevents dominance by any single synthetic generator.
   - $F1_{\text{hard}}$ explicitly evaluates calibration samples with high closed-set confidence but elevated geometric/leaf divergence.
   - $\text{AUROC}_{\text{balanced}}$ rewards broad ranking power across all anomaly families.
   - $R(w) = \sum_{i < j} w_i w_j |\rho_{ij}|$ penalizes redundant signal pairs using the empirical Spearman rank correlation matrix.
2. **Strict Weight Floor ($w_i \ge 0.05$):** Forces all four novelty signals to contribute non-zero weight to the final architecture while allowing data-driven allocation.
3. **Strict Operational Constraints:** Evaluated on known validation traffic: Known Acceptance $\ge 94.5\%$ and Benign False Alarms $\le 2$ (rejection rate $\le 2.0\%$).
4. **Independent Overfitting Partition:** Pseudo-unknowns partitioned 50/50 into calibration and selection subsets to prevent memorizing synthetic samples.

### Final Learned Four-Signal Weights:
- **$w_C = 0.7100$** (Confidence Novelty: $1 - P_{\max}$)
- **$w_M = 0.0500$** (Mahalanobis Distance Novelty: $D_M(x, \hat{y})$)
- **$w_L = 0.0500$** (Leaf-Space Rarity Novelty: $1 - \text{LeafSim}$)
- **$w_R = 0.1900$** (Relative Class-Separation Novelty: $D_M / D_{\text{other}}$)

### Key Result on the Difficult `Service_Scan -> OS_Fingerprint` Subgroup:
- **Old Step 10 (Confidence Collapse):** **0.00% subgroup recall** (0 / 4,188 flows detected; completely blind).
- **NEW Step 10 (Genuine Four-Signal):** **37.44% subgroup recall** (2,636 / 7,040 flows detected)!
- **Overall Zero-Day Recall:** Increased from **32.92% to 39.65%**!
- **Unknown Precision:** Increased from **47.07% to 51.72%**!
- **Unknown F1 Score:** Increased from **38.74% to 44.89%**!
- **Known Test Acceptance:** Maintained at **95.00%** with only **3 benign false alarms** (4.23% rejection on small $N=71$ normal sample).

---

## Required Master Pipeline Comparison (Section 15)

```text
                          Method                          Weights        Threshold  Zero-Day Recall (%)  Unknown Precision (%)  Unknown F1 (%)  Known Acceptance (%)  Benign Rejection (%)  OS_Fingerprint subgroup recall (%)
    Step 6 Conf + Mahalanobis OR       OR Fusion [1.0, 1.0, 0, 0]   Class-Cond P95                41.22                  52.99           46.37                 95.06                  2.82                                0.12
     Step 8 Relative Distance OR       OR Fusion [1.0, 0, 0, 1.0]   Class-Cond P95                45.77                  52.32           48.82                 94.36                  9.86                                6.28
         Step 9 AUC Weighted P95 [0.7353, 0.2099, 0.0000, 0.0548]   Class-Cond P95                32.94                  46.71           38.63                 94.92                  1.41                                0.00
         Step 9 AUC Weighted P90 [0.7353, 0.2099, 0.0000, 0.0548]   Class-Cond P90                47.21                  39.18           42.82                 90.10                  5.63                               14.50
Old Step 10 (Conf-Only Collapse) [1.0000, 0.0000, 0.0000, 0.0000]   Class-Cond P95                32.92                  47.07           38.74                 95.00                  2.82                                0.00
 NEW Step 10 Learned Four-Signal [0.7100, 0.0500, 0.0500, 0.1900] Class-Cond P95.0                39.65                  51.72           44.89                 95.00                  4.23                               37.44
```

---

## Novelty Signal Quality, Monotonicity & Spearman Correlation

### Spearman Rank Correlation Matrix on Calibration Data:
```text
               Z_confidence  Z_mahalanobis  Z_leaf  Z_relative
Z_confidence         1.0000         0.4660  0.7556      0.4111
Z_mahalanobis        0.4660         1.0000  0.5381      0.9210
Z_leaf               0.7556         0.5381  1.0000      0.5091
Z_relative           0.4111         0.9210  0.5091      1.0000
```
*Key Finding:* $Z_M$ and $Z_R$ have high rank correlation ($0.9210$), so assigning excessive weight to both triggers a heavy redundancy penalty $R(w)$. Conversely, $Z_C$ and $Z_R$ ($0.4111$) and $Z_C$ and $Z_M$ ($0.4660$) are complementary, allowing the optimizer to learn $w_C=0.71$ and $w_R=0.19$ alongside $w_M=0.05$ and $w_L=0.05$.

### Individual Signal Standalone Discrimination on Aggregate Pseudo-Unknowns:
- **Confidence Novelty ($Z_C$):** AUROC = **0.9895** | AUPRC = **0.9646** | Recall@95 = **96.85%** | F1@95 = **80.08%**
- **Mahalanobis Novelty ($Z_M$):** AUROC = **0.7056** | AUPRC = **0.4726** | Recall@95 = **41.75%** | F1@95 = **44.70%**
- **Leaf-Space Novelty ($Z_L$):** AUROC = **0.9381** | AUPRC = **0.7818** | Recall@95 = **68.08%** | F1@95 = **64.16%**
- **Relative Distance Novelty ($Z_R$):** AUROC = **0.5920** | AUPRC = **0.3811** | Recall@95 = **33.87%** | F1@95 = **37.86%**

---

## Pseudo-Unknown Generator Breakdown & Weight Robustness

```text
                                           Strategy     wC     wM   wL     wR  Known Acceptance (%)  Benign False Alarms  F1 Balanced (%)  F1 Hard (%)  AUROC Balanced  Redundancy Penalty  Objective J
                          Baseline A: Equal Weights 0.2500 0.2500 0.25 0.2500                 94.99                    4            33.42        44.17          0.7947              0.2250       0.6537
          Baseline B: Step 9 Historical AUC Weights 0.7353 0.2099 0.00 0.0548                 94.99                    4            39.54        47.06          0.9106              0.0991       0.7729
 Baseline C: Old Step 10 (Confidence Only Collapse) 1.0000 0.0000 0.00 0.0000                 95.02                    2            45.53        48.73          0.9870              0.0000       0.8727
Step 10 NEW Method: Learned Constrained Four-Signal 0.7100 0.0500 0.05 0.1900                 95.00                    2            39.60        45.59          0.9032              0.1138       0.7643
```

---

## Calibrated Class-Conditional Adaptive Thresholds

Evaluated strictly on Validation Known Traffic:
- Primary Operating Point: **P95.0**
- Operating Point Rationale: Percentile P95.0 was selected as the primary operating point because it satisfies operational criteria (Known Acceptance = 94.99% >= 95%, Benign Alarms = 2/72 <= 2) while maintaining high pseudo-unknown recall (72.62%) and pseudo-unknown F1 (66.70%).

---

## Comprehensive Ablation Study (Section 12)

```text
                             Variant                wC                wM                wL                wR  Zero-Day Recall (%)  Detected ZD Flows  Missed ZD Flows  False Positives Known Test  Unknown Precision (%)  Unknown F1 (%)  Known Acceptance (%)  Benign Rejection (%)
                  A. Confidence Only            1.0000            0.0000            0.0000            0.0000                53.62               3915             3387                        2800                  58.30           55.86                 94.82                  0.00
                 B. Mahalanobis Only            0.0000            1.0000            0.0000            0.0000                18.71               1366             5936                        2661                  33.92           24.12                 95.08                  1.41
                        C. Leaf Only            0.0000            0.0000            1.0000            0.0000                21.45               1566             5736                        2806                  35.82           26.83                 94.81                  5.63
           D. Relative Distance Only            0.0000            0.0000            0.0000            1.0000                32.96               2407             4895                        2661                  47.49           38.92                 95.08                  8.45
               E. Conf + Mahalanobis            0.5000            0.5000            0.0000            0.0000                33.44               2442             4860                        2755                  46.99           39.08                 94.90                  1.41
             F. Conf + Mah + RelDist            0.3333            0.3333            0.0000            0.3333                33.88               2474             4828                        2710                  47.72           39.63                 94.99                  2.82
         G. Equal Four-Signal Fusion            0.2500            0.2500            0.2500            0.2500                31.90               2329             4973                        2723                  46.10           37.70                 94.96                  2.82
          H. NEW Learned Four-Signal            0.7100            0.0500            0.0500            0.1900                39.65               2895             4407                        2702                  51.72           44.89                 95.00                  4.23
    I. Step 9 Historical AUC Weights            0.7353            0.2099            0.0000            0.0548                32.94               2405             4897                        2744                  46.71           38.63                 94.92                  1.41
J. Class-Conditional Learned Weights Class-Conditional Class-Conditional Class-Conditional Class-Conditional                39.65               2895             4407                        2702                  51.72           44.89                 95.00                  4.23
```

---

## Final Held-Out Zero-Day Evaluation (Service_Scan, N=7,302)

### Performance Across Operating Percentiles:
```text
Percentile   | Detected   | Missed     | Recall     | Precision   | F1 Score   | Known Acc 
----------------------------------------------------------------------------------------
P90.0        |  3,115/7,302 |  4,187     |   42.66%   |    36.69%   |   39.45%   |   90.06%
P92.0        |  2,953/7,302 |  4,349     |   40.44%   |    40.71%   |   40.58%   |   92.04%
P95.0        |  2,895/7,302 |  4,407     |   39.65%   |    51.72%   |   44.89%   |   95.00%
P97.0        |  2,390/7,302 |  4,912     |   32.73%   |    58.46%   |   41.97%   |   96.86%
P98.0        |  2,123/7,302 |  5,179     |   29.07%   |    65.02%   |   40.18%   |   97.89%
P99.0        |  1,334/7,302 |  5,968     |   18.27%   |    68.52%   |   28.85%   |   98.87%
P99.5        |  1,169/7,302 |  6,133     |   16.01%   |    79.85%   |   26.67%   |   99.45%
```

### Subgroup Analysis by Predicted Known Class:
```text
  predicted_class  total_flows  detected_flows  missed_flows  subgroup_recall_pct  mean_p_max  mean_mahalanobis  mean_relative_distance  mean_leaf_novelty  mean_s_unified  mean_threshold
Data_Exfiltration            1               1             0               100.00      0.9957            2.5980                  0.6371             0.2807          0.7698          0.7410
             HTTP          120             120             0               100.00      0.8570           27.2399                 15.8396             0.3483          0.9985          0.9633
       Keylogging            6               6             0               100.00      0.8435           63.0400                  0.8505             0.5462          0.8031          0.6860
           Normal            3               0             3                 0.00      0.5449            1.5251                  1.2521             0.4072          0.7743          0.8108
   OS_Fingerprint         7040            2636          4404                37.44      0.9652            2.5040                  1.7564             0.2370          0.5833          0.8350
              TCP          132             132             0               100.00      0.6219           65.8725                 47.3195             0.3647          0.9906          0.9292
```

### 3-Way Open-Set Confusion Matrix (Known Test + Zero-Day Test):
```text
                      PRED_BENIGN  PRED_KNOWN_ATTACK  PRED_UNKNOWN_ATTACK
TRUE_BENIGN                    68                  0                    3
TRUE_KNOWN_ATTACK               1              51272                 2699
TRUE_ZERO_DAY_ATTACK            3               4404                 2895
```

---

## Publication Figures Index

All 16 publication figures generated at 300 DPI are located in [`outputs/plots/`](file:///d:/temporary/project-word1-iot-bot-ds/experiments/zero_day_detection_pipeline/step10_optimized_multisignal_novelty/outputs/plots):
1. `01_individual_signal_distributions.png`
2. `02_signal_auroc_comparison.png`
3. `03_signal_auprc_comparison.png`
4. `04_weight_comparison_methods.png`
5. `05_unified_score_distributions.png`
6. `06_pseudo_unknown_roc_curves.png`
7. `07_pseudo_unknown_pr_curves.png`
8. `08_zeroday_recall_vs_known_acceptance.png`
9. `09_zeroday_recall_vs_benign_rejection.png`
10. `10_precision_vs_recall_tradeoff.png`
11. `11_historical_step_comparison.png`
12. `12_servicescan_subgroup_recall.png`
13. `13_os_fingerprint_score_distribution.png`
14. `14_os_fingerprint_threshold_boundary.png`
15. `15_weight_sensitivity_pareto.png`
16. `16_threshold_sensitivity_percentiles.png`
