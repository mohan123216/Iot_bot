# Scientific Reproducibility & Retraining Verification Report

**Execution Timestamp:** 2026-10-10 18:02:01  
**Total Retraining Execution Time:** 18.68 seconds  
**Reproducibility Status:** 100% IDENTICAL DETERMINISTIC MATCH

---

## 1. Executive Summary

This independent retraining run was executed from scratch to rigorously verify that our proposed solution achieves **100% reproducible results**. All newly trained model weights, evaluation metrics, and comparison logs have been saved into this separate folder:
- **Models Directory:** `retraining_verification/models/`
- **Results Directory:** `retraining_verification/results/`

---

## 2. Tier-1 Balanced Multi-Class Classifier Verification

| Metric | Original Benchmark | Newly Retrained Run | Delta ($\Delta$) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Multi-Class Accuracy** | **98.97%** | **98.97%** | +0.0000% | **Exact Match** |
| **Macro Average F1-Score** | **98.26%** | **98.26%** | +0.0000% | **Exact Match** |
| **Normal Benign Specificity** | **99.82%** | **99.82%** | +0.00% | **Exact Match** |
| **Data Exfiltration F1-Score** | **93.62%** | **93.62%** | +0.00% | **Exact Match** |

---

## 3. Table 9 / Leave-One-Subclass-Out (LOCO) Zero-Day Verification

Direct comparison across all 10 individual held-out zero-day attack experiments:

| Class | Subclass | Original Precision | Retrained Precision | Original Recall | Retrained Recall | Original F1 | Retrained F1 | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DoS** | **HTTP** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **MATCH (100%)** |
| **DoS** | **TCP** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **MATCH (100%)** |
| **DoS** | **UDP** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **MATCH (100%)** |
| **DDoS** | **HTTP** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **MATCH (100%)** |
| **DDoS** | **TCP** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **MATCH (100%)** |
| **DDoS** | **UDP** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **MATCH (100%)** |
| **Reconnaissance** | **OS_Fingerprint** | 1.00 | 1.00 | 0.99 | 0.99 | 1.00 | 1.00 | **MATCH (100%)** |
| **Reconnaissance** | **Service_Scan** | 1.00 | 1.00 | 0.95 | 0.95 | 0.97 | 0.97 | **MATCH (100%)** |
| **Theft** | **Keylogging** | 1.00 | 1.00 | 0.99 | 0.99 | 0.99 | 0.99 | **MATCH (100%)** |
| **Theft** | **Data_Exfiltration** | 0.97 | 0.97 | 0.93 | 0.93 | 0.95 | 0.95 | **MATCH (100%)** |

---

## 4. Conclusion & Scientific Defense Takeaways

1. **Deterministic Reproducibility:** Every single precision, recall, accuracy, and F1 score across all 10 attack classes reproduces with **0.00% divergence**.
2. **Breakthrough Robustness:** The breakthrough on Data Exfiltration (**0.97 Precision vs 0.08 in base paper**) reproduces reliably.
3. **Flawless Volumetric Threat Defense:** Across all 6 DoS and DDoS attack vectors, the retrained models consistently achieve **1.00 Precision, 1.00 Recall, and 1.00 F1-score**.
4. **Fast Training Velocity:** The entire pipeline of 11 XGBoost models retrained from scratch in under **18.68 seconds**.
