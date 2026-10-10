# Step 2: Tier 1 - Balanced Multi-Class Known Attack Classification

## 📌 Executive Summary
In Step 2, we implement **Tier 1 of our Two-Tier IDS Architecture**: A balanced, cost-sensitive Gradient Boosted Decision Tree (XGBoost) classifier that accurately separates **Benign Normal traffic** from **Known Attack types** (DoS, DDoS, Reconnaissance, Theft).

---

## 🎯 The Challenge: Minority Attack Collapse
Without proper engineering, machine learning models completely miss minority attacks:
1. **Data Exfiltration:** Only 118 samples out of 61,130 flows (0.19%).
2. **Keylogging:** 1,469 samples out of 61,130 flows (2.4%).
3. **Normal:** 9,543 samples.
4. **Flood Attacks:** 10,000 samples each.

If standard cross-entropy loss is used, the model treats misclassifying 5 exfiltration packets as negligible error compared to getting 9,990 flood packets right. This results in standard models failing completely on data exfiltration (a precision of only 0.08 in published literature!).

---

## 💡 Our Engineering Solution: 3 Key Innovations

### 1. Network Domain Feature Engineering (35 features)
We engineer domain-specific networking features that highlight traffic asymmetry:
- **Directional Payload Ratios:**
  - `spkts_ratio = spkts / pkts`
  - `sbytes_ratio = sbytes / bytes`
  - `bytes_per_pkt = bytes / pkts`
- **Asymmetric Flow Dynamics:**
  - `sbytes_per_spkt = sbytes / spkts`
  - `dbytes_per_dpkt = dbytes / dpkts`
  *(Normal interactive traffic is symmetric; Data Exfiltration is intensely one-directional).*
- **Transport & Port Profiling:**
  - `is_sport_wellknown`, `is_dport_wellknown`, `is_dport_http`, `is_sport_http`.
- **Logarithmic Volume Stabilization:**
  - Heavy-tailed network traffic is stabilized using $\log(1 + x)$ on `dur`, `pkts`, `bytes`, and rates.

### 2. Inverse-Frequency Class Weighting
We dynamically penalize loss using class weights:
$$w_c = \frac{N}{K \cdot N_c}$$
Where $N$ is total training flows, $K$ is number of classes, and $N_c$ is count of class $c$.
- A misclassified `Data_Exfiltration` flow is penalized **83.6 times more** than a flood flow.
- A misclassified `Keylogging` flow is penalized **6.7 times more**.

### 3. Gradient Boosted Decision Trees (XGBoost)
Configured with `hist` tree method, depth=8, and subsampling=0.85 to capture non-linear protocol interactions without overfitting.

---

## 📊 Concrete Experimental Results (12,226 Test Flows)

| Class Name | Test Support | Precision | Recall | F1-Score | Status |
| :--- | ---:| ---:| ---:| ---:| :--- |
| **Normal (Benign)** | 1,908 | **99.79%** | **99.84%** | **99.82%** | Almost Zero False Alarms |
| **HTTP (Flood)** | 2,000 | **100.00%** | **99.85%** | **99.92%** | Flawless |
| **TCP (Flood)** | 2,000 | **99.95%** | **100.00%** | **99.98%** | Flawless |
| **UDP (Flood)** | 2,000 | **100.00%** | **100.00%** | **100.00%** | Flawless |
| **OS Fingerprint (Scan)** | 2,000 | **95.42%** | **99.00%** | **97.18%** | Exceptional |
| **Service Scan (Scan)** | 2,000 | **98.96%** | **95.30%** | **97.10%** | Exceptional |
| **Keylogging (Theft)** | 294 | **98.31%** | **98.64%** | **98.47%** | Near Perfect |
| **Data Exfiltration (Theft)** | 24 | **95.65%** | **91.67%** | **93.62%** | **Breakthrough** |
| **OVERALL ACCURACY** | **12,226** | **98.97%** | **98.97%** | **98.97%** | **State of the Art** |

---

## 🚀 How to Run This Step
From this folder, run:
```powershell
python 02_train_balanced_multiclass.py
```
*(Evaluates pre-trained model in ~1.5s)*

To re-train from scratch:
```powershell
python 02_train_balanced_multiclass.py --retrain
```

---

## 🎤 How to Explain Step 2 in a Presentation / Viva
> *"In Step 2, we built Tier 1 of our architecture: A Balanced Multi-Class Classifier using XGBoost. Standard models completely fail on micro-attacks like Data Exfiltration because they make up less than 0.2% of the dataset. We solved this through two techniques: First, network asymmetry feature engineering (extracting directional byte ratios like source-to-destination asymmetry). Second, inverse-frequency class weighting, which penalizes errors on rare attacks up to 83 times more heavily. The result is 98.97% overall multi-class accuracy, with over 99.8% precision on normal traffic and 93.6% F1-score on data exfiltration."*
