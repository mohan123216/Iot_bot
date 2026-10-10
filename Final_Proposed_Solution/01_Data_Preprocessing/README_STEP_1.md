# Step 1: Dataset Preprocessing & Stratified Reservoir Sampling

## 📌 Executive Summary
In this step, we ingest the massive **UNSW Bot-IoT dataset (73,370,443 flows across 74 raw CSV files, ~15.3 GB)**, resolve its catastrophic class imbalance, sanitize non-generalizable network artifacts, and produce a balanced, representative **61,130-flow research corpus**.

---

## 🎯 The Core Problem: The 73.3 Million Flow Imbalance
In the raw UNSW Bot-IoT traffic:
- **DDoS & DoS Floods:** Comprise **99.9%** of the entire dataset (over 71.4 million flows).
- **Benign Normal Traffic:** Barely **0.013%** (only 9,543 flows out of 73.3 million).
- **Keylogging (Theft):** Barely **0.002%** (1,469 flows).
- **Data Exfiltration (Theft):** Barely **0.00016%** (only 118 flows).

### Why Naive Random Sampling Fails:
If you take a standard 1% or 0.1% naive random sample:
- You would get over 70,000 flood packets.
- You would get only **9 Normal packets** and **0 Data Exfiltration packets**!
- Any machine learning model trained on that would simply predict "DDoS" on every flow and achieve 99.9% naive accuracy while being **100% blind to stealth attacks and unable to recognize normal benign IoT telemetry**.

---

## 💡 Our Engineering Solution: Equal-Quota Streaming Reservoir

We engineered a **two-tier streaming sampling policy** that processes all 74 CSV files in chunks of 250,000 rows without memory crashes:
1. **100% Full Retention of Minority & Rare Classes:**
   - Retained **all 9,543 Normal flows** (100% retention).
   - Retained **all 1,469 Keylogging flows** (100% retention).
   - Retained **all 118 Data Exfiltration flows** (100% retention).
2. **Quota-Capped Sampling of Flood & Scan Classes:**
   - Capped `HTTP`, `TCP`, `UDP`, `OS_Fingerprint`, and `Service_Scan` to exactly **10,000 flows each**.

### Final Curated Dataset Distribution:
| Subcategory | Category / Role | Raw Flows in 74 Files | Curated Sample Count | Retention Strategy |
| :--- | :--- | ---:| ---:| :--- |
| **Normal** | Benign IoT Telemetry | 9,543 | **9,543** | **100% Preserved** |
| **Data Exfiltration** | Theft (Stealth Attack) | 118 | **118** | **100% Preserved** |
| **Keylogging** | Theft (Stealth Attack) | 1,469 | **1,469** | **100% Preserved** |
| **HTTP** | DoS / DDoS | 49,477 | **10,000** | Stratified Quota |
| **OS Fingerprint** | Reconnaissance | 358,275 | **10,000** | Stratified Quota |
| **Service Scan** | Reconnaissance | 1,463,364 | **10,000** | Stratified Quota |
| **TCP** | DoS / DDoS | 31,863,600 | **10,000** | Stratified Quota |
| **UDP** | DoS / DDoS | 39,624,597 | **10,000** | Stratified Quota |
| **TOTAL** | — | **73,370,443** | **61,130** | **Balanced Corpus** |

---

## 🛡️ Anti-Leakage Sanitization (Zero Shortcut Learning)
To ensure academic validity and zero data leakage:
- **Excluded Features:** `saddr` (Source IP), `daddr` (Destination IP), `smac`, `dmac`, `stime`, `ltime` (Timestamps), `pkSeqID`, `seq`.
- **Why?** An IDS must learn **network traffic dynamics and behavioral signatures**, not memorize fixed IP addresses or testbed MAC addresses. If IP addresses were included, the model would suffer from "shortcut learning" (e.g. `saddr == 192.168.1.1` => Attack), rendering it completely useless in the real world.

---

## 🚀 How to Run This Step
From this folder, run:
```powershell
python 01_stream_clean_full_dataset.py
```
- **Execution Time:** ~0.5 seconds (reads parquet instantly).
- **Outputs Produced:**
  - `dataset/fulldataset_cleaned_sample.parquet` (Optimized binary column format, 3.7 MB).
  - `dataset/fulldataset_cleaned_sample.csv` (Standard human-readable CSV, 7.3 MB).
  - `dataset/class_distribution_full_74files.csv` (Audit table).

---

## 🎤 How to Explain Step 1 in a Presentation / Viva
> *"In Step 1, we tackled the extreme class imbalance of the 73.3 million flow UNSW Bot-IoT dataset, where DoS floods represent 99.9% of traffic while benign normal traffic is only 0.013%. Instead of naive sampling which would erase the rare attacks, we implemented an Equal-Quota Streaming Reservoir. We preserved 100% of all 9,543 normal flows and all 1,587 theft attacks, while capping flood classes at 10,000 flows each. This produced a balanced 61,130-flow representative corpus with zero IP-address leakage."*
