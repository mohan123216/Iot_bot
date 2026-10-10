# Quick Cheat Sheet for Viva & Project Defense
### Hybrid Open-Set Multi-Signal Zero-Day IDS (HOMZ-Engine)

Use this quick-reference guide during your project presentation, defense, or viva examination.

---

### Q1: "Why did you use 61,130 flows instead of the full 73.3 million flows?"
**Answer:**  
In the raw UNSW Bot-IoT dataset, DoS and DDoS floods comprise **99.9%** (over 71.4 million flows), while normal traffic is only **0.013%** (9,543 flows). Training on 73.3 million flows is computationally prohibitive (~15.3 GB) and would bias the model to predict DDoS 100% of the time.  
We engineered an **Equal-Quota Streaming Reservoir** that:
1. Retained **100% of all 9,543 Normal flows** and **100% of all 1,587 Theft attacks**.
2. Capped flood attacks at **10,000 flows each**.  
This preserves the full information spectrum of rare classes without memory overload or majority class bias.

---

### Q2: "Why can't you combine DoS and DDoS into a single attack category?"
**Answer:**  
While both aim to deny service, their underlying network arrival mechanics are fundamentally different:
- **DoS:** Generated from a single attacker source IP/port with persistent TCP handshakes and sustained connection duration.
- **DDoS:** Generated from thousands of distributed botnet nodes with randomized ephemeral ports, micro-durations, and rapid fan-in.  
Combining them conflates two distinct threat profiles. Our system preserves all 10 distinct Class-Subclass combinations, exactly matching the academic standard.

---

### Q3: "What is the Softmax Overconfidence Fallacy, and why does it fail on Zero-Day attacks?"
**Answer:**  
Standard neural networks and classifiers apply a Softmax layer at the end: $\sum_{k=1}^K p_k(x) = 1.0$.  
Because the probabilities are forced to sum to 100%, an alien zero-day attack that belongs to *none* of the known classes is forced to pick the highest arbitrary known logit with near-100% false confidence. This causes baseline models to suffer from a dismal **33.3% recall** on zero-day attacks.

---

### Q4: "How does your Tier-2 Engine solve the Softmax overconfidence issue?"
**Answer:**  
We decouple classification from novelty detection using **5 independent signals**:
1. **Logit Free-Energy Score:** Operates on raw unnormalized logits ($E(x) = -T \log \sum e^{f_k/T}$), reflecting true unconstrained energy.
2. **Log-Manifold Ledoit-Wolf Mahalanobis Geometry:** Measures geometric distance from known class centroids in a regularized $\log(1+x)$ metric space.
3. **Relative Margin Ratio:** Catches samples falling in the ambiguous void between known classes.
4. **Tree Leaf-Space Traversal Novelty:** Detects if test flows traverse unprecedented paths in the 100 decision trees.
5. **Protocol-Port Semantic Anomaly:** Flags protocol-port mismatches (e.g. unidirectional UDP floods targeting port 80).  
We fuse these using **Soft-Max Pooling ($\beta = 5.0$)**, which guarantees that an anomaly in ANY single signal immediately triggers a zero-day alert.

---

### Q5: "How did you solve the Data Exfiltration Precision collapse (0.08 in the paper vs 0.97 in yours)?"
**Answer:**  
In the reference paper (*ACM TOPS 2025*), Data Exfiltration precision collapsed to **0.08 (8%)** due to the *Base Rate Fallacy*. Because exfiltration has only 118 flows, even a 5% false alarm rate on normal traffic overwhelms the true positives.  
We solved this by engineering **Directional Flow Asymmetry Features** (`spkts/dpkts` and `sbytes/dbytes`). Benign IoT telemetry exhibits bidirectional ping-ack payload symmetry, whereas Data Exfiltration exhibits severe unidirectional source-to-destination payload asymmetry. This eliminated false alarms and elevated Precision to **0.97** and F1 to **0.95**.

---

### Q6: "How did you solve Sister-Class Shadowing between OS Fingerprint and Service Scan?"
**Answer:**  
Both OS Fingerprinting and Service Scanning belong to the `Reconnaissance` family and use `Nmap`. When one was held out during training, the model absorbed it as the other, causing recall to collapse to **11%**.  
We resolved this by modeling **Port Dispersion and Flow Dynamics**:
- **Service Scanning:** Horizontal sweep across thousands of destination ports (`dport` diversity: 8,275 unique ports).
- **OS Fingerprinting:** Vertical probe targeting specific banner probes across a concentrated set of ports (1,216 ports).  
Adding port dispersion ratios lifted Service Scan recall from **11% to 95.88%**.

---

### Q7: "How do you prove that your results are not caused by Data Leakage?"
**Answer:**  
We enforced three strict integrity controls:
1. **Feature Exclusion:** Stripped all source/destination IPs (`saddr`, `daddr`), MACs, and timestamps to prevent the model from memorizing testbed IP shortcuts.
2. **Cryptographic Row-ID Tracking:** Verified via assertions that $\text{Train IDs} \cap \text{Test IDs} = \emptyset$.
3. **Zero Test Contamination:** No SMOTE or synthetic oversampling was ever applied to the test sets.

---

### Q8: "What are your 5 official evaluation metrics, and how do they compare with the base paper?"
**Answer:**  
We strictly evaluate the 5 standard metrics demanded by academic literature:
1. **Precision:** $\frac{TP}{TP + FP}$
2. **Recall:** $\frac{TP}{TP + FN}$
3. **Accuracy:** $\frac{TP + TN}{TP + TN + FP + FN}$
4. **F1-Score:** $\frac{2 \cdot P \cdot R}{P + R}$
5. **Error Rate:** $1.0 - \text{Accuracy}$

**Summary of Outperformance:**
- **DoS / DDoS Floods:** Our Precision/Recall is **1.00** across all 6 subclasses (vs 0.77-0.94 in paper).
- **Data Exfiltration:** Our Precision is **0.97** (vs 0.08 in paper).
- **Keylogging:** Our Precision is **1.00** (vs 0.80 in paper).
- **Error Rates:** Ours remain between **0.00 and 0.04** (vs 0.02 and 0.14 in paper).
