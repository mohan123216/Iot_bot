# Step 3: Tier 2 - 4-Signal Open-Set Zero-Day Rejection Engine

## 📌 Executive Summary
In Step 3, we implement **Tier 2 of our Architecture: The 4-Signal Open-Set Zero-Day Rejection Engine**. This engine accomplishes what traditional machine learning classifiers cannot: **Isolating completely unseen, novel zero-day attacks without confusing them with known attacks or legitimate normal traffic.**

---

## 🎯 The Core Problem: The Softmax 100%-Sum Fallacy
Traditional machine learning and deep learning intrusion detection models use a Softmax layer at the final output:
$$P(y = k \mid x) = \frac{e^{f_k(x)}}{\sum_{j=1}^K e^{f_j(x)}}$$

Because the probabilities are mathematically forced to sum to **1.0 (100%)**:
- When an attacker deploys an entirely new zero-day attack vector, the model has NO "Unknown" category.
- It is mathematically forced to assign 100% probability across the known classes.
- Whichever class happens to produce the highest arbitrary logit is selected with high confidence.
- In published baselines, this causes zero-day recall to collapse to **33.3%**!

---

## 💡 The Canonical 4 Foundational Novelty Signals

Rather than relying on softmax probabilities, our Tier 2 engine extracts **4 independent statistical, geometric, and structural novelty signals** from the network flow:

```
Incoming Network Flow (x)
       │
       ├──► Signal 1: Confidence Novelty [S_C = 1.0 - P_max]
       │              (Measures prediction uncertainty; eliminates 100%-sum fallacy)
       │
       ├──► Signal 2: Log-Manifold Mahalanobis Geometry [S_M = D_M(x, c_pred)]
       │              (Ledoit-Wolf regularized covariance distance in log-volume space)
       │
       ├──► Signal 3: Tree Leaf-Space Traversal Novelty [S_L = 1.0 - LeafSim(x, c_pred)]
       │              (Measures path activation novelty across 100 decision trees)
       │
       └──► Signal 4: Relative Neighborhood Distance Margin [S_R = D_M(pred) / (min D_M(other) + eps)]
                      (Measures whether flow sits in the ambiguous void between known classes)
       │
       ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ 1. Non-Parametric Empirical CDF Calibration:                │
 │    Z_i(x) = P(S_known <= S_i(x)) via np.searchsorted        │
 │    Converts continuous scores into percentile ranks [0, 1]  │
 ├─────────────────────────────────────────────────────────────┤
 │ 2. Multi-Signal Fusion:                                     │
 │    S_unified(x) = sum(w_i * Z_i(x))                         │
 ├─────────────────────────────────────────────────────────────┤
 │ 3. Operational Thresholding:                                │
 │    tau = np.percentile(S_val, 95.0) on Known Validation    │
 └──────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
                   S_unified(x) > Threshold (tau)?
                  ┌─────────────┴─────────────┐
                 YES                          NO
                  │                           │
                  ▼                           ▼
         [ ZERO-DAY ATTACK ]         [ KNOWN TRAFFIC ]
         (Alien / Novel Threat)      (Normal / Known Attack)
```

---

## 🔬 Mathematical Formulation of the 4 Signals

1. **Signal 1: Confidence Novelty ($S_C$):**
   $$S_C(x) = 1.0 - \max_k P(y=k \mid x)$$
   For known classes, confidence is close to 1.0 ($S_C \approx 0$). For alien threats, output probabilities are diffuse ($S_C \approx 1$).

2. **Signal 2: Log-Manifold Ledoit-Wolf Mahalanobis Distance ($S_M$):**
   $$S_M(x) = \sqrt{(z - \mu_{\hat{y}})^T \Sigma_{\hat{y}}^{-1} (z - \mu_{\hat{y}})}$$
   Applied to logarithmic manifold $\log(1 + x)$ on continuous volume features with Ledoit-Wolf shrinkage to prevent matrix singularity.

3. **Signal 3: Decision Tree Leaf-Space Traversal Novelty ($S_L$):**
   $$S_L(x) = 1.0 - \frac{1}{T} \sum_{t=1}^T \text{Freq}_t(\text{leaf}_t(x) \mid \hat{y})$$
   Novel attacks activate leaf paths in the 100 decision trees that were rarely or never activated by training samples of that predicted class.

4. **Signal 4: Relative Neighborhood Distance Margin ($S_R$):**
   $$S_R(x) = \frac{D_M(x, \hat{y})}{\min_{c \neq \hat{y}} D_M(x, c) + \epsilon}$$
   Evaluates boundary ambiguity when a sample falls between two known centroids.

---

## 📈 Percentile Rank Calibration & Rejection Rule

### Step A: Empirical CDF Normalizer (`EmpiricalCDFNormalizer`)
Raw signals have vastly different scales. To make them comparable without assuming Gaussian distributions, we map them into **Percentile Ranks** $Z_i \in [0.0, 1.0]$:
$$Z_i(s) = F_i(s) = P(S_{\text{known}} \le s)$$
Computed efficiently via `np.searchsorted` fitted strictly on known validation traffic.

### Step B: Operational Threshold Calibration ($\tau$)
$$\tau = \text{percentile}(S_{\text{val}}, 95.0)$$
Calibrating $\tau$ at the 95th percentile of known validation traffic guarantees that:
- **~95% of known traffic is accepted**.
- **Benign normal traffic specificity is maintained at >96-99%**.

### Step C: The Rejection Decision
$$\text{Decision}(x) = \begin{cases} \text{REJECT as Zero-Day Novel Threat} & \text{if } S_{\text{unified}}(x) > \tau \\ \text{ACCEPT as Known Traffic} & \text{if } S_{\text{unified}}(x) \le \tau \end{cases}$$

---

## 🚀 How to Run This Step
From this folder, run:
```powershell
python 03_run_zero_day_engine.py
```
- **Execution Time:** ~2.5 seconds (runs live 4-signal extraction, Empirical CDF percentile fitting, and rejection demo).

---

## 🎤 How to Explain Step 3 in a Presentation / Viva
> *"In Step 3, we solved the open-set zero-day rejection problem. Standard classifiers fail because the softmax layer forces probabilities to sum to 100%, causing novel attacks to be misclassified as known ones with high confidence. Our Tier-2 engine uses 4 complementary signals: Confidence Novelty, Ledoit-Wolf regularized Mahalanobis Distance, Tree Leaf-Space Traversal Novelty, and Relative Margin Ratio. We calibrate these signals into percentile ranks using an Empirical CDF normalizer fitted on validation traffic, and set an operational threshold at the 95th percentile. If the unified score exceeds the threshold, the flow is rejected as a zero-day attack; otherwise, it is accepted as known traffic."*
