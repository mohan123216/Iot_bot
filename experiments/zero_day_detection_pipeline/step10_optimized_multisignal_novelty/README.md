# Step 10: Optimized Multi-Signal Novelty Fusion with Learned Weights and Adaptive Thresholds

**Project:** Robust Zero-Day Attack Detection in IoT Network Traffic via Open-Set Recognition  
**Experiment Name:** `step10_optimized_multisignal_novelty`

---

## 1. Overview & Objectives

Step 10 advances multi-signal novelty detection beyond the heuristic weighting methods of Step 9 by implementing a mathematically principled, constrained optimization framework.

### Four Key Innovations:
1. **Per-Signal Empirical CDF Calibration ($Z_i = F_i(S_i)$):** Maps raw scores of diverse units and scales into uniform, calibrated percentile ranks under legitimate known traffic.
2. **Resolution of Leaf-Space Novelty Degradation:** Restored the class-conditional tree frequency profiling, enabling leaf novelty to serve as an informative structural regularizer.
3. **Constrained Open-Set Weight Optimization:** Explores the 4-simplex ($\sum w_i = 1, w_i \ge 0$) to optimize pseudo-unknown detection while strictly enforcing operational bounds on known traffic acceptance ($\ge 95\%$) and benign false alarms ($\le 2\%$).
4. **Adaptive Class-Conditional Thresholding:** Evaluates thresholds $\tau_c(p)$ tailored to each closed-set predicted class across fine percentiles (P90 to P99.5).

---

## 2. Directory Structure

```text
step10_optimized_multisignal_novelty/
│
├── config.yaml                      # Hyperparameters, paths, feature lists, loss weights
├── README.md                        # Documentation and replication instructions
├── run_step10_pipeline.py           # Master end-to-end execution runner
│
├── scripts/
│   ├── common_utils.py              # Shared artifact loaders and path resolvers
│   ├── 01_prepare_calibration.py    # Zero-day quarantine and model verification
│   ├── 02_compute_raw_signals.py    # Vectorized raw signal computation (S_C, S_M, S_L, S_R)
│   ├── 03_calibrate_signals.py      # Nonparametric empirical CDF normalizer fitting
│   ├── 04_generate_pseudo_unknowns.py # Multi-strategy pseudo-unknown synthesis (N=4,500)
│   ├── 05_evaluate_signal_quality.py # AUROC, AUPRC, and monotonicity verification
│   ├── 06_optimize_weights.py       # Simplex grid search and SLSQP optimization
│   ├── 07_validate_weight_robustness.py # Cross-mechanism stability and class-conditional study
│   ├── 08_build_unified_score.py    # Continuous score fusion builder
│   ├── 09_optimize_thresholds.py    # Class-conditional adaptive threshold calibration
│   ├── 10_evaluate_known_test.py    # Unbiased known-traffic benchmark evaluation
│   ├── 11_evaluate_zero_day.py      # Held-out zero-day (Service_Scan) one-time evaluation
│   ├── 12_ablation_study.py         # 11-variant ablation study across signals and fusion
│   ├── 13_generate_plots.py         # 16 publication-quality figures (300 DPI)
│   └── 14_generate_final_report.py  # Comprehensive markdown report and manifest
│
└── outputs/
    ├── calibration/                 # Empirical CDF parameters and calibrated signals
    ├── pseudo_unknown/              # Synthesized pseudo-unknowns and generation report
    ├── signal_analysis/             # AUROC, AUPRC, ROC points, and monotonicity check
    ├── weights/                     # Learned weights, Pareto frontier, and robustness metrics
    ├── thresholds/                  # Class-conditional thresholds and operating point report
    ├── known_test/                  # Known-test predictions, confusion matrix, metrics
    ├── zero_day/                    # Zero-day predictions, subgroup analysis, master comparison
    ├── ablation/                    # Full ablation results across 11 configurations
    ├── plots/                       # 16 high-resolution publication PNGs
    └── reports/                     # Detailed scientific research report and manifest
```

---

## 3. Strict Data-Leakage Rule

`Service_Scan` is the completely held-out zero-day attack.
- It is strictly quarantined from training, calibration, pseudo-unknown generation, weight optimization, threshold calibration, and operating point selection.
- Partition counts:
  - `train.parquet`: known classes only, `Service_Scan` count = **0**
  - `validation.parquet`: known classes only, `Service_Scan` count = **0**
  - `known_test.parquet`: known classes only, `Service_Scan` count = **0**
  - `zeroday_test.parquet`: `Service_Scan` only, count = **7,302** (100% pure)
- Evaluated **only once** in Script 11 after the complete system is permanently frozen.

---

## 4. Replication Instructions

To run the complete Step 10 pipeline from scratch:

```bash
cd experiments/zero_day_detection_pipeline/step10_optimized_multisignal_novelty
python run_step10_pipeline.py
```
