# Step 9: Data-Driven Weighted Multi-Signal Adaptive Novelty Detection

**Pipeline Root**: `experiments/zero_day_detection_pipeline/`  
**Status**: Completed and Verified  

## Overview
Replaces heuristic OR-based decision rules with a unified continuous novelty score:
$$S_\text{unified}(x) = w_C Z_C(x) + w_M Z_M(x) + w_L Z_L(x) + w_R Z_R(x)$$
whose signal weights are learned from calibration data (validation known traffic + pseudo-unknown samples), followed by class-conditional adaptive percentile thresholding.

## Key Results (P95 Operating Point):
- **Zero-Day Recall (`Service_Scan`)**: **32.94%** (2,405 / 7,302 flows)
- **Unknown Precision**: **46.71%**
- **Unknown F1**: **38.63%**
- **Known-Test Acceptance Rate**: **94.92%**
- **Benign (`Normal`) False Alarm Rate**: **1.41%**

## Scripts:
- `scripts/01_prepare_calibration.py`: Integrity audit and partition verification.
- `scripts/02_compute_novelty_signals.py`: Computes raw signals on validation known flows.
- `scripts/03_percentile_normalization.py`: Fits empirical CDF normalizer $Z_i \in [0, 1]$.
- `scripts/04_generate_pseudo_unknowns.py`: Synthesizes 4,500 controlled pseudo-unknown flows.
- `scripts/05_calculate_signal_quality.py`: Evaluates AUROC, AUPRC, and F1 for each signal.
- `scripts/06_learn_weights.py`: Computes AUC-derived and optimized weights.
- `scripts/07_build_unified_score.py`: Constructs unified continuous novelty score.
- `scripts/08_calibrate_adaptive_thresholds.py`: Computes class-conditional thresholds across percentiles.
- `scripts/09_evaluate_known_test.py`: Unbiased known-test evaluation.
- `scripts/10_evaluate_zero_day.py`: Final zero-day evaluation on `Service_Scan`.
- `scripts/11_generate_final_report.py`: Generates 7 publication plots (300 DPI) and full report.
