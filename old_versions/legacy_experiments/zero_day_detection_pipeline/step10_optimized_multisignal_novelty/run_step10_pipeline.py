"""
run_step10_pipeline.py: Master execution script for Step 10
Optimized Multi-Signal Novelty Fusion with Learned Weights and Adaptive Thresholds
"""

import os
import sys
import time
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(SCRIPT_DIR, "scripts")

def main():
    total_start = time.time()
    print("=" * 85)
    print(">>> EXECUTING COMPLETE STEP 10 PIPELINE: OPTIMIZED MULTI-SIGNAL NOVELTY FUSION <<<")
    print("=" * 85)

    steps = [
        ("01_prepare_calibration.py", "01. Prepare Calibration & Zero-Day Quarantine Verification"),
        ("02_compute_raw_signals.py", "02. Compute Validation Raw Novelty Signals (4 Signals)"),
        ("03_calibrate_signals.py", "03. Fit Empirical CDF Calibration Mappings"),
        ("04_generate_pseudo_unknowns.py", "04. Generate Multi-Strategy Pseudo-Unknown Samples (N=4,500)"),
        ("05_evaluate_signal_quality.py", "05. Evaluate Novelty Signal Quality, Discrimination & Monotonicity"),
        ("06_optimize_weights.py", "06. Learn Constrained Simplex Fusion Weights"),
        ("07_validate_weight_robustness.py", "07. Validate Weight Robustness Across Mechanisms & Class-Conditional Study"),
        ("08_build_unified_score.py", "08. Build Continuous Unified Score for Validation Set"),
        ("09_optimize_thresholds.py", "09. Calibrate Adaptive Class-Conditional Thresholds & Select Operating Point"),
        ("10_evaluate_known_test.py", "10. Evaluate Known Test Traffic (54,043 flows)"),
        ("11_evaluate_zero_day.py", "11. Evaluate Held-Out Zero-Day Attack (Service_Scan, 7,302 flows)"),
        ("12_ablation_study.py", "12. Comprehensive Multi-Signal Ablation Study (11 Configurations)"),
        ("13_generate_plots.py", "13. Generate 16 Publication-Quality Figures (300 DPI)"),
        ("14_generate_final_report.py", "14. Compile Final Research Report & Experiment Manifest")
    ]

    for script_name, desc in steps:
        t0 = time.time()
        print(f"\n--- Running Step: {desc} ---")
        script_path = os.path.join(SCRIPTS_DIR, script_name)

        cmd = [sys.executable, script_path]
        res = subprocess.run(cmd, cwd=SCRIPTS_DIR)
        if res.returncode != 0:
            print(f"FAILED: {script_name} exited with return code {res.returncode}")
            sys.exit(res.returncode)

        elapsed = time.time() - t0
        print(f"Step completed in {elapsed:.2f}s.")

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 85)
    print(f">>> STEP 10 COMPLETE PIPELINE EXECUTED SUCCESSFULLY IN {total_elapsed:.2f}s ({total_elapsed/60.0:.2f} mins) <<<")
    print("=" * 85)

if __name__ == "__main__":
    main()
