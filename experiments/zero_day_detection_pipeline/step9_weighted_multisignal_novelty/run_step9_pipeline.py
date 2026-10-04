"""
run_step9_pipeline.py: Master execution script for Step 9:
Data-Driven Weighted Multi-Signal Adaptive Novelty Detection
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
    print(">>> EXECUTING COMPLETE STEP 9 PIPELINE: DATA-DRIVEN WEIGHTED MULTI-SIGNAL <<<")
    print("=" * 85)

    steps = [
        ("01_prepare_calibration.py", "01. Prepare Calibration & Integrity Verification"),
        ("02_compute_novelty_signals.py", "02. Compute Validation Novelty Signals"),
        ("03_percentile_normalization.py", "03. Fit Empirical CDF Percentile Normalizer"),
        ("04_generate_pseudo_unknowns.py", "04. Generate Multi-Strategy Pseudo-Unknown Samples"),
        ("05_calculate_signal_quality.py", "05. Calculate Novelty Signal Quality (AUROC/AUPRC)"),
        ("06_learn_weights.py", "06. Learn Data-Driven Signal Weights"),
        ("07_build_unified_score.py", "07. Build Continuous Unified Score Builder"),
        ("08_calibrate_adaptive_thresholds.py", "08. Calibrate Class-Conditional Adaptive Thresholds"),
        ("09_evaluate_known_test.py", "09. Evaluate Known-Test Traffic (54,043 flows)"),
        ("10_evaluate_zero_day.py", "10. Evaluate Held-Out Zero-Day (Service_Scan, 7,302 flows)"),
        ("11_generate_final_report.py", "11. Generate Publication Plots & Final Research Report")
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
    print(f">>> STEP 9 COMPLETE PIPELINE EXECUTED SUCCESSFULLY IN {total_elapsed:.2f}s <<<")
    print("=" * 85)

if __name__ == "__main__":
    main()
