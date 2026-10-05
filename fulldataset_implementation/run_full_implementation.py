#!/usr/bin/env python3
"""
run_full_implementation.py: Master end-to-end runner for full dataset zero-day detection
Executes:
  1. 01_stream_clean_full_dataset.py
  2. 02_loao_multisignal_pipeline.py
  3. 03_generate_comprehensive_plots.py
  4. 04_generate_final_report.py
"""

import os
import sys
import time
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(SCRIPT_DIR, "scripts")

def run_step(script_name):
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    print("\n" + "#" * 85)
    print(f"### LAUNCHING PIPELINE STEP: {script_name} ###")
    print("#" * 85 + "\n")
    t0 = time.time()
    res = subprocess.run([sys.executable, script_path], check=True)
    t_elapsed = time.time() - t0
    print(f"\n>>> Step {script_name} completed in {t_elapsed:.2f}s <<<\n")
    return res.returncode

def main():
    total_start = time.time()
    print("=" * 90)
    print(">>> ZERO-DAY NOVELTY DETECTION: FULL DATASET IMPLEMENTATION MASTER RUNNER <<<")
    print("=" * 90)

    steps = [
        "01_stream_clean_full_dataset.py",
        "02_loao_multisignal_pipeline.py",
        "03_generate_comprehensive_plots.py",
        "04_generate_final_report.py"
    ]

    for step in steps:
        run_step(step)

    total_time = time.time() - total_start
    print("=" * 90)
    print(f">>> ALL PIPELINE STEPS COMPLETED SUCCESSFULLY IN {total_time:.2f}s ({total_time/60.0:.2f} min) <<<")
    print("=" * 90)

if __name__ == "__main__":
    main()
