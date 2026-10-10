#!/usr/bin/env python3
"""
========================================================================================
MASTER PIPELINE RUNNER: FINAL PROPOSED SOLUTION
========================================================================================
Project: Hybrid Open-Set Multi-Signal Zero-Day Intrusion Detection Engine (HOMZ-Engine)
Dataset: UNSW Bot-IoT Benchmark (73,370,443 flows)
Reference Paper: "Multi-Stage Enhanced Zero Trust IDS for Unknown Attack Detection in IoT"
                 ACM Transactions on Privacy and Security (ACM TOPS, 2025)

Executes all 6 steps sequentially:
  - Step 1: 01_stream_clean_full_dataset.py       (Data Preprocessing & Sampling)
  - Step 2: 02_train_balanced_multiclass.py       (Tier 1 Balanced Multi-Class Classifier)
  - Step 3: 03_run_zero_day_engine.py             (Tier 2 Multi-Signal Zero-Day Engine)
  - Step 4: 04_evaluate_paper_table9.py           (Academic Base Paper Table 9 Benchmark)
  - Step 5: 05_rigorous_dual_evaluation_audit.py  (Data Leakage & Integrity Audit)
  - Step 6: generate_presentation_dashboards.py   (Publication Figures & Visual Dashboards)
========================================================================================
"""

import os
import sys
import time
import subprocess

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    {
        "step_num": 1,
        "title": "STEP 1: DATA PREPROCESSING & STREAMING RESERVOIR SAMPLING",
        "script": os.path.join(CURRENT_DIR, "01_Data_Preprocessing", "01_stream_clean_full_dataset.py"),
        "cwd": os.path.join(CURRENT_DIR, "01_Data_Preprocessing")
    },
    {
        "step_num": 2,
        "title": "STEP 2: TIER 1 - BALANCED MULTI-CLASS KNOWN ATTACK CLASSIFIER",
        "script": os.path.join(CURRENT_DIR, "02_Tier1_Known_Classification", "02_train_balanced_multiclass.py"),
        "cwd": os.path.join(CURRENT_DIR, "02_Tier1_Known_Classification")
    },
    {
        "step_num": 3,
        "title": "STEP 3: TIER 2 - MULTI-SIGNAL OPEN-SET ZERO-DAY NOVELTY ENGINE",
        "script": os.path.join(CURRENT_DIR, "03_Tier2_ZeroDay_Detection", "03_run_zero_day_engine.py"),
        "cwd": os.path.join(CURRENT_DIR, "03_Tier2_ZeroDay_Detection")
    },
    {
        "step_num": 4,
        "title": "STEP 4: OFFICIAL REPRODUCTION & OUTPERFORMANCE OF ACM TOPS TABLE 9",
        "script": os.path.join(CURRENT_DIR, "04_Paper_Table9_Benchmark", "04_evaluate_paper_table9.py"),
        "cwd": os.path.join(CURRENT_DIR, "04_Paper_Table9_Benchmark")
    },
    {
        "step_num": 5,
        "title": "STEP 5: RIGOROUS DATA INTEGRITY & ZERO LEAKAGE ASSERTION AUDIT",
        "script": os.path.join(CURRENT_DIR, "05_Integrity_Audit", "05_rigorous_dual_evaluation_audit.py"),
        "cwd": os.path.join(CURRENT_DIR, "05_Integrity_Audit")
    },
    {
        "step_num": 6,
        "title": "STEP 6: PUBLICATION-GRADE PRESENTATION DASHBOARDS GENERATOR",
        "script": os.path.join(CURRENT_DIR, "06_Presentation_Dashboards", "generate_presentation_dashboards.py"),
        "cwd": os.path.join(CURRENT_DIR, "06_Presentation_Dashboards")
    }
]

def main():
    total_start = time.time()
    print("=" * 90, flush=True)
    print(">>> HYBRID MULTI-SIGNAL ZERO-DAY IDS: MASTER PIPELINE RUNNER <<<", flush=True)
    print(">>> EXECUTING STEPS 1 TO 6 SEQUENTIALLY <<<", flush=True)
    print("=" * 90, flush=True)

    step_times = []

    for step in STEPS:
        print("\n" + "#" * 90, flush=True)
        print(f"### {step['title']} ###", flush=True)
        print("#" * 90 + "\n", flush=True)
        t0 = time.time()
        res = subprocess.run([sys.executable, "-u", step["script"]], cwd=step["cwd"], check=True)
        t_elapsed = time.time() - t0
        step_times.append((step["step_num"], step["title"], t_elapsed))
        print(f"\n>>> Step {step['step_num']} completed successfully in {t_elapsed:.2f}s <<<\n", flush=True)

    total_time = time.time() - total_start
    print("=" * 90, flush=True)
    print(">>> MASTER PIPELINE EXECUTION SUMMARY <<<", flush=True)
    print("=" * 90, flush=True)
    print(f"{'Step':<8} | {'Task Description':<62} | {'Time (s)':<10}", flush=True)
    print("-" * 90, flush=True)
    for s_num, s_desc, s_dur in step_times:
        print(f"Step {s_num:<3} | {s_desc[:60]:<62} | {s_dur:>8.2f}s", flush=True)
    print("-" * 90, flush=True)
    print(f"{'TOTAL':<8} | {'Complete End-to-End Execution':<62} | {total_time:>8.2f}s", flush=True)
    print("=" * 90, flush=True)
    print(">>> ALL PIPELINE STEPS COMPLETED WITH ZERO ERRORS! <<<", flush=True)

if __name__ == "__main__":
    main()
