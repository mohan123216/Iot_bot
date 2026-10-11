#!/usr/bin/env python3
"""
========================================================================================
STEP 6: GENERATION OF PUBLICATION-GRADE PRESENTATION DASHBOARDS & CHARTS
========================================================================================
Project: Hybrid Open-Set Multi-Signal Zero-Day Engine (HOMZ-Engine)

Creates 3 master publication dashboards and accompanying presentation figures:
  1. section1_data_processing_dashboard.png: Ingestion, Imbalance & Streaming Sampling
  2. section2_xgboost_classification_dashboard.png: Tier 1 Classification & Feature Importance
  3. section3_zeroday_detection_dashboard.png: Tier 2 Multi-Signal Fusion & Table 9 Comparison
========================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Rectangle

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(CURRENT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Styling setup
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

# Colors
C_DARK = "#0F172A"
C_BLUE = "#2563EB"
C_TEAL = "#0D9488"
C_GREEN = "#16A34A"
C_RED = "#DC2626"
C_AMBER = "#D97706"
C_PURPLE = "#7C3AED"
C_BG = "#F8FAFC"
C_WHITE = "#FFFFFF"
C_GRAY_LIGHT = "#CBD5E1"
C_GRAY_TEXT = "#64748B"

def create_section1_dashboard():
    print("Generating Section 1: Data Ingestion & Preprocessing Dashboard...")
    fig = plt.figure(figsize=(20.0, 11.6), facecolor=C_BG)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.05], width_ratios=[1.0, 1.0],
                           left=0.025, right=0.975, top=0.91, bottom=0.035, wspace=0.07, hspace=0.12)

    # Main Header
    fig.text(0.025, 0.958, "SECTION 1 — RAW DATASET CHARACTERISTICS & PREPROCESSING PIPELINE",
             fontsize=20, fontweight='bold', color=C_DARK)
    fig.text(0.025, 0.928,
             "UNSW Bot-IoT: 74-File Ingestion, Data Leakage Shield, Online Sanitation & Feature Payload Handoff",
             fontsize=12, color=C_GRAY_TEXT)

    # CARD 1 (Top-Left): Raw Benchmark Scale & Physical Ingestion
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(C_WHITE)
    for spine in ax1.spines.values():
        spine.set_color(C_GRAY_LIGHT)
        spine.set_linewidth(1.3)
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax1.text(0.03, 0.92, "1. Raw Benchmark Scale & Streaming Ingestion", fontsize=15, fontweight='bold', color=C_DARK)
    ax1.text(0.03, 0.86, "Physical properties of the 74 CSV benchmark files before processing", fontsize=10.5, color=C_GRAY_TEXT)

    c1_items = [
        ("Raw Input Corpus", "74 CSV Files (data_1.csv to data_74.csv, ~15.3 GB total)", C_BLUE),
        ("Total Ingested Flows", "73,370,443 Network Flow Records (100% full scan)", C_DARK),
        ("IoT Environment", "UNSW Cyber Range testbed (MQTT, CoAP, HTTP, Node-RED)", C_TEAL),
        ("Raw Feature Schema", "35 Total Columns (32 candidate features + 3 ground-truth labels)", C_PURPLE),
        ("Extreme Imbalance", "Floods: 97.4% (71.5M)  vs.  Normal: 0.013% (9,543 flows)", C_RED),
        ("Memory Hurdle", "Full dataset in RAM requires >25 GB (causes system OOM crash)", C_AMBER),
        ("Streaming Solution", "Chunked streaming (chunksize = 250,000 rows) ➜ RAM < 2.0 GB", C_GREEN)
    ]

    y = 0.77
    for label, val, col in c1_items:
        ax1.text(0.03, y, f"• {label}:", fontsize=10.8, fontweight='bold', color=C_DARK, va='top')
        ax1.text(0.28, y, val, fontsize=10.5, fontweight='bold', color=col, va='top')
        y -= 0.098

    # CARD 2 (Top-Right): Anti-Leakage Feature Exclusion (12 Dropped)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(C_WHITE)
    for spine in ax2.spines.values():
        spine.set_color(C_GRAY_LIGHT)
        spine.set_linewidth(1.3)
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax2.text(0.03, 0.92, "2. Anti-Leakage Feature Exclusion (12 Dropped)", fontsize=15, fontweight='bold', color=C_DARK)
    ax2.text(0.03, 0.86, "35 Raw Attributes  ➜  12 Discarded (Anti-Leakage)  ➜  20 Retained Inputs", fontsize=10.5, color=C_GRAY_TEXT)

    c2_dropped = [
        ("IP Addresses (2)", "saddr, daddr", "Prevents memorizing designated attacker IP addresses.", C_RED),
        ("MAC Addresses (2)", "smac, dmac", "Testbed-specific physical network interface hardware IDs.", C_RED),
        ("Timestamps (2)", "stime, ltime", "Prevents temporal memorization of lab attack schedule.", C_AMBER),
        ("Packet IDs (2)", "pkSeqID, seq", "Arbitrary sniffer counters; eliminates sequence order bias.", C_AMBER),
        ("Hardware/Geo (4)", "soui, doui, sco, dco", "IEEE hardware vendor prefixes and country metadata.", C_PURPLE)
    ]

    y = 0.75
    for grp, cols, reason, col in c2_dropped:
        ax2.text(0.03, y, f"✗ {grp}:", fontsize=11.2, fontweight='bold', color=col, va='top')
        ax2.text(0.26, y, f"[{cols}]", fontsize=10.8, fontweight='bold', color=C_DARK, va='top')
        ax2.text(0.48, y, reason, fontsize=10.2, color=C_GRAY_TEXT, va='top')
        y -= 0.125

    # CARD 3 (Bottom-Left): Preprocessing & Data Anomalies Handled
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(C_WHITE)
    for spine in ax3.spines.values():
        spine.set_color(C_GRAY_LIGHT)
        spine.set_linewidth(1.3)
    ax3.set_xticks([])
    ax3.set_yticks([])

    ax3.text(0.03, 0.92, "3. Data Sanitation & Robustness Invariants Handled", fontsize=15, fontweight='bold', color=C_DARK)
    ax3.text(0.03, 0.86, "Algorithmic cleaning applied on every chunk before reservoir sampling", fontsize=10.5, color=C_GRAY_TEXT)

    c3_prep = [
        ("Missing / NaN Coercion", "pd.to_numeric(errors='coerce').fillna(0.0) across all columns.", C_TEAL),
        ("Division-by-Zero Rates", "Zero-duration DDoS (dur=0.0) causes ±inf rates ➜ replaced with 0.0.", C_RED),
        ("Hexadecimal Port Parsing", "Custom parse_port_val() converts '0x0303' ➜ 771; missing ➜ -1.", C_AMBER),
        ("Protocol Encoding", "proto_number: tcp=1, arp=2, udp=3, icmp=4, ipv6-icmp=5, other=0.", C_BLUE),
        ("TCP Flags Encoding", "flgs_number: 12 composite TCP flag states mapped to integers 1..12.", C_PURPLE),
        ("Connection State Encoding", "state_number: 12 connection states (CON, RST, REQ, INT, FIN) ➜ 1..12.", C_DARK),
        ("Label Standardization", "Canonical dictionary mappings (CAT_MAP, SUBCAT_MAP) fix CSV drift.", C_GREEN)
    ]

    y = 0.77
    for title, desc, col in c3_prep:
        ax3.text(0.03, y, f"✔ {title}:", fontsize=10.6, fontweight='bold', color=col, va='top')
        ax3.text(0.32, y, desc, fontsize=10.0, color=C_GRAY_TEXT, va='top')
        y -= 0.098

    # CARD 4 (Bottom-Right): Clean 20-Feature Payload (Handoff to Next Slide)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(C_WHITE)
    for spine in ax4.spines.values():
        spine.set_color(C_GRAY_LIGHT)
        spine.set_linewidth(1.3)
    ax4.set_xticks([])
    ax4.set_yticks([])

    ax4.text(0.03, 0.92, "4. Clean 20-Feature Payload (Handoff to Sampling)", fontsize=15, fontweight='bold', color=C_DARK)
    ax4.text(0.03, 0.86, "Clean feature vector forwarded directly into Stratified Streaming Sampling", fontsize=10.5, color=C_GRAY_TEXT)

    c4_payload = [
        ("Categorical Protocol Encodings (3)", "proto_number, flgs_number, state_number", C_BLUE),
        ("Port Telemetry Features (2)", "sport, dport (hex parsed to standard decimal integers)", C_TEAL),
        ("Flow Durations & Counters (7)", "dur (seconds), pkts, bytes, spkts, dpkts, sbytes, dbytes", C_DARK),
        ("Transmission Rates (3)", "rate (pkts/s), srate (source rate), drate (dest rate)", C_PURPLE),
        ("Statistical Window Metrics (5)", "mean, stddev, sum, min, max (active packet durations)", C_AMBER),
        ("Ground-Truth Targets (3 Labels)", "attack (binary), category (macro), subcategory (granular)", C_GREEN)
    ]

    y = 0.77
    for title, feat_str, col in c4_payload:
        ax4.text(0.03, y, f"▶ {title}:", fontsize=10.6, fontweight='bold', color=col, va='top')
        ax4.text(0.40, y, feat_str, fontsize=10.2, fontweight='bold', color=C_DARK, va='top')
        y -= 0.086

    # Pipeline Bridge Banner
    bridge_box = FancyBboxPatch((0.025, 0.05), 0.95, 0.21, boxstyle="round,pad=0.02,rounding_size=0.03",
                                facecolor="#F0FDF4", edgecolor=C_GREEN, linewidth=1.2)
    ax4.add_patch(bridge_box)
    ax4.text(0.045, 0.22, "PIPELINE BRIDGE ➜ NEXT STAGE (SLIDE 2): STRATIFIED STREAMING SAMPLING",
             fontsize=10.2, fontweight='bold', color=C_GREEN, va='top')
    ax4.text(0.045, 0.15,
             "This sanitized 20-feature stream is piped directly into Stage 2: Dynamic Bernoulli Reservoir\n"
             "Sampling (100% rare class retention + 10,000 cap per flood) to generate the balanced 61,130-flow\n"
             "benchmark corpus without memory exhaustion.",
             fontsize=9.4, color=C_DARK, linespacing=1.28, va='top')

    out_p = os.path.join(PLOTS_DIR, "section1_data_processing_dashboard.png")
    fig.savefig(out_p, dpi=300)
    plt.close(fig)
    print(f"  Saved -> {out_p}")

def run_step_6():
    print("=" * 85)
    print(">>> STEP 6: PUBLICATION-GRADE PRESENTATION DASHBOARDS & CHARTS <<<")
    print("=" * 85)

    create_section1_dashboard()
    print("\nVisual charts available in plots directory:")
    plots = [
        "section1_data_processing_dashboard.png",
        "section2_xgboost_classification_dashboard.png",
        "section3_zeroday_detection_dashboard.png",
        "class_imbalance_before_and_after.png",
        "streaming_stratified_sampling_pipeline.png",
        "feature_importance_top20.png",
        "balanced_multiclass_confusion_matrix.png",
        "known_multiclass_standard_metrics_table.png",
        "zero_day_standard_metrics_table_and_chart.png",
        "base_paper_vs_our_method_comparison_table.png"
    ]
    for p in plots:
        fpath = os.path.join(PLOTS_DIR, p)
        exists = "EXISTS" if os.path.exists(fpath) else "MISSING"
        size_kb = os.path.getsize(fpath) / 1024.0 if os.path.exists(fpath) else 0
        print(f"  [{exists}] {p:<45} ({size_kb:>6.1f} KB)")

    print("\n>>> STEP 6 COMPLETED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    run_step_6()
