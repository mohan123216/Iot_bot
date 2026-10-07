"""
common_utils.py: Shared utilities and models loader for Step 10
Optimized Multi-Signal Novelty Fusion with Learned Weights and Adaptive Thresholds
"""

import os
import sys
import json
import yaml
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STEP10_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
BASE_DIR = os.path.abspath(os.path.join(STEP10_DIR, ".."))
CONFIG_PATH = os.path.join(STEP10_DIR, "config.yaml")

def load_step10_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_abs_path(rel_path):
    if os.path.isabs(rel_path):
        return rel_path
    # relative to workspace root (parent of experiments)
    workspace_root = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
    p1 = os.path.join(workspace_root, rel_path)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(BASE_DIR, rel_path)
    if os.path.exists(p2):
        return p2
    return p1

def ensure_output_dirs():
    subdirs = [
        "calibration",
        "pseudo_unknown",
        "signal_analysis",
        "weights",
        "thresholds",
        "ablation",
        "known_test",
        "zero_day",
        "plots",
        "reports"
    ]
    for sd in subdirs:
        path = os.path.join(STEP10_DIR, "outputs", sd)
        os.makedirs(path, exist_ok=True)

def load_frozen_artifacts():
    cfg = load_step10_config()
    model_path = get_abs_path(cfg["paths"]["model_path"])
    mapping_path = get_abs_path(cfg["paths"]["class_mapping_path"])
    mah_stats_path = get_abs_path(cfg["paths"]["mahalanobis_stats_path"])
    leaf_prof_path = get_abs_path(cfg["paths"]["leaf_profiles_path"])

    # 1. Load Model
    clf = xgb.XGBClassifier()
    clf.load_model(model_path)
    booster = clf.get_booster()
    num_trees = booster.num_boosted_rounds() * 7
    if num_trees != 700:
        raise ValueError(f"Tree count mismatch! Expected 700, got {num_trees}")

    # 2. Load Mapping
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    # 3. Load Mahalanobis Stats
    with open(mah_stats_path, "rb") as f:
        mah_stats = pickle.load(f)

    # 4. Load Leaf Profiles
    with open(leaf_prof_path, "rb") as f:
        leaf_profiles = pickle.load(f)

    return {
        "clf": clf,
        "booster": booster,
        "num_trees": num_trees,
        "mapping": mapping,
        "mah_stats": mah_stats,
        "leaf_profiles": leaf_profiles,
        "cfg": cfg
    }
