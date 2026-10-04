"""
03_percentile_normalization.py: Nonparametric Empirical CDF Rank Normalizer
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from common_utils import load_step9_config, get_abs_path

class PercentileNormalizer:
    def __init__(self):
        self.signal_names = ["signal_confidence", "signal_mahalanobis", "signal_leaf", "signal_relative"]
        self.sorted_references = {}
        self.n_samples = {}
        self.quantiles_summary = {}

    def fit(self, val_signals_df):
        for sig in self.signal_names:
            vals = val_signals_df[sig].values.astype(np.float64)
            # Sort for fast vectorized binary search
            sorted_v = np.sort(vals)
            self.sorted_references[sig] = sorted_v
            self.n_samples[sig] = len(sorted_v)

            # Store summary quantiles for readable reference
            q_grid = np.linspace(0, 100, 101)
            q_vals = np.percentile(sorted_v, q_grid)
            self.quantiles_summary[sig] = {f"p_{q:.1f}": float(v) for q, v in zip(q_grid, q_vals)}

    def transform(self, signals_df):
        """
        Transforms raw signal columns into Z_i in [0.0, 1.0] using empirical CDF.
        Z_i(s) = P(Val <= s) = np.searchsorted(sorted_references, s, side='right') / N
        """
        out_df = pd.DataFrame(index=signals_df.index)
        for sig in self.signal_names:
            raw_s = signals_df[sig].values.astype(np.float64)
            sorted_ref = self.sorted_references[sig]
            n = self.n_samples[sig]
            # Empirical CDF rank
            ranks = np.searchsorted(sorted_ref, raw_s, side="right") / float(n)
            # Clip strictly to [0.0, 1.0]
            out_df[f"Z_{sig.replace('signal_', '')}"] = np.clip(ranks, 0.0, 1.0)
        return out_df

    def save(self, json_path):
        save_dict = {
            "signal_names": self.signal_names,
            "n_samples": self.n_samples,
            "quantiles_summary": self.quantiles_summary,
            "sorted_references": {sig: self.sorted_references[sig].tolist() for sig in self.signal_names}
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(save_dict, f)

    def load(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.signal_names = data["signal_names"]
        self.n_samples = data["n_samples"]
        self.quantiles_summary = data["quantiles_summary"]
        self.sorted_references = {sig: np.array(data["sorted_references"][sig], dtype=np.float64) for sig in self.signal_names}

def run_percentile_normalization():
    print("=" * 80)
    print(">>> STEP 9: 03 - FIT EMPIRICAL CDF PERCENTILE NORMALIZER <<<")
    print("=" * 80)

    step9_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    calib_scores_path = os.path.join(step9_dir, "outputs", "calibration", "calibration_signal_scores.parquet")
    if not os.path.exists(calib_scores_path):
        raise FileNotFoundError(f"Missing calibration scores: {calib_scores_path}")

    val_signals = pd.read_parquet(calib_scores_path)
    print(f"Loaded calibration scores for {len(val_signals):,} validation flows.")

    normalizer = PercentileNormalizer()
    normalizer.fit(val_signals)

    ref_json_path = os.path.join(step9_dir, "outputs", "calibration", "percentile_reference.json")
    normalizer.save(ref_json_path)
    print(f"Saved percentile normalization reference to: {ref_json_path} ({os.path.getsize(ref_json_path)/(1024*1024):.2f} MB)")

    # Verify normalization on validation set itself
    Z_val = normalizer.transform(val_signals)
    print("\nEmpirical CDF Verification on Validation Known Traffic (Should be ~Uniform in [0, 1]):")
    for col in Z_val.columns:
        print(f"  {col:<15}: Min={Z_val[col].min():.4f} | Median={Z_val[col].median():.4f} | Max={Z_val[col].max():.4f} | P95={Z_val[col].quantile(0.95):.4f}")

    return normalizer

if __name__ == "__main__":
    run_percentile_normalization()
