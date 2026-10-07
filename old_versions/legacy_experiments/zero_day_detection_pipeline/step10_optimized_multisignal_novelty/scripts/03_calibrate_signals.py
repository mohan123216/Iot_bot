"""
03_calibrate_signals.py: Fit Empirical CDF Calibration for each signal on validation known traffic
Step 10: Optimized Multi-Signal Novelty Fusion
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from common_utils import load_step10_config, get_abs_path, ensure_output_dirs, STEP10_DIR

class EmpiricalCDFNormalizer:
    """
    Nonparametric Empirical CDF calibrator.
    Maps raw novelty score S_i into calibrated percentile rank Z_i in [0.0, 1.0].
    Strict Rule: Fitted strictly on Validation Known Traffic ONLY.
    Z_i(s) = F_i(s) = P(S_known <= s)
    Higher Z_i represents higher novelty relative to known traffic.
    """
    def __init__(self):
        self.signal_names = ["signal_confidence", "signal_mahalanobis", "signal_leaf", "signal_relative"]
        self.short_keys = {
            "signal_confidence": "Z_confidence",
            "signal_mahalanobis": "Z_mahalanobis",
            "signal_leaf": "Z_leaf",
            "signal_relative": "Z_relative"
        }
        self.sorted_references = {}
        self.n_samples = {}
        self.statistics = {}
        self.percentile_table = {}

    def fit(self, val_signals_df):
        for sig in self.signal_names:
            vals = val_signals_df[sig].values.astype(np.float64)
            sorted_v = np.sort(vals)
            n = len(sorted_v)
            self.sorted_references[sig] = sorted_v
            self.n_samples[sig] = n

            # Basic descriptive statistics
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals))
            min_val = float(np.min(vals))
            max_val = float(np.max(vals))
            median_val = float(np.median(vals))
            q25 = float(np.percentile(vals, 25.0))
            q75 = float(np.percentile(vals, 75.0))
            p90 = float(np.percentile(vals, 90.0))
            p95 = float(np.percentile(vals, 95.0))
            p99 = float(np.percentile(vals, 99.0))
            p99_5 = float(np.percentile(vals, 99.5))

            self.statistics[sig] = {
                "sample_count": n,
                "mean": round(mean_val, 6),
                "std": round(std_val, 6),
                "min": round(min_val, 6),
                "p25": round(q25, 6),
                "median": round(median_val, 6),
                "p75": round(q75, 6),
                "iqr": round(q75 - q25, 6),
                "p90": round(p90, 6),
                "p95": round(p95, 6),
                "p99": round(p99, 6),
                "p99_5": round(p99_5, 6),
                "max": round(max_val, 6)
            }

            # Dense percentile grid for reference mapping (0% to 100% in 0.5% steps)
            p_grid = np.linspace(0, 100, 201)
            q_vals = np.percentile(sorted_v, p_grid)
            self.percentile_table[sig] = {
                f"p_{p:.1f}": round(float(v), 6) for p, v in zip(p_grid, q_vals)
            }

    def transform(self, signals_df):
        """
        Transforms raw signal columns into calibrated Z_i in [0.0, 1.0].
        Z_i(s) = np.searchsorted(sorted_references, s, side='right') / N
        """
        out_df = pd.DataFrame(index=signals_df.index)
        for sig in self.signal_names:
            raw_s = signals_df[sig].values.astype(np.float64)
            sorted_ref = self.sorted_references[sig]
            n = self.n_samples[sig]
            ranks = np.searchsorted(sorted_ref, raw_s, side="right") / float(n)
            col_name = self.short_keys[sig]
            out_df[col_name] = np.clip(ranks, 0.0, 1.0)
        return out_df

    def save(self, json_path):
        save_dict = {
            "signal_names": self.signal_names,
            "short_keys": self.short_keys,
            "n_samples": self.n_samples,
            "statistics": self.statistics,
            "percentile_table": self.percentile_table,
            "sorted_references": {sig: self.sorted_references[sig].tolist() for sig in self.signal_names}
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(save_dict, f, indent=2)

    def load(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.signal_names = data["signal_names"]
        self.short_keys = data["short_keys"]
        self.n_samples = data["n_samples"]
        self.statistics = data["statistics"]
        self.percentile_table = data["percentile_table"]
        self.sorted_references = {sig: np.array(data["sorted_references"][sig], dtype=np.float64) for sig in self.signal_names}

def run_calibrate_signals():
    print("=" * 80)
    print(">>> STEP 10: 03 - CALIBRATE SIGNALS (EMPIRICAL CDF RANK MAPPING) <<<")
    print("=" * 80)

    ensure_output_dirs()
    raw_val_path = os.path.join(STEP10_DIR, "outputs", "calibration", "raw_validation_signals.parquet")
    if not os.path.exists(raw_val_path):
        raise FileNotFoundError(f"Missing raw validation signals: {raw_val_path}")

    df_raw_val = pd.read_parquet(raw_val_path)
    print(f"Loaded raw signals for {len(df_raw_val):,} validation known flows.")

    normalizer = EmpiricalCDFNormalizer()
    normalizer.fit(df_raw_val)

    calib_json_path = os.path.join(STEP10_DIR, "outputs", "calibration", "per_signal_calibration.json")
    normalizer.save(calib_json_path)
    print(f"Saved calibration reference to: {calib_json_path} ({os.path.getsize(calib_json_path)/(1024*1024):.2f} MB)")

    # Transform validation set itself to verify calibration
    Z_val = normalizer.transform(df_raw_val)
    # Append predictions and ground truth for reference
    for col in ["pred_idx", "pred_class", "true_class"]:
        if col in df_raw_val.columns:
            Z_val[col] = df_raw_val[col].values

    out_calib_parquet = os.path.join(STEP10_DIR, "outputs", "calibration", "calibrated_validation_signals.parquet")
    Z_val.to_parquet(out_calib_parquet, index=False)
    print(f"Saved calibrated validation signals to: {out_calib_parquet}")

    print("\n--- Empirical CDF Calibrated Verification on Validation Known Traffic ---")
    print(f"{'Signal':<20} | {'Min':<8} | {'Median':<8} | {'Mean':<8} | {'Std':<8} | {'P95':<8} | {'Max':<8}")
    print("-" * 75)
    for k in ["Z_confidence", "Z_mahalanobis", "Z_leaf", "Z_relative"]:
        z = Z_val[k]
        print(f"{k:<20} | {z.min():<8.4f} | {z.median():<8.4f} | {z.mean():<8.4f} | {z.std():<8.4f} | {z.quantile(0.95):<8.4f} | {z.max():<8.4f}")

    return normalizer

if __name__ == "__main__":
    run_calibrate_signals()
