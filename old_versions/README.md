# Archive: Old Versions & Prototype Experiments

This directory contains historical iterations, research prototypes, and initial step-by-step implementations developed prior to the unified Full-Dataset Implementation.

---

## Directory Contents

### 1. `legacy_experiments/`
Contains the incremental research prototypes (`step4` through `step10`) on preliminary subsamples:
- **`step4/`**: Distance & Mahalanobis novelty baseline.
- **`step5/`**: Tree leaf-space novelty profiling.
- **`step6/`**: Initial hybrid open-set detection framework.
- **`step7/`**: Robustness and noise ablation experiments.
- **`step8/`**: Attack family generalization experiments.
- **`step9_weighted_multisignal_novelty/`**: 4-signal weighted novelty fusion.
- **`step10_optimized_multisignal_novelty/`**: Simplex-optimized novelty weights on prototype splits.

### 2. `superseded_scripts/`
Contains earlier iterations of scripts that were upgraded:
- **`02_loao_multisignal_pipeline.py`**: Version 1 baseline 4-signal LOAO pipeline.
- **`02b_advanced_zero_day_engine.py`**: Version 2 advanced energy engine prototype.
- **`03_generate_comprehensive_plots.py`**: Version 1 baseline figure plotter.

### 3. `legacy_outputs/`
Contains outputs produced by earlier pipeline iterations:
- **`loao_evaluations/`**: `loao_master_results.csv`, `loao_advanced_master_results.csv`, etc.
- **`weights_and_thresholds/`**: Baseline simplex learned weights & thresholds.
- **`plots/`**: Earlier v1 evaluation charts.

---

## Current Active Implementation
All active, production-grade code evaluated across all **73,370,443 flows** is located in:
- `../fulldataset_implementation/`
  - Core scripts: `scripts/`
  - Cleaned corpus: `data/fulldataset_cleaned_sample.parquet`
  - Configuration: `configs/config.yaml`
  - Models: `models/`
- Centralized outputs: `../outputs/`
