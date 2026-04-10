# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a time-series anomaly detection research repo built on top of [EasyTSAD](https://github.com/dawnvince/EasyTSAD). It implements and benchmarks custom anomaly detection methods against standard datasets (SMD, SMAP, MSL, PSM, SWaT).

## Environment Setup

**Python version requirement: 3.11** (EasyTSAD requires < 3.13; use `/opt/homebrew/bin/python3.11` on macOS if needed)

```bash
# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running Experiments

Each pipeline script runs a method end-to-end via EasyTSAD's controller. **Always import `TSADController` before importing any method class** — EasyTSAD uses metaclass registration and a circular import occurs otherwise.

```bash
# Run PGRF-Net on PSM dataset
/opt/homebrew/bin/python3.11 pipelines/pgrf.py

# Run KANAD on PSM dataset
/opt/homebrew/bin/python3.11 pipelines/kanad.py
```

To switch datasets, edit the `datasets` list in the pipeline script:
```python
controller.set_dataset(
    datasets=["PSM", "SMD", "SMAP", "MSL"],  # or "SWaT"
    dirname=DATASETS_DIR,
    dataset_type="MTS",
)
```

Results and logs are written to `Results/` and `TSADEval.log`.

## Architecture

### EasyTSAD Integration

Methods plug into EasyTSAD by subclassing `BaseMethod` and implementing:
- `train_valid_phase(tsTrain: MTSData)` — receives training split
- `train_valid_phase_all_in_one(tsTrains: Dict[str, MTSData])` — joint training across datasets (used by KANAD with `training_schema="all_in_one"`)
- `test_phase(tsData: MTSData)` — populates `self.__anomaly_score`
- `anomaly_score() -> np.ndarray` — returns score array aligned to test series length
- `param_statistic(save_file)` — saves model summary

The `MTSData` object provides `.train` and `.test` as `(T, N)` numpy arrays.

### Method: PGRF-Net (`methods/pgrf/`)

A two-stage Prototype-Guided Reasoning Framework for MTS anomaly detection.

**Architecture** (`model.py`):
- `FrequencyDecomposition`: splits input into time-invariant (dominant frequencies) and time-variant components via FFT masking
- Dual Conformer encoders process each component; outputs fused via `fusion_linear`
- `StructuralProtoBank`: learns a bank of causal graph masks (selected per-sample via Gumbel-Softmax); encodes inter-variable relationships
- `ContextProtoBank`: measures global context deviation via cosine distance to learned prototypes
- `SpikeProtoBank`: detects point anomalies using distance-to-nearest-prototype in a Conv+pooling embedding space
- `gate_controller`: Stage 2 module that learns to weight the four evidence streams

**Training** (`training.py`):
- **Stage 1** (`train_model_stage1`): trains the full network with a composite loss including focal MSE, structural mask regularization, acyclicity penalty (`h_func` — DAG constraint), L1 sparsity, and proto deviation terms
- **Stage 2** (`train_model_stage2`): freezes backbone; fine-tunes only `gate_controller`, `context_proto_bank`, and `spike_proto_bank` using pseudo-normal samples

**Inference** (`inference.py`): produces four MinMax-normalized score arrays (`predictive`, `structural`, `contextual`, `spike`). Final score in `pgrf_method.py` combines them as `(1 - alpha) * s_pred + alpha * weighted_expl`.

### Method: KANAD (`methods/kanad/kanad.py`)

A channel-independent Kolmogorov-Arnold-inspired network for next-step prediction. Treats each variable of an MTS independently (CI strategy): input `(T, N)` → windowed tensors `(num_windows*N, window)`. Uses cosine basis functions of configurable `order`, Conv1d layers, and BatchNorm. Anomaly score = per-step mean absolute prediction error across channels.

Supports `training_schema="all_in_one"` via `train_valid_phase_all_in_one`, which concatenates training data across all provided datasets before windowing.

### Datasets (`datasets/`)

Each dataset subfolder contains an `info.json` used by EasyTSAD to locate train/test/label files. Dataset download sources are in `README.md`.

## Key Conventions

- **Import order matters**: Always `from EasyTSAD.Controller import TSADController` before any method import in pipeline scripts.
- **Model build is lazy**: `PGRFNet` is instantiated inside `train_valid_phase` once `num_vars` (N) is known from data — not in `__init__`.
- **Score alignment**: All score arrays must be padded to match `len(tsData.test)`. PGRF pads `window_size` zeros at the front; KANAD does the same.
- **Checkpoint files**: Stage 1/2 training saves temporary checkpoints to `checkpoint_stage1.pt` / `checkpoint_stage2.pt` in the working directory.
