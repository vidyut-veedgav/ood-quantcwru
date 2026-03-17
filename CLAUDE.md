# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ood-quantcwru** implements **PGRF-Net** (Prototypical Graph Reasoning Framework), a two-stage unsupervised time series anomaly detection model. It targets multivariate time series from finance and infrastructure monitoring (SMAP, MSL, SMD, PSM, SWaT datasets).

## Running the Code

### 1. Preprocess Data
```bash
python data/preprocessing/preprocess_pgrf.py --dataset SMAP
python data/preprocessing/preprocess_pgrf.py --dataset MSL
python data/preprocessing/preprocess_pgrf.py --dataset SMD
python data/preprocessing/preprocess_pgrf.py --dataset PSM
```
Outputs: `datasets/processed/pgrf/{dataset}/{entity}/train.npy`, `test.npy`, `test_labels.npy`

### 2. Run Full Pipeline
```bash
python pipelines/pgrf.py --dataset SMAP
python pipelines/pgrf.py --dataset SMD
```
Outputs: `results_pgrf_{dataset}.csv` with per-entity F1/Precision/Recall/AUROC/AUCPR.

### 3. Validate / Inspect Data
```bash
python scripts/validate_smd.py
python scripts/inspect_smap_msl.py
```

## Architecture

```
Raw Data → Preprocessing → [Stage 1: Evidence Extraction] → [Stage 2: Gated Fusion] → Inference → Evaluation
```

### Preprocessing (`data/preprocessing/`)
- First-order differencing → StandardScaler (fit on train only) → ±3σ clipping
- Per-entity output files (one entity = one machine/spacecraft/channel)

### Data Loading (`data/dataloaders/`)
- `BaseTimeSeriesDataset`: abstract windowed dataset returning `(window, labels)`
- `PGRFEntityDataset`: forecast-style windows — returns `(X, Y, L)` where `Y` is the next timestep target

### Model (`methods/pgrf/model.py`) — `PGRFNet`
Five components:
1. **FrequencyDecomposition** — FFT splits signal into time-invariant (top M% bins) and time-variant parts
2. **Conformer Blocks** — Multi-head attention + GLU-CNN, one each for invariant/variant streams
3. **Prototype Banks** — `StructuralProtoBank` (graph masks), `ContextProtoBank`, `SpikeProtoBank`
4. **Evidence Fusion Network** — Gate controller with Gumbel-Softmax selects among 4 evidence types
5. **Prediction Heads** — One linear predictor per variable

### Training (`methods/pgrf/training.py`)
- **Stage 1** (50 epochs): trains all components; loss = focal reconstruction + mask regularization + DAG acyclicity penalty (`h_func`) + prototype sparsity + context/spike terms
- **Stage 2** (20 epochs): freezes Stage 1; fine-tunes `gate_controller`, `context_proto_bank`, `spike_proto_bank`; loss focuses on gate entropy + normal-sample suppression
- 80/20 train/val split; early stopping (patience=10 Stage 1, 5 Stage 2)
- Checkpoints: `checkpoint_stage1.pt`, `checkpoint_stage2.pt`

### Inference (`methods/pgrf/inference.py`)
Produces 4 MinMax-normalized anomaly scores: `predictive` (MSE), `structural` (mask deviation), `contextual`, `spike`.

### Pipeline (`pipelines/pgrf.py`)
Per-entity loop: preprocess → window → Stage 1 → Stage 2 → infer → combine scores as `(1-α)*predictive + α*mean(explanatory)` (α=0.1) → `bf_search` threshold → metrics → CSV.

## Key Parameters (in `pipelines/pgrf.py`)
| Param | Value |
|-------|-------|
| `window_size` | 60 |
| `batch_size` | 128 |
| `lr` (both stages) | 1e-4 |
| `epochs_stage1` | 50 |
| `epochs_stage2` | 20 |
| `num_prototypes` | 10 |
| `alpha` (score mix) | 0.1 |

## Dependencies
No `requirements.txt` exists. Key libraries: `torch`, `numpy`, `pandas`, `scikit-learn`, `tqdm`.

## Notes
- SWaT loader (`data/dataloaders/swat.py`) is not currently working.
- No test suite exists yet.
- Dataset raw files must be downloaded separately (see README.md for links) and placed under `datasets/raw/`.
