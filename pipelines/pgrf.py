# pipelines/pgrf.py
#
# Runs PGRF-Net via the EasyTSAD controller on all supported datasets.
#
# Usage (must use Python 3.11 — EasyTSAD requires <3.13):
#   /opt/homebrew/bin/python3.11 pipelines/pgrf.py

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# TSADController must be imported first to fully initialize EasyTSAD before
# pgrf_method.py imports BaseMethod — otherwise a circular import occurs.
from EasyTSAD.Controller import TSADController

# Import PGRF so it registers itself in EasyTSAD's BaseMethodMeta registry
from methods.pgrf.pgrf_method import PGRF  # noqa: F401

DATASETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))

HPARAMS = {
    'window_size': 60,
    'num_protos': 10,
    'alpha': 0.1,
    # Stage 1
    'epochs_stage1': 2,
    'lr': 1e-4,
    'batch_size': 128,
    'patience_stage1': 10,
    'focal_gamma': 2.0,
    'focal_alpha': 0.5,
    'anomaly_weight': 10.0,
    'mask_reg_weight': 0.01,
    'mask_diff_weight': 0.001,
    'acyclic_penalty_weight': 1e-4,
    'lambda1': 1e-3,
    'sparsity_lambda': 1e-3,
    'context_loss_weight_stage1': 0.01,
    'spike_loss_weight_stage1': 0.01,
    # Stage 2
    'epochs_stage2': 2,
    'lr_stage2': 1e-4,
    'patience_stage2': 5,
    'pseudo_normal_percent': 0.25,
    'gate_normal_suppress_weight': 0.1,
    'gate_entropy_weight': 0.001,
    'context_loss_weight_stage2': 0.2,
    'spike_loss_weight_stage2': 0.2,
}

controller = TSADController()
controller.set_dataset(
    datasets=["PSM"],
    dirname=DATASETS_DIR,
    dataset_type="MTS",
)
controller.run_exps(
    method="PGRF",
    training_schema="mts",
    hparams=HPARAMS,
)
