"""
Runner for KANAD on SMD, SMAP, MSL, and PSM datasets.

Usage (must use Python 3.11 — EasyTSAD requires <3.13):
    /opt/homebrew/bin/python3.11 pipelines/kanad.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# TSADController must be imported first to fully initialize EasyTSAD before
# the method module imports BaseMethod — otherwise a circular import occurs.
from EasyTSAD.Controller import TSADController

# Import KANAD so it registers itself in EasyTSAD's BaseMethodMeta registry
from methods.kanad.kanad import KANAD  # noqa: F401

DATASETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))

HPARAMS = {
    "batch_size": 1024,
    "window": 60,
    "epochs": 50,
    "lr": 1e-3,
    "order": 5,
}

controller = TSADController()
controller.set_dataset(
    datasets=["PSM"],
    dirname=DATASETS_DIR,
    dataset_type="MTS",
)
controller.run_exps(
    method="KANAD",
    training_schema="all_in_one",
    hparams=HPARAMS,
)
