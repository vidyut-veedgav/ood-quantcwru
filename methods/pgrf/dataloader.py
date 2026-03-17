# data/dataloaders/pgrf_loader.py
#
# PGRF requires a forecast-style windowing scheme:
#   X : (window_size, N)  — the input context window
#   Y : (N,)              — the single next timestep to predict
#   L : scalar float      — anomaly label at that next timestep
#
# This is incompatible with BaseTimeSeriesDataset which returns
# (window, label_sequence), so this is a standalone Dataset class.
#
# The loader operates on a single preprocessed entity directory, e.g.:
#   datasets/processed/pgrf/SMAP/P-1/
#   datasets/processed/pgrf/SMD/machine-1-1/
#   datasets/processed/pgrf/PSM/
#
# Use the dataset-level helpers at the bottom of this file to get the
# list of all entity directories for a given dataset, which the PGRF
# pipeline iterates over to train one model per entity.
 
import os
import numpy as np
import torch
from torch.utils.data import Dataset
 
 
class PGRFEntityDataset(Dataset):
    """
    Forecast-window dataset for a single preprocessed entity.
 
    For each valid position i in [0, T - window_size):
        X[i] = data[i : i + window_size]          shape (window_size, N)
        Y[i] = data[i + window_size]               shape (N,)
        L[i] = labels[i + window_size]             scalar
 
    Parameters
    ----------
    entity_dir : str
        Path to the directory containing train.npy / test.npy / test_labels.npy
        for a single entity.
    split : str
        'train' or 'test'.
    window_size : int
        Number of timesteps in the input context window.
    step : int
        Stride between consecutive windows. Default 1.
    """
 
    def __init__(self, entity_dir: str, split: str,
                 window_size: int = 60, step: int = 1):
        assert split in ('train', 'test'), "split must be 'train' or 'test'"
 
        self.entity_dir = entity_dir
        self.split = split
        self.window_size = window_size
        self.step = step
 
        self.data, self.labels = self._load(entity_dir, split)
 
        # Valid start positions: window ends at i + window_size,
        # which must be < len(data) so Y exists.
        self.indices = list(range(0, len(self.data) - window_size, step))
 
    def _load(self, entity_dir: str, split: str):
        if split == 'train':
            data = np.load(os.path.join(entity_dir, 'train.npy'))
            labels = np.zeros(len(data), dtype=np.float32)
        else:
            data = np.load(os.path.join(entity_dir, 'test.npy'))
            labels = np.load(os.path.join(entity_dir, 'test_labels.npy'))
        return data.astype(np.float32), labels.astype(np.float32)
 
    def __len__(self):
        return len(self.indices)
 
    def __getitem__(self, idx):
        start = self.indices[idx]
        end = start + self.window_size
 
        x = torch.tensor(self.data[start:end],   dtype=torch.float32)  # (W, N)
        y = torch.tensor(self.data[end],          dtype=torch.float32)  # (N,)
        l = torch.tensor(self.labels[end],        dtype=torch.float32)  # scalar
        return x, y, l
 
    @property
    def num_vars(self) -> int:
        """Number of features (N). Convenience property for model init."""
        return self.data.shape[1]
 
 
# ---------------------------------------------------------------------------
# Dataset-level helpers
# ---------------------------------------------------------------------------
 
def get_entity_dirs(processed_pgrf_root: str, dataset_name: str) -> list[dict]:
    """
    Return a list of dicts describing every entity for a given dataset.
    Each dict has:
        'entity_id'  : str   — human-readable identifier
        'entity_dir' : str   — path to the directory with train/test/labels
 
    Parameters
    ----------
    processed_pgrf_root : str
        Root of the PGRF processed directory, e.g. 'datasets/processed/pgrf'.
    dataset_name : str
        One of 'SMAP', 'MSL', 'SMD', 'PSM'.
    """
    dataset_name = dataset_name.upper()
    dataset_dir = os.path.join(processed_pgrf_root, dataset_name)
 
    if dataset_name == 'PSM':
        # PSM is a single entity — files live directly in the dataset dir
        return [{'entity_id': 'psm', 'entity_dir': dataset_dir}]
 
    # SMAP, MSL, SMD: one subdirectory per entity
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"Processed PGRF directory not found: {dataset_dir}\n"
            f"Run data/preprocessing/preprocess_pgrf.py first."
        )
 
    entity_dirs = []
    for name in sorted(os.listdir(dataset_dir)):
        full_path = os.path.join(dataset_dir, name)
        if os.path.isdir(full_path):
            entity_dirs.append({'entity_id': name, 'entity_dir': full_path})
 
    if not entity_dirs:
        raise FileNotFoundError(
            f"No entity subdirectories found under {dataset_dir}."
        )
 
    return entity_dirs