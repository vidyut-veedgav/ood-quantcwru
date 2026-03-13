# data/dataloaders/swat.py
import numpy as np
from data.dataloaders.base import BaseTimeSeriesDataset

class SWaTDataset(BaseTimeSeriesDataset):
    def __init__(self, processed_dir: str, split: str,
                 window_size: int = 100, step: int = 1):
        super().__init__(
            processed_dir=f'{processed_dir}/SWaT',
            split=split,
            window_size=window_size,
            step=step
        )

    def load(self, processed_dir: str, split: str):
        if split == 'train':
            data   = np.load(f'{processed_dir}/train.npy')
            labels = np.zeros(len(data), dtype=np.float32)
        else:
            data   = np.load(f'{processed_dir}/test.npy')
            labels = np.load(f'{processed_dir}/test_labels.npy')
        return data, labels