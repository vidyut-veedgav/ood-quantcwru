# data/dataloaders/smap_msl.py
import numpy as np
from data.dataloaders.base import BaseTimeSeriesDataset

class SMAPMSLDataset(BaseTimeSeriesDataset):
    """
    Dataset loader for SMAP and MSL.
    Expects preprocessed .npy files at:
        processed_dir/SMAP/train.npy
        processed_dir/SMAP/test.npy
        processed_dir/SMAP/test_labels.npy
    (same structure for MSL)
    """

    def __init__(self, processed_dir: str, spacecraft: str, split: str,
                 window_size: int = 100, step: int = 1):
        assert spacecraft in ('SMAP', 'MSL'), "spacecraft must be 'SMAP' or 'MSL'"
        self.spacecraft = spacecraft
        # Pass the spacecraft subfolder as processed_dir to base
        super().__init__(
            processed_dir=f'{processed_dir}/{spacecraft}',
            split=split,
            window_size=window_size,
            step=step
        )

    def load(self, processed_dir: str, split: str):
        if split == 'train':
            data   = np.load(f'{processed_dir}/train.npy')
            labels = np.zeros(len(data), dtype=np.float32)  # train is all normal
        else:
            data   = np.load(f'{processed_dir}/test.npy')
            labels = np.load(f'{processed_dir}/test_labels.npy')
        return data, labels