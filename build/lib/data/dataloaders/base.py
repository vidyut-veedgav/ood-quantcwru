# data/dataloaders/base.py
from abc import ABC, abstractmethod
import torch
from torch.utils.data import Dataset

class BaseTimeSeriesDataset(Dataset, ABC):
    """
    Abstract base class for all time series anomaly detection datasets.
    Subclasses must implement load(), which returns (data, labels) as numpy float32 arrays.
    """

    def __init__(self, processed_dir: str, split: str, window_size: int, step: int = 1):
        assert split in ('train', 'test'), "split must be 'train' or 'test'"
        
        self.processed_dir = processed_dir
        self.split = split
        self.window_size = window_size
        self.step = step

        self.data, self.labels = self.load(processed_dir, split)
        # Build list of valid window start positions
        self.indices = list(range(0, len(self.data) - window_size + 1, step))

    @abstractmethod
    def load(self, processed_dir: str, split: str):
        """
        Load preprocessed data for the given split.
        Returns:
            data:   np.ndarray of shape (T, n_features), dtype float32
            labels: np.ndarray of shape (T,), dtype float32, 0=normal 1=anomaly
        """
        pass

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        end   = start + self.window_size
        x = torch.tensor(self.data[start:end])     # shape: (window_size, n_features)
        y = torch.tensor(self.labels[start:end])   # shape: (window_size,)
        return x, y