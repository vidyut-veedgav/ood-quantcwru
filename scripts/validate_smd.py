# scripts/validate_smd.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from torch.utils.data import DataLoader
from data.dataloaders.smd import SMDDataset

# Check full array normalization
for split in ['train', 'test']:
    arr = np.load(f'datasets/processed/SMD/{split}.npy')
    print(f'SMD {split} mean: {arr.mean():.4f}  std: {arr.std():.4f}')

# Check dataloader output
for split in ['train', 'test']:
    ds     = SMDDataset('datasets/processed', split, window_size=100)
    loader = DataLoader(ds, batch_size=32, shuffle=(split == 'train'))
    x, y   = next(iter(loader))
    print(f'\nSMD {split}')
    print(f'  x shape: {x.shape}')   # expect (32, 100, 38)
    print(f'  y shape: {y.shape}')   # expect (32, 100)
    print(f'  x mean:  {x.mean():.3f}')
    print(f'  x std:   {x.std():.3f}')