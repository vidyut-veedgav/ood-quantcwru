# scripts/validate_smap_msl.py
from torch.utils.data import DataLoader
from data.dataloaders.smap_msl import SMAPMSLDataset

for spacecraft in ['SMAP', 'MSL']:
    for split in ['train', 'test']:
        ds     = SMAPMSLDataset('datasets/processed', spacecraft, split, window_size=100)
        loader = DataLoader(ds, batch_size=32, shuffle=(split == 'train'))
        x, y   = next(iter(loader))
        print(f'{spacecraft} {split}')
        print(f'  x shape: {x.shape}')   # expect (32, 100, 25) for SMAP, (32, 100, 55) for MSL
        print(f'  y shape: {y.shape}')   # expect (32, 100)
        print(f'  x mean:  {x.mean():.3f}')  # expect ≈ 0
        print(f'  x std:   {x.std():.3f}')   # expect ≈ 1

import numpy as np

print("\n--- Normalization check on full processed arrays ---")
for spacecraft in ['SMAP', 'MSL']:
    train = np.load(f'datasets/processed/{spacecraft}/train.npy')
    test  = np.load(f'datasets/processed/{spacecraft}/test.npy')
    print(f'\n{spacecraft}')
    print(f'  train mean: {train.mean():.4f}   (expect ≈ 0)')
    print(f'  train std:  {train.std():.4f}    (expect ≈ 1)')
    print(f'  test mean:  {test.mean():.4f}   (expect ≈ 0)')
    print(f'  test std:   {test.std():.4f}    (expect ≈ 1)')