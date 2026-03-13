import numpy as np
from torch.utils.data import DataLoader
from data.dataloaders.psm import PSMDataset

# full array check
for split in ['train', 'test']:
    arr = np.load(f'datasets/processed/PSM/{split}.npy')
    print(f'PSM {split} mean: {arr.mean():.4f} std: {arr.std():.4f}')

labels = np.load('datasets/processed/PSM/test_labels.npy')
print(f'anomaly ratio: {labels.mean():.2%}') # expect ~27%

# Dataloader check
for split in ['train', 'test']:
    ds     = PSMDataset('datasets/processed', split, window_size=100)
    loader = DataLoader(ds, batch_size=32, shuffle=(split == 'train'))
    x, y   = next(iter(loader))
    print(f'\nPSM {split}')
    print(f'  x shape: {x.shape}')   # expect (32, 100, 25)
    print(f'  y shape: {y.shape}')   # expect (32, 100)
    print(f'  x mean:  {x.mean():.3f}')
    print(f'  x std:   {x.std():.3f}')
