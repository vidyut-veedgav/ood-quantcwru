import numpy as np
from torch.utils.data import DataLoader
from data.dataloaders.swat import SWaTDataset

# full array check
for split in ['train', 'test']:
    arr = np.load(f'datasets/processed/SWaT/{split}.npy')
    print(f'SWaT {split} mean: {arr.mean():.4f} std: {arr.std():.4f}')

labels = np.load('datasets/processed/SWaT/test_labels.npy')
print(f'anomaly ration: {labels.mean():.2%}') # expect 3.79%

# dataloader check
for split in ['train', 'test']:
    ds = SWaTDataset('datasets/processed', split, window_size = 100)
    loader = DataLoader(ds, batch_size = 32, shuffle = (split == 'train'))
    x, y = next(iter(loader))
    print(f'\nSWaT {split}')
    print(f'  x shape: {x.shape}')   # expect (32, 100, 51)
    print(f'  y shape: {y.shape}')   # expect (32, 100)
    print(f'  x mean:  {x.mean():.3f}')
    print(f'  x std:   {x.std():.3f}')