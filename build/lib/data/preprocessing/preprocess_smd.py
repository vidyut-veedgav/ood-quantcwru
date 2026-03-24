import numpy as np
import os
from sklearn.preprocessing import StandardScaler

RAW_DIR = 'datasets/raw/SMD'
OUT_DIR = 'datasets/processed/SMD'

# All 28 machine IDs
MACHINES = [
    'machine-1-1', 'machine-1-2', 'machine-1-3', 'machine-1-4', 'machine-1-5',
    'machine-1-6', 'machine-1-7', 'machine-1-8',
    'machine-2-1', 'machine-2-2', 'machine-2-3', 'machine-2-4', 'machine-2-5',
    'machine-2-6', 'machine-2-7', 'machine-2-8', 'machine-2-9',
    'machine-3-1', 'machine-3-2', 'machine-3-3', 'machine-3-4', 'machine-3-5',
    'machine-3-6', 'machine-3-7', 'machine-3-8', 'machine-3-9', 'machine-3-10',
    'machine-3-11'
]

def preprocess():
    all_train, all_test, all_labels = [], [], []

    for machine in MACHINES:
        train = np.loadtxt(f'{RAW_DIR}/train/{machine}.txt',      delimiter=',').astype(np.float32)
        test  = np.loadtxt(f'{RAW_DIR}/test/{machine}.txt',       delimiter=',').astype(np.float32)
        label = np.loadtxt(f'{RAW_DIR}/test_label/{machine}.txt', delimiter=',').astype(np.float32)

        # Fit scaler on train only
        scaler = StandardScaler()
        train  = scaler.fit_transform(train)
        test   = scaler.transform(test)

        # Clip extreme outliers
        train  = np.clip(train, -3, 3)
        test   = np.clip(test,  -3, 3)

        all_train.append(train)
        all_test.append(test)
        all_labels.append(label)

    os.makedirs(OUT_DIR, exist_ok=True)

    train_arr  = np.concatenate(all_train,  axis=0)
    test_arr   = np.concatenate(all_test,   axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)

    np.save(f'{OUT_DIR}/train.npy',       train_arr)
    np.save(f'{OUT_DIR}/test.npy',        test_arr)
    np.save(f'{OUT_DIR}/test_labels.npy', labels_arr)

    print(f'SMD saved to {OUT_DIR}')
    print(f'  train shape:   {train_arr.shape}')
    print(f'  test shape:    {test_arr.shape}')
    print(f'  anomaly ratio: {labels_arr.mean():.2%}')
    print(f'  expected ~4%')

if __name__ == '__main__':
    preprocess()