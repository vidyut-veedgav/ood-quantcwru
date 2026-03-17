import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

RAW_DIR = 'datasets/raw/SWaT'
OUT_DIR = 'datasets/processed/SWaT'

def preprocess():
    normal = pd.read_csv(f'{RAW_DIR}/normal.csv')
    merged = pd.read_csv(f'{RAW_DIR}/merged.csv')

    # Strip leading spaces from column names
    normal.columns = normal.columns.str.strip()
    merged.columns = merged.columns.str.strip()

    # Drop Timestamp and label columns to get feature matrix
    drop_cols = ['Timestamp', 'Normal/Attack']
    train = normal.drop(columns=drop_cols).values.astype(np.float32)
    test  = merged.drop(columns=drop_cols).values.astype(np.float32)

    # Extract labels from merged: Attack=1, Normal=0
    labels = (merged['Normal/Attack'] == 'Attack').values.astype(np.float32)

    # Fit scaler on train only
    scaler = StandardScaler()
    train  = scaler.fit_transform(train)
    test   = scaler.transform(test)

    # Clip outliers
    train  = np.clip(train, -3, 3)
    test   = np.clip(test,  -3, 3)

    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(f'{OUT_DIR}/train.npy',       train)
    np.save(f'{OUT_DIR}/test.npy',        test)
    np.save(f'{OUT_DIR}/test_labels.npy', labels)

    print(f'SWaT saved to {OUT_DIR}')
    print(f'  train shape:   {train.shape}')    # expect (1387098, 51)
    print(f'  test shape:    {test.shape}')     # expect (1441719, 51)
    print(f'  anomaly ratio: {labels.mean():.2%}')  # expect ~3.79%

if __name__ == '__main__':
    preprocess()