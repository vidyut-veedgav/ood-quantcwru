import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

RAW_DIR = 'datasets/raw/PSM'
OUT_DIR = 'datasets/processed/PSM'

def preprocess():
    # Load raw files
    train_df = pd.read_csv(f'{RAW_DIR}/train.csv')
    test_df  = pd.read_csv(f'{RAW_DIR}/test.csv')
    label_df = pd.read_csv(f'{RAW_DIR}/test_label.csv')

    # # Drop timestamp column — not a feature
    # train = train_df.drop(columns=['timestamp_(min)']).values.astype(np.float32)
    # test  = test_df.drop(columns=['timestamp_(min)']).values.astype(np.float32)

    # # Extract label column as flat array
    # labels = label_df['label'].values.astype(np.float32)

    # Drop timestamp column — not a feature
    train = train_df.drop(columns=['timestamp_(min)']).values.astype(np.float32)
    test  = test_df.drop(columns=['timestamp_(min)']).values.astype(np.float32)

    # Extract label column as flat array
    labels = label_df['label'].values.astype(np.float32)

    # Fix NaNs BEFORE scaling
    print("NaNs before fix (train):", np.isnan(train).sum())
    print("NaNs before fix (test):", np.isnan(test).sum())

    # Compute column means from TRAIN ONLY
    col_means = np.nanmean(train, axis=0)

    # Fill train NaNs
    inds = np.where(np.isnan(train))
    train[inds] = np.take(col_means, inds[1])

    # Fit scaler on train only
    scaler = StandardScaler()
    train  = scaler.fit_transform(train)
    test   = scaler.transform(test)

    # Clip outliers
    train  = np.clip(train, -3, 3)
    test   = np.clip(test,  -3, 3)

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(f'{OUT_DIR}/train.npy',       train)
    np.save(f'{OUT_DIR}/test.npy',        test)
    np.save(f'{OUT_DIR}/test_labels.npy', labels)

    print(f'PSM saved to {OUT_DIR}')
    print(f'  train shape:   {train.shape}')    # expect (132481, 25)
    print(f'  test shape:    {test.shape}')     # expect (87841, 25)
    print(f'  anomaly ratio: {labels.mean():.2%}')  # expect ~27%

if __name__ == '__main__':
    preprocess()