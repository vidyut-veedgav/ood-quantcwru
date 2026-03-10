import numpy as np
import pandas as pd
import json
import os
from sklearn.preprocessing import StandardScaler

RAW_DIR  = 'datasets/raw/SMAP_MSL'
OUT_DIR  = 'datasets/processed'

def make_label_array(test_length, anomaly_sequences_str):
    labels = np.zeros(test_length, dtype=np.float32)
    sequences = json.loads(anomaly_sequences_str)   # converts string → list of [start, end]
    for start, end in sequences:
        labels[start:end] = 1.0
    return labels

def preprocess(spacecraft):
    df       = pd.read_csv(f'{RAW_DIR}/labeled_anomalies.csv')
    channels = df[df['spacecraft'] == spacecraft]

    all_train, all_test, all_labels = [], [], []

    for _, row in channels.iterrows():
        chan_id = row['chan_id']
        train   = np.load(f'{RAW_DIR}/train/{chan_id}.npy').astype(np.float32)
        test    = np.load(f'{RAW_DIR}/test/{chan_id}.npy').astype(np.float32)

        # Fit scaler on train only, apply same scaler to test
        scaler  = StandardScaler()
        train   = scaler.fit_transform(train)
        test    = scaler.transform(test)

        labels  = make_label_array(len(test), row['anomaly_sequences'])

        all_train.append(train)
        all_test.append(test)
        all_labels.append(labels)

    # Save outputs
    out_path = f'{OUT_DIR}/{spacecraft}'
    os.makedirs(out_path, exist_ok=True)

    train_arr  = np.concatenate(all_train,  axis=0)
    test_arr   = np.concatenate(all_test,   axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)

    np.save(f'{out_path}/train.npy',       train_arr)
    np.save(f'{out_path}/test.npy',        test_arr)
    np.save(f'{out_path}/test_labels.npy', labels_arr)

    print(f'\n{spacecraft} saved to {out_path}')
    print(f'  train shape:   {train_arr.shape}')
    print(f'  test shape:    {test_arr.shape}')
    print(f'  anomaly ratio: {labels_arr.mean():.2%}')

if __name__ == '__main__':
    preprocess('SMAP')
    preprocess('MSL')