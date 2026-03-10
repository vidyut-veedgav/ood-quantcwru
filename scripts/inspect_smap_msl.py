# scripts/inspect_smap_msl.py
import numpy as np
import pandas as pd

df = pd.read_csv('datasets/raw/SMAP_MSL/labeled_anomalies.csv')
print(df.columns.tolist())
print(df.head(5))
print(df['spacecraft'].value_counts())

# Inspect one channel
ch = np.load('datasets/raw/SMAP_MSL/train/A-1.npy')
print(f"\nA-1 train shape: {ch.shape}")

ch_test = np.load('datasets/raw/SMAP_MSL/test/A-1.npy')
print(f"A-1 test shape:  {ch_test.shape}")