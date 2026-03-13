# scripts/inspect_psm.py
import pandas as pd
import numpy as np

train = pd.read_csv('datasets/raw/PSM/train.csv')
test  = pd.read_csv('datasets/raw/PSM/test.csv')
label = pd.read_csv('datasets/raw/PSM/test_label.csv')

print(f'train shape: {train.shape}')
print(f'test shape:  {test.shape}')
print(f'label shape: {label.shape}')
print(f'\ntrain columns: {train.columns.tolist()}')
print(f'label columns: {label.columns.tolist()}')
print(f'\ntrain head:\n{train.head(3)}')
print(f'\nlabel head:\n{label.head(3)}')