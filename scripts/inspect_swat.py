import pandas as pd
import numpy as np


attacked = pd.read_csv('datasets/raw/SWaT/attack.csv')
normal = pd.read_csv('datasets/raw/SWaT/normal.csv')
merged = pd.read_csv('datasets/raw/SWaT/merged.csv')

print(f"normal shape: {normal.shape}")
print(f"attack shape: {attacked.shape}")
print(f"\nnormal columns: {normal.columns.tolist()}")
print(f"attack columns: {attacked.columns.tolist()}")
print(f"\nnormal head: \n{normal.head(3)}")
print(f"\nattack label values: {attacked.iloc[:, -1].unique()}")

# add to inspect_swat.py
print(f"\nnormal label values: {normal.iloc[:, -1].unique()}")
print(f"attack label value counts:\n{attacked['Normal/Attack'].value_counts()}")
print(f"normal label value counts:\n{normal['Normal/Attack'].value_counts()}")


# add to inspect_swat.py
print(f'\nmerged shape: {merged.shape}')
print(f'merged label value counts:\n{merged["Normal/Attack"].value_counts()}')
print(f'anomaly ratio: {(merged["Normal/Attack"] == "Attack").mean():.2%}')