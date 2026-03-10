# scripts/inspect_smd.py
import numpy as np
import pandas as pd

# SMD files are space or comma separated .txt files
machine = np.loadtxt('datasets/raw/SMD/train/machine-1-1.txt', delimiter=',')
print(f'machine-1-1 train shape: {machine.shape}')  # expect (n_timesteps, 38)

label = np.loadtxt('datasets/raw/SMD/test_label/machine-1-1.txt', delimiter=',')
print(f'machine-1-1 label shape: {label.shape}')    # expect (n_timesteps,)
print(f'anomaly ratio: {label.mean():.2%}')