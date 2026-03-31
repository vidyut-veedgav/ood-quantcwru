

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_BASE = 'datasets/raw'
OUT_BASE = 'datasets/processed/catch'   # [VS PGRF] output root is catch/, not pgrf/


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# [VS PGRF] No _difference() function here. PGRF has one (currently commented
# out) that can be re-enabled if needed. CATCH operates on the raw signal so
# differencing is not applicable and is omitted entirely.

def _scale_and_clip(train: np.ndarray, test: np.ndarray):
    """
    Fit StandardScaler on train only, transform both splits, clip at +-3.

    Identical to the PGRF version. Scaler is fit on train only to avoid
    any data leakage from the test split into the scaling parameters.

    Parameters
    ----------
    train : np.ndarray, shape (T_train, N)
    test  : np.ndarray, shape (T_test,  N)

    Returns
    -------
    scaled_train : np.ndarray
    scaled_test  : np.ndarray
    """
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train)
    scaled_test  = scaler.transform(test)

    scaled_train = np.clip(scaled_train, -3, 3).astype(np.float32)
    scaled_test  = np.clip(scaled_test,  -3, 3).astype(np.float32)

    return scaled_train, scaled_test


def _save(out_dir: str, train: np.ndarray, test: np.ndarray, labels: np.ndarray):
    """Save processed arrays to disk. Identical to the PGRF version."""
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'train.npy'),       train)
    np.save(os.path.join(out_dir, 'test.npy'),        test)
    np.save(os.path.join(out_dir, 'test_labels.npy'), labels)


def _process_entity(train_raw: np.ndarray, test_raw: np.ndarray,
                    labels: np.ndarray, out_dir: str, entity_id: str):
    """
    Run the full preprocessing pipeline for a single entity and save to disk.

    [VS PGRF] Only difference: no differencing step. PGRF's _process_entity
    calls _difference() before _scale_and_clip(). Here we go straight to
    _scale_and_clip() on the raw arrays.
    """
    train_final, test_final = _scale_and_clip(train_raw, test_raw)
    _save(out_dir, train_final, test_final, labels)
    print(f'  {entity_id}: train={train_final.shape}, '
          f'test={test_final.shape}, '
          f'anomaly_ratio={labels.mean():.2%}')


# ---------------------------------------------------------------------------
# Per-dataset preprocessing functions
# ---------------------------------------------------------------------------

def preprocess_smap_msl(spacecraft: str):
    """
    Preprocess one spacecraft (SMAP or MSL) channel by channel.

    Raw layout  : datasets/raw/SMAP_MSL/
                      labeled_anomalies.csv
                      train/<chan_id>.npy
                      test/<chan_id>.npy

    Output layout: datasets/processed/catch/<spacecraft>/<chan_id>/
                       train.npy  test.npy  test_labels.npy

    [VS PGRF] Identical logic and raw data layout. Only the output root
    (catch/ vs pgrf/) and the absence of differencing differ.
    """
    assert spacecraft in ('SMAP', 'MSL')
    raw_dir = os.path.join(RAW_BASE, 'SMAP_MSL')
    anomalies_df = pd.read_csv(os.path.join(raw_dir, 'labeled_anomalies.csv'))
    channels = anomalies_df[anomalies_df['spacecraft'] == spacecraft]

    print(f'\nPreprocessing {spacecraft} ({len(channels)} channels)...')

    for _, row in channels.iterrows():
        chan_id   = row['chan_id']
        train_raw = np.load(os.path.join(raw_dir, 'train', f'{chan_id}.npy')).astype(np.float32)
        test_raw  = np.load(os.path.join(raw_dir, 'test',  f'{chan_id}.npy')).astype(np.float32)

        if train_raw.ndim == 1: train_raw = train_raw.reshape(-1, 1)
        if test_raw.ndim  == 1: test_raw  = test_raw.reshape(-1, 1)

        labels = np.zeros(len(test_raw), dtype=np.float32)
        sequences = json.loads(row['anomaly_sequences'])
        for start, end in sequences:
            labels[start:end] = 1.0

        out_dir = os.path.join(OUT_BASE, spacecraft, chan_id)
        _process_entity(train_raw, test_raw, labels, out_dir, chan_id)

    print(f'{spacecraft} done.')


def preprocess_smd():
    """
    Preprocess each of the 28 SMD machines individually.

    Raw layout  : datasets/raw/SMD/
                      train/<machine_id>.txt
                      test/<machine_id>.txt
                      test_label/<machine_id>.txt

    Output layout: datasets/processed/catch/SMD/<machine_id>/
                       train.npy  test.npy  test_labels.npy

    [VS PGRF] Identical logic. Linear interpolation is applied to the train
    split only (same as PGRF) to fill sensor dropout gaps. No differencing.
    """
    raw_dir   = os.path.join(RAW_BASE, 'SMD')
    train_dir = os.path.join(raw_dir, 'train')
    machine_files = sorted(f for f in os.listdir(train_dir) if f.endswith('.txt'))

    print(f'\nPreprocessing SMD ({len(machine_files)} machines)...')

    for filename in machine_files:
        machine_id = filename.replace('.txt', '')
        train_path = os.path.join(raw_dir, 'train',      filename)
        test_path  = os.path.join(raw_dir, 'test',       filename)
        label_path = os.path.join(raw_dir, 'test_label', filename)

        if not all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            print(f'  {machine_id}: missing files, skipping.')
            continue

        train_raw = np.loadtxt(train_path, delimiter=',').astype(np.float32)
        test_raw  = np.loadtxt(test_path,  delimiter=',').astype(np.float32)
        labels    = np.loadtxt(label_path, delimiter=',').astype(np.float32)

        # Interpolate train only — test NaNs represent real missing data.
        # [VS PGRF] Identical treatment.
        train_raw = pd.DataFrame(train_raw).interpolate(
            method='linear', limit_direction='both').values.astype(np.float32)

        out_dir = os.path.join(OUT_BASE, 'SMD', machine_id)
        _process_entity(train_raw, test_raw, labels, out_dir, machine_id)

    print('SMD done.')


def preprocess_psm():
    """
    Preprocess PSM (single entity, CSV input).

    Raw layout  : datasets/raw/PSM/
                      train.csv  test.csv  test_label.csv

    Output layout: datasets/processed/catch/PSM/
                       train.npy  test.npy  test_labels.npy

    [VS PGRF] Identical logic and raw data layout.
    """
    raw_dir = os.path.join(RAW_BASE, 'PSM')

    train_raw = pd.read_csv(os.path.join(raw_dir, 'train.csv')) \
        .drop(columns=['timestamp_(min)'])
    test_raw  = pd.read_csv(os.path.join(raw_dir, 'test.csv')) \
        .drop(columns=['timestamp_(min)'])
    labels    = pd.read_csv(os.path.join(raw_dir, 'test_label.csv'))['label'] \
        .values.astype(np.float32)

    # [ADDED] PSM train has NaNs in 12 columns from sensor dropout in the raw
    # CSV. Interpolate train only — consistent with how SMD train is handled.
    # Test NaNs are left as-is since they may represent real missing data.
    # [VS PGRF] Identical fix applied to both preprocessors at the same time.
    train_raw = train_raw.interpolate(
        method='linear', limit_direction='both').values.astype(np.float32)
    test_raw  = test_raw.values.astype(np.float32)

    out_dir = os.path.join(OUT_BASE, 'PSM')
    print('\nPreprocessing PSM...')
    _process_entity(train_raw, test_raw, labels, out_dir, 'psm')
    print('PSM done.')


def preprocess_swat():
    # [VS PGRF] Identical stub — SWaT is pending in both preprocessors until
    # the correct column layout of the raw CSV files is confirmed.
    raise NotImplementedError(
        'SWaT preprocessing is pending. Raw files (normal.csv, attack.csv) '
        'need column inspection before this can be implemented safely.'
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    preprocess_smap_msl('SMAP')
    preprocess_smap_msl('MSL')
    preprocess_smd()
    preprocess_psm()
    print('\nAll CATCH preprocessing complete.')
    print(f'Output written to: {OUT_BASE}/')
