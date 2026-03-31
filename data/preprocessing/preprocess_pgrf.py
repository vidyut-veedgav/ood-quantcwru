# data/preprocessing/preprocess_pgrf.py
#
# PGRF-specific preprocessing pipeline.
#
# What makes this different from the standard preprocessing scripts:
#   1. First-order differencing is applied before scaling. PGRF's architecture
#      is designed to model the *dynamics* (changes) of a signal rather than
#      its absolute values. Differencing converts the raw signal into a
#      change-per-timestep representation, which makes the stationarity
#      assumption more plausible and helps the model detect abrupt deviations.
#
#   2. Output is per-channel (SMAP/MSL) and per-machine (SMD) rather than
#      aggregated. PGRF trains a separate model instance for each entity, so
#      each entity needs its own train/test/labels triplet on disk.
#
#   3. Everything else follows repo conventions: scaler is fit on train only,
#      test is transformed with the same scaler, both are clipped at +-3sigma.
#
# Output structure under datasets/processed/pgrf/:
#   SMAP/P-1/train.npy, test.npy, test_labels.npy
#   SMAP/P-2/...
#   MSL/M-1/...
#   SMD/machine-1-1/...
#   PSM/train.npy, test.npy, test_labels.npy

import os
import json                                    # [CHANGED] was: import ast
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_BASE = 'datasets/raw'
OUT_BASE = 'datasets/processed/pgrf'           # [ADDED] new output root; original wrote to no fixed location


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _difference(train: np.ndarray, test: np.ndarray):
    """
    Apply first-order differencing to train and test, handling the boundary
    correctly.

    For the train split, the first row differences with itself (producing a
    row of zeros), which is the standard numpy prepend=data[0:1] convention
    used in the original PGRF code.

    For the test split, the first row must difference against the *last row of
    the raw train data* — not the first row of test. Without this, the
    transition point between train and test would produce a spike that is an
    artifact of the split boundary rather than a real signal change. This is
    the key edge case that the original PGRF code gets wrong by concatenating
    train+test before differencing and then re-splitting.

    Parameters
    ----------
    train : np.ndarray, shape (T_train, N)
    test  : np.ndarray, shape (T_test,  N)

    Returns
    -------
    train_diff : np.ndarray, shape (T_train, N)
    test_diff  : np.ndarray, shape (T_test,  N)
    """
    # [SAME] same differencing logic as original apply_first_order_differencing
    train_diff = np.diff(train, axis=0, prepend=train[0:1])

    # [ADDED] correct boundary: test[0] diffs against train[-1], not test[0]
    # Original code concatenated train+test and differenced the whole array,
    # which accidentally handled the boundary but required leaky scaler fitting.
    test_prepend = np.concatenate([train[-1:], test[:-1]], axis=0)
    test_diff = test - test_prepend

    return train_diff.astype(np.float32), test_diff.astype(np.float32)


def _scale_and_clip(train: np.ndarray, test: np.ndarray):
    """
    Fit StandardScaler on train only, transform both splits, clip at +-3.

    Parameters
    ----------
    train : np.ndarray, shape (T_train, N), already differenced
    test  : np.ndarray, shape (T_test,  N), already differenced

    Returns
    -------
    scaled_train : np.ndarray
    scaled_test  : np.ndarray
    """
    # [CHANGED] original _preprocess_and_scale in data_loader_pgrf.py called
    # scaler.fit_transform on the full concatenation of train+test — data leakage.
    # Now scaler is fit on train only and test is transformed separately.
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train)
    scaled_test = scaler.transform(test)        # [CHANGED] was: fit_transform on full data

    # [ADDED] clipping was absent in the original PGRF preprocessing.
    # Repo convention: clip both splits at +-3 sigma after scaling.
    scaled_train = np.clip(scaled_train, -3, 3).astype(np.float32)
    scaled_test = np.clip(scaled_test, -3, 3).astype(np.float32)

    return scaled_train, scaled_test


def _save(out_dir: str, train: np.ndarray, test: np.ndarray, labels: np.ndarray):
    # [ADDED] entirely new helper; original code did not save to disk at all —
    # it returned arrays directly to main_pgrf.py at runtime.
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'train.npy'), train)
    np.save(os.path.join(out_dir, 'test.npy'), test)
    np.save(os.path.join(out_dir, 'test_labels.npy'), labels)


def _process_entity(train_raw: np.ndarray, test_raw: np.ndarray,
                    labels: np.ndarray, out_dir: str, entity_id: str):
    # [ADDED] entirely new helper; original code had no per-entity pipeline.

    # [DIFFERENCING DISABLED] First-order differencing has been commented out to
    # standardize preprocessing across all methods in this benchmark. CATCH and
    # future methods operate on the raw (scaled) signal, so differencing is removed
    # from PGRF to ensure a fair comparison. To re-enable, uncomment the two lines
    # below and replace train_raw/test_raw with train_diff/test_diff in the
    # _scale_and_clip call.
    # train_diff, test_diff = _difference(train_raw, test_raw)
    # train_final, test_final = _scale_and_clip(train_diff, test_diff)

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

    Output layout: datasets/processed/pgrf/<spacecraft>/<chan_id>/
                       train.npy  test.npy  test_labels.npy
    """
    assert spacecraft in ('SMAP', 'MSL')
    raw_dir = os.path.join(RAW_BASE, 'SMAP_MSL')
    anomalies_df = pd.read_csv(os.path.join(raw_dir, 'labeled_anomalies.csv'))
    channels = anomalies_df[anomalies_df['spacecraft'] == spacecraft]

    print(f'\nPreprocessing {spacecraft} ({len(channels)} channels)...')

    for _, row in channels.iterrows():
        chan_id = row['chan_id']
        train_raw = np.load(os.path.join(raw_dir, 'train', f'{chan_id}.npy')).astype(np.float32)
        test_raw = np.load(os.path.join(raw_dir, 'test', f'{chan_id}.npy')).astype(np.float32)

        if train_raw.ndim == 1:
            train_raw = train_raw.reshape(-1, 1)
        if test_raw.ndim == 1:
            test_raw = test_raw.reshape(-1, 1)

        # [CHANGED] was: ast.literal_eval(anomaly_sequences_str)
        # json.loads is correct here since the sequences are valid JSON arrays.
        labels = np.zeros(len(test_raw), dtype=np.float32)
        sequences = json.loads(row['anomaly_sequences'])
        for start, end in sequences:
            labels[start:end] = 1.0

        # [CHANGED] original _load_smap_msl returned a list of dicts with raw arrays.
        # Now we run the full pipeline and save per-channel to disk.
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

    Output layout: datasets/processed/pgrf/SMD/<machine_id>/
                       train.npy  test.npy  test_labels.npy
    """
    raw_dir = os.path.join(RAW_BASE, 'SMD')
    train_dir = os.path.join(raw_dir, 'train')
    machine_files = sorted(f for f in os.listdir(train_dir) if f.endswith('.txt'))

    print(f'\nPreprocessing SMD ({len(machine_files)} machines)...')

    for filename in machine_files:
        machine_id = filename.replace('.txt', '')
        train_path = os.path.join(raw_dir, 'train', filename)
        test_path = os.path.join(raw_dir, 'test', filename)
        label_path = os.path.join(raw_dir, 'test_label', filename)

        if not all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            print(f'  {machine_id}: missing files, skipping.')
            continue

        train_raw = np.loadtxt(train_path, delimiter=',').astype(np.float32)
        test_raw = np.loadtxt(test_path, delimiter=',').astype(np.float32)
        labels = np.loadtxt(label_path, delimiter=',').astype(np.float32)

        # [SAME] interpolation on train was also in original _load_smd.
        # [CHANGED] now applied only to train_raw before differencing,
        # not to test (test NaNs would indicate real missing data, not sensor gaps).
        train_raw = pd.DataFrame(train_raw).interpolate(
            method='linear', limit_direction='both').values.astype(np.float32)

        # [CHANGED] original _load_smd returned raw arrays; now runs full pipeline.
        out_dir = os.path.join(OUT_BASE, 'SMD', machine_id)
        _process_entity(train_raw, test_raw, labels, out_dir, machine_id)

    print('SMD done.')


def preprocess_psm():
    """
    Preprocess PSM (single entity, CSV input).

    Raw layout  : datasets/raw/PSM/
                      train.csv  test.csv  test_label.csv

    Output layout: datasets/processed/pgrf/PSM/
                       train.npy  test.npy  test_labels.npy
    """
    raw_dir = os.path.join(RAW_BASE, 'PSM')

    train_raw = pd.read_csv(os.path.join(raw_dir, 'train.csv')) \
        .drop(columns=['timestamp_(min)'])
    test_raw = pd.read_csv(os.path.join(raw_dir, 'test.csv')) \
        .drop(columns=['timestamp_(min)'])
    labels = pd.read_csv(os.path.join(raw_dir, 'test_label.csv'))['label'] \
        .values.astype(np.float32)

    # [ADDED] PSM train has NaNs in 12 columns from sensor dropout in the raw
    # CSV. Interpolate train only — consistent with how SMD train is handled.
    # Test NaNs are left as-is since they may represent real missing data.
    train_raw = train_raw.interpolate(
        method='linear', limit_direction='both').values.astype(np.float32)
    test_raw = test_raw.values.astype(np.float32)

    # [CHANGED] original _load_psm returned raw arrays; now runs full pipeline.
    out_dir = os.path.join(OUT_BASE, 'PSM')
    print('\nPreprocessing PSM...')
    _process_entity(train_raw, test_raw, labels, out_dir, 'psm')
    print('PSM done.')


def preprocess_swat():
    # [CHANGED] original _load_swat expected .xlsx files from iTrust portal.
    # Raw files present are normal.csv, attack.csv, merged.csv — different
    # format and column layout. Stub raises NotImplementedError until the
    # correct column names and label encoding are confirmed from the CSV files.
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
    print('\nAll PGRF preprocessing complete.')
    print(f'Output written to: {OUT_BASE}/')