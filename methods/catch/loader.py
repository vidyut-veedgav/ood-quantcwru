

import os
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Entity directory helpers  [SAME AS pgrf_loader.py]
# ---------------------------------------------------------------------------

def get_entity_dirs(processed_catch_root: str, dataset_name: str) -> list[dict]:
    """
    Return a list of dicts describing every entity for a given dataset.

    Each dict has:
        'entity_id'  : str  — human-readable identifier
        'entity_dir' : str  — path to the directory with train/test/labels

    This function is structurally identical to get_entity_dirs() in
    data/dataloaders/pgrf_loader.py. The only difference is that callers
    pass the catch processed root (datasets/processed/catch/) rather than
    the pgrf root (datasets/processed/pgrf/).

    Parameters
    ----------
    processed_catch_root : str
        Root of the CATCH processed directory, e.g. 'datasets/processed/catch'.
    dataset_name : str
        One of 'SMAP', 'MSL', 'SMD', 'PSM'.
    """
    dataset_name = dataset_name.upper()
    dataset_dir  = os.path.join(processed_catch_root, dataset_name)

    if dataset_name == 'PSM':
        # PSM is a single entity — files live directly in the dataset dir,
        # not inside a per-entity subdirectory.
        return [{'entity_id': 'psm', 'entity_dir': dataset_dir}]

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"Processed CATCH directory not found: {dataset_dir}\n"
            f"Run methods/catch/preprocess_catch.py first."
        )

    entity_dirs = []
    for name in sorted(os.listdir(dataset_dir)):
        full_path = os.path.join(dataset_dir, name)
        if os.path.isdir(full_path):
            entity_dirs.append({'entity_id': name, 'entity_dir': full_path})

    if not entity_dirs:
        raise FileNotFoundError(
            f"No entity subdirectories found under {dataset_dir}."
        )

    return entity_dirs


# ---------------------------------------------------------------------------
# Per-entity array loader  [SAME AS pgrf_loader.py's _load(), simplified]
# ---------------------------------------------------------------------------

def load_entity(entity_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the preprocessed numpy arrays for a single entity.

    Returns raw numpy arrays — no windowing, no tensors, no scaling.
    The pipeline passes these directly to the CATCH training and inference
    wrappers, which handle windowing internally via SegLoader.

    Parameters
    ----------
    entity_dir : str
        Path to a directory containing train.npy, test.npy, test_labels.npy.

    Returns
    -------
    train_data  : np.ndarray, shape (T_train, N)
    test_data   : np.ndarray, shape (T_test,  N)
    test_labels : np.ndarray, shape (T_test,)
    """
    train_data  = np.load(os.path.join(entity_dir, 'train.npy')).astype(np.float32)
    test_data   = np.load(os.path.join(entity_dir, 'test.npy')).astype(np.float32)
    test_labels = np.load(os.path.join(entity_dir, 'test_labels.npy')).astype(np.float32)
    return train_data, test_data, test_labels


# ---------------------------------------------------------------------------
# numpy → DataFrame bridge  [CATCH-SPECIFIC — no equivalent in pgrf_loader.py]
# ---------------------------------------------------------------------------

def to_catch_dataframe(array: np.ndarray, freq: str = 's') -> pd.DataFrame:
    """
    Wrap a numpy array in a pd.DataFrame with a synthetic uniform datetime index.

    WHY THIS EXISTS:
    CATCH's wrapper class (CATCH/ts_benchmark/baselines/catch/CATCH.py) expects
    its train and test inputs to be pd.DataFrames with a datetime index. This
    requirement comes from detect_hyper_param_tune(), which calls:

        pd.infer_freq(train_data.index)

    to determine the temporal sampling rate of the data and store it as
    self.config.freq. That value is later used by CATCH's internal data
    provider (anomaly_detection_data_provider in
    CATCH/ts_benchmark/baselines/utils.py) when constructing SegLoader
    instances for training and inference.

    Rather than modifying the submodule to accept numpy arrays directly, we
    satisfy the expectation by attaching a synthetic datetime index. The
    actual data values are not changed in any way — only the pandas index is
    synthetic.

    The default frequency 's' (1-second intervals) is what CATCH's
    detect_hyper_param_tune() will fall back to when it cannot infer a
    meaningful frequency from the index. Since our datasets do not have real
    timestamps, 's' is the correct sentinel value to use here.
    Note: uppercase 'S' was deprecated in pandas 2.2 — lowercase 's' is used.

    Parameters
    ----------
    array : np.ndarray, shape (T, N)
        Pre-scaled numpy array for one split (train or test).
    freq : str
        Pandas frequency string for the synthetic datetime index.
        Default 'S' (seconds) matches CATCH's own fallback in
        detect_hyper_param_tune().

    Returns
    -------
    pd.DataFrame with shape (T, N) and a DatetimeIndex.
    """
    n_timesteps = array.shape[0]
    n_vars      = array.shape[1]

    # Build column names matching CATCH's convention (ch-1, ch-2, ...)
    # CATCH does not rely on specific column names for anomaly detection,
    # but named columns make debugging easier and match how CATCH's
    # benchmark datasets are formatted.
    columns = [f'ch-{i+1}' for i in range(n_vars)]

    # Synthetic datetime index — values are arbitrary, only the regularity
    # (uniform spacing) matters so that pd.infer_freq() returns a clean
    # frequency string ('S') rather than None or raising an exception.
    index = pd.date_range(start='2000-01-01', periods=n_timesteps, freq=freq)

    return pd.DataFrame(array, index=index, columns=columns)
