

import sys
import os
import io
import numpy as np

# model.py sets the sys.path bridge — import it first
from methods.catch.model import build_catch_model  # noqa: F401
from methods.catch.loader import to_catch_dataframe
from methods.catch.training import _make_identity_scaler


def infer_catch_scores(catch_instance, test_data: np.ndarray) -> np.ndarray:
    """
    Run CATCH inference on pre-scaled test data and return per-timestep scores.

    Parameters
    ----------
    catch_instance : CATCH
        Trained CATCH wrapper returned by train_catch_model(). Must have a
        valid early_stopping.check_point (set during training).
    test_data : np.ndarray, shape (T_test, N)
        Pre-scaled test array from load_entity(). No further scaling applied.

    Returns
    -------
    scores : np.ndarray, shape (T_test,)
        Per-timestep anomaly scores. Higher = more anomalous.
        First seq_len timesteps may be zero if the test set is shorter than
        one window (extremely rare in practice).
        Tail is zero-padded if len(test_data) % seq_len != 0.
    """
    n_vars   = test_data.shape[0]
    T_test   = test_data.shape[0]
    seq_len  = catch_instance.config.seq_len

    # --- Scaler bypass ---
    # Replace the fitted scaler with an identity scaler so detect_score()'s
    # opening transform call leaves test data unchanged.
    # The identity scaler has mean_=0, scale_=1 for every feature, so
    # transform(x) = (x - 0) / 1 = x.
    n_features = test_data.shape[1]
    catch_instance.scaler = _make_identity_scaler(n_features)

    # --- Wrap test data as DataFrame ---
    # detect_score() passes the DataFrame to anomaly_detection_data_provider()
    # which feeds it into SegLoader. The datetime index is required to get
    # past the scaler transform call at the top of detect_score().
    test_df = to_catch_dataframe(test_data)

    # --- Suppress verbose per-batch printing from detect_score() ---
    # detect_score() prints testing loss values for every batch. We redirect
    # stdout to a StringIO buffer during the call and restore it after.
    _stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        raw_scores, _ = catch_instance.detect_score(test_df)
    finally:
        sys.stdout = _stdout

    # --- Align scores to len(test_data) ---
    # raw_scores has length: n_windows * seq_len, where
    #   n_windows = (T_test - seq_len) // seq_len + 1  (SegLoader 'thre' mode)
    # This covers timesteps [0 : n_windows * seq_len].
    # Any remaining timesteps [n_windows * seq_len : T_test] are not scored.
    # We zero-pad the tail so the output is exactly T_test long.
    scores = np.zeros(T_test, dtype=np.float32)
    covered = min(len(raw_scores), T_test)
    scores[:covered] = raw_scores[:covered]

    return scores
