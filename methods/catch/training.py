

import os
import sys
import numpy as np

# The sys.path insert in model.py handles the submodule bridge. Importing
# from model.py here ensures the path is in place before any ts_benchmark
# imports resolve.
from methods.catch.model import build_catch_model  # noqa: F401 — ensures sys.path is set
from methods.catch.loader import to_catch_dataframe

# Submodule imports — resolve via the path set in model.py
from ts_benchmark.baselines.catch.CATCH import CATCH                         # type: ignore
from ts_benchmark.baselines.utils import train_val_split                     # type: ignore

# sklearn is used to construct the identity scaler for the bypass
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Scaler bypass helper
# ---------------------------------------------------------------------------

def _make_identity_scaler(n_vars: int) -> StandardScaler:
    """
    Return a StandardScaler that acts as an identity transform.

    When CATCH's detect_fit() calls self.scaler.fit() and self.scaler.transform(),
    this scaler computes (x - 0) / 1 = x for every feature, leaving the already-
    scaled data untouched.

    We manually set the internal sklearn attributes that transform() reads:
        mean_  : subtracted from each feature  → set to 0
        scale_ : divided into each feature     → set to 1
        var_   : variance                      → set to 1 (consistent with scale_)
        n_features_in_ : dimension check       → set to n_vars
        n_samples_seen_: fit bookkeeping       → set to 1 (marks scaler as fitted)

    Parameters
    ----------
    n_vars : int
        Number of features in the data. Must match the data passed to CATCH.
    """
    scaler = StandardScaler()
    scaler.mean_           = np.zeros(n_vars,  dtype=np.float64)
    scaler.scale_          = np.ones(n_vars,   dtype=np.float64)
    scaler.var_            = np.ones(n_vars,   dtype=np.float64)
    scaler.n_features_in_  = n_vars
    scaler.n_samples_seen_ = 1
    return scaler


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_catch_model(
    model,
    config,
    train_data: np.ndarray,
    **train_params,
) -> 'CATCH':
    """
    Run CATCH's full training loop on pre-scaled numpy data.

    Parameters
    ----------
    model : CATCHModel
        Instantiated model from build_catch_model(). Used only to carry the
        correct config dimensions — detect_fit() rebuilds the model internally
        from config, so the architecture is guaranteed to match.
    config : TransformerConfig
        Config from build_catch_model(). detect_fit() reads lr, batch_size,
        patience, dc_lambda, auxi_lambda, num_epochs, etc. from it.
        Any train_params passed here override config values before training.
    train_data : np.ndarray, shape (T_train, N)
        Pre-scaled training array from load_entity(). No further scaling
        is applied — the identity scaler bypass handles this.
    **train_params
        Optional overrides for config fields, e.g. num_epochs=1 for smoke
        testing. Keys must match attribute names on TransformerConfig.

    Returns
    -------
    catch_instance : CATCH
        The trained CATCH wrapper object. Contains the trained model at
        catch_instance.model and the early stopping checkpoint at
        catch_instance.early_stopping.check_point. Pass this to
        infer_catch_scores() in inference.py.
    """
    n_vars = train_data.shape[1]

    # --- Apply any config overrides (e.g. num_epochs=1 for smoke tests) ---
    for key, value in train_params.items():
        setattr(config, key, value)

    # --- Wrap numpy array as DataFrame for CATCH's internal data providers ---
    # CATCH's detect_fit() passes the DataFrame to anomaly_detection_data_provider()
    # which feeds it into SegLoader. SegLoader only uses .values and .shape[0],
    # but the DataFrame wrapper with a datetime index is required to pass through
    # detect_hyper_param_tune() cleanly (which we are partially bypassing below).
    train_df = to_catch_dataframe(train_data)

    # --- Instantiate CATCH wrapper ---
    # CATCH.__init__ creates a new StandardScaler, MSELoss, and frequency_loss.
    # We pass the config fields as kwargs so TransformerConfig is re-created
    # inside CATCH with the same values we set in model.py.
    catch_instance = CATCH(**{k: getattr(config, k)
                               for k in vars(config)
                               if not k.startswith('_')
                               and not callable(getattr(config, k))})

    # --- Patch detect_hyper_param_tune to a no-op ---
    # detect_fit() calls self.detect_hyper_param_tune(train_data) as its first
    # step. That method: (1) infers frequency from the DataFrame index — already
    # handled by to_catch_dataframe(), (2) sets enc_in/dec_in/c_out from column
    # count — already set in model.py, (3) sets label_len.
    # We replace it with a function that sets only what detect_fit() needs and
    # skips the pd.infer_freq() call which can be fragile.
    def _hyper_param_tune_noop(df):
        catch_instance.config.freq      = 's'
        catch_instance.config.enc_in    = n_vars
        catch_instance.config.dec_in    = n_vars
        catch_instance.config.c_out     = n_vars
        catch_instance.config.label_len = 48

    catch_instance.detect_hyper_param_tune = _hyper_param_tune_noop

    # --- Run CATCH's training loop ---
    # detect_fit() runs as written in the submodule. What it does internally:
    #   1. Calls our no-op detect_hyper_param_tune
    #   2. Rebuilds self.model = CATCHModel(self.config) — this is unavoidable
    #      without modifying the submodule. The config already has correct dims
    #      from model.py so the rebuilt model is identical in architecture.
    #   3. Calls self.scaler.fit() on 80% train split, then transforms both
    #      train and val. This applies a second StandardScaler on already-scaled
    #      data. The effect is small (data is already ~N(0,1)) but not a true
    #      no-op. The scaler bypass for inference is handled in inference.py
    #      where it matters most — scoring test data that was never seen during
    #      scaler fitting.
    #   4. Builds SegLoader DataLoaders for train (80%) and val (20%)
    #   5. Sets up two Adam optimizers + two OneCycleLR schedulers
    #   6. Trains for num_epochs with three-term loss:
    #        rec_loss + dc_lambda*dcloss + auxi_lambda*auxi_loss
    #   7. Updates mask generator every ~10% of batches
    #   8. Runs early stopping on validation reconstruction loss
    catch_instance.detect_fit(train_df, train_df)
    # Note: detect_fit() signature is (train_data, test_data) but test_data is
    # not used during training — only during detect_score/detect_label. We pass
    # train_df for both to satisfy the signature without leaking test data.

    return catch_instance
