

import os
import sys

# Add the CATCH submodule root to sys.path so that imports of the form
#   from ts_benchmark.baselines.catch... import ...
# resolve correctly when running from the repo root.
# os.path.dirname(__file__) = .../ood-quantcwru/methods/catch/
# Two levels up lands at the repo root, then we enter the CATCH submodule.
_SUBMODULE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'CATCH')
)
if _SUBMODULE_PATH not in sys.path:
    sys.path.insert(0, _SUBMODULE_PATH)

# These imports now resolve via the path inserted above.
# TransformerConfig: holds all CATCH hyperparameters. CATCHModel reads from it.
# CATCHModel: the actual PyTorch nn.Module — frequency patching + channel fusion.
from ts_benchmark.baselines.catch.CATCH import TransformerConfig
from ts_benchmark.baselines.catch.models.CATCH_model import CATCHModel

import torch


# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------
# These are the CATCH paper defaults from TransformerConfig / CATCH.py.
# They can be overridden per-dataset by passing kwargs to build_catch_model().
# Keeping them here (rather than inlining in the function) makes it easy to
# see at a glance what CATCH's configuration looks like, and easy to tune.

DEFAULT_HYPER_PARAMS = {
    'seq_len':        192,   # sliding window length fed into the model
    'patch_size':     16,    # size of each frequency patch
    'patch_stride':   8,     # stride between frequency patches
    'e_layers':       3,     # number of transformer encoder layers
    'n_heads':        2,     # attention heads in the cross-channel transformer
    'd_model':        128,   # model embedding dimension
    'cf_dim':         64,    # channel fusion dimension
    'd_ff':           256,   # feedforward dimension inside transformer
    'head_dim':       64,    # per-head dimension
    'dropout':        0.2,
    'head_dropout':   0.1,
    'individual':     0,     # 0 = shared head across channels, 1 = per-channel
    'revin':          1,     # 1 = use reversible instance normalisation (RevIN)
    'affine':         0,     # RevIN affine parameters (0 = off)
    'subtract_last':  0,     # RevIN subtract_last mode (0 = off)
    'regular_lambda': 0.5,   # contrastive loss regularisation weight
    'temperature':    0.07,  # contrastive loss temperature
    'auxi_loss':      'MAE',
    'auxi_type':      'complex',
    'auxi_mode':      'fft',
    'auxi_lambda':    0.005, # weight of auxiliary frequency reconstruction loss
    'dc_lambda':      0.005, # weight of dynamical contrastive (channel discovery) loss
    'score_lambda':   0.05,  # weight of frequency score at inference time
    'lr':             0.0001,
    'Mlr':            0.00001, # separate lr for the mask generator
    'pct_start':      0.3,
    'lradj':          'type1',
    'num_epochs':     3,
    'batch_size':     128,
    'patience':       3,
    'module_first':   True,
    'mask':           False,
    'pretrained_model': None,
    'inference_patch_size':   32,
    'inference_patch_stride': 1,
    'anomaly_ratio':  [0.1, 0.5, 1.0, 2, 3, 5.0, 10.0, 15, 20, 25],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_catch_model(n_vars: int, seq_len: int, **hyperparams):
    """
    Construct a CATCHModel and its associated TransformerConfig.

    Parameters
    ----------
    n_vars : int
        Number of input variables (channels) for this entity. This is
        data-dependent and must be set at runtime — it cannot live in
        DEFAULT_HYPER_PARAMS. In CATCHModel this controls the size of the
        channel mask generator and the RevIN layer.
    seq_len : int
        Length of the sliding window used during training and inference.
        Must match the window size used in SegLoader (set in training.py).
    **hyperparams
        Any key in DEFAULT_HYPER_PARAMS can be overridden here, allowing
        per-dataset tuning without changing the defaults.

    Returns
    -------
    model : CATCHModel
        Instantiated PyTorch model, on CPU. Caller moves to GPU if needed.
    config : TransformerConfig
        The config object used to build the model. Passed to training.py
        so it can configure the optimizers, schedulers, and loss functions
        that also read from config (e.g. config.lr, config.dc_lambda).
    """
    # Merge defaults with any caller-supplied overrides.
    params = {**DEFAULT_HYPER_PARAMS, **hyperparams}

    # Build the TransformerConfig. This sets every attribute that CATCHModel
    # reads during __init__ (patch_size, patch_stride, d_model, cf_dim, etc.).
    config = TransformerConfig(**params)

    # Data-dependent fields: these depend on the specific entity being processed
    # and must be set after the config is created.
    #   enc_in / c_in : number of input channels — used by CATCHModel for RevIN
    #                   and the channel mask generator
    #   c_out          : number of output channels — same as input for reconstruction
    #   seq_len        : window length — already in params but we enforce consistency
    #   label_len      : required by some internal utilities; set to 48 (CATCH default)
    #   task_name      : read by some layers to switch behaviour
    config.enc_in    = n_vars
    config.c_in      = n_vars
    config.c_out     = n_vars
    config.seq_len   = seq_len
    config.label_len = 48
    config.task_name = 'anomaly_detection'

    model = CATCHModel(config)

    return model, config
