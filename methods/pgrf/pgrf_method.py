import numpy as np
import torch
import torchinfo

from EasyTSAD.Methods import BaseMethod
from EasyTSAD.DataFactory import MTSData

from methods.pgrf.model import PGRFNet
from methods.pgrf.training import train_model_stage1, train_model_stage2
from methods.pgrf.inference import infer_scores
from methods.pgrf.utils import create_windows


FIXED_ALPHA = 0.1  # weight of explanatory scores vs predictive score


def _compute_combined_score(scores_dict: dict, alpha: float) -> np.ndarray:
    s_pred = scores_dict['predictive_scores']
    s_structural = scores_dict['structural_scores']
    s_ctx = scores_dict['contextual_scores']
    s_spike = scores_dict['spike_scores']

    expl = np.stack([s_structural, s_ctx, s_spike], axis=-1)
    expl_sum = expl.sum(axis=1, keepdims=True) + 1e-8
    expl_norm = expl / expl_sum
    expl_score = np.sum(expl_norm * expl, axis=1)

    return (1 - alpha) * s_pred + alpha * expl_score


class PGRF(BaseMethod):
    def __init__(self, params: dict) -> None:
        super().__init__()
        self.__anomaly_score = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.window_size = params.get('window_size', 60)
        self.num_protos = params.get('num_protos', 10)
        self.alpha = params.get('alpha', FIXED_ALPHA)

        # All remaining params forwarded to train_model_stage1/2
        _skip = {'window_size', 'num_protos', 'alpha'}
        self.train_params = {k: v for k, v in params.items() if k not in _skip}

        # Model is built lazily in train_valid_phase once num_vars is known
        self.model = None

    def _build_model(self, num_vars: int) -> None:
        self.model = PGRFNet(
            num_vars=num_vars,
            seq_len=self.window_size,
            num_protos=self.num_protos,
            num_context_protos=self.num_protos,
            num_spike_protos=self.num_protos,
        )
        if self.device == 'cuda':
            self.model.cuda()

    def train_valid_phase(self, tsTrain: MTSData) -> None:
        train_data = tsTrain.train.astype(np.float32)   # (T, N)
        train_labels = np.zeros(len(train_data), dtype=np.float32)

        X, Y, L = create_windows(train_data, train_labels, self.window_size)
        if X.shape[0] == 0:
            return

        self._build_model(num_vars=train_data.shape[1])
        train_model_stage1(self.model, X, Y, L, **self.train_params)
        train_model_stage2(self.model, X, Y, L, **self.train_params)

    def test_phase(self, tsData: MTSData) -> None:
        if self.model is None:
            self.__anomaly_score = np.zeros(len(tsData.test))
            return

        scores_dict = infer_scores(self.model, tsData.test.astype(np.float32), self.window_size)

        if not scores_dict:
            self.__anomaly_score = np.zeros(len(tsData.test))
            return

        self.__anomaly_score = _compute_combined_score(scores_dict, self.alpha)

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score

    def param_statistic(self, save_file: str) -> None:
        # Model is not yet built when EasyTSAD calls this (before train_valid_phase).
        # Write a placeholder; the real summary would require num_vars.
        with open(save_file, 'w') as f:
            f.write("PGRF-Net — model summary unavailable before training (num_vars is data-dependent).\n")
