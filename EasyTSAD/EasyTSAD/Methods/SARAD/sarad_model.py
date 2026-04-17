import numpy as np
import os
from .. import BaseMethod
from ...DataFactory import TSData
from pipelines.sarad import run_sarad_pipeline

class SARAD(BaseMethod):
    def __init__(self, params, cuda) -> None:
        super().__init__()
        self.params = params
        self.cuda = cuda
        self.__anomaly_score = None

        # must match what you set in SARAD config
        self.output_dir = "outputs/sarad_run"

    # -------------------------------
    # TRAIN
    # -------------------------------
    def train_valid_phase(self, tsTrain: TSData):
        run_sarad_pipeline(self.params)

    # -------------------------------
    # TEST
    # -------------------------------
    def test_phase(self):
        score_path = os.path.join(self.output_dir, "scores.npy")

        if not os.path.exists(score_path):
            raise RuntimeError(f"scores.npy not found at {score_path}")

        scores = np.load(score_path)

        test_len = len(self.tsData.test)
        full_scores = np.zeros(test_len)

        full_scores[-len(scores):] = scores

        self.__anomaly_score = full_scores

    # -------------------------------
    # RETURN SCORES
    # -------------------------------
    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score