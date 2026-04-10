from typing import Dict

import numpy as np
import torch as th
import torchinfo
import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from EasyTSAD.DataFactory import MTSData
from EasyTSAD.Exptools import EarlyStoppingTorch
from EasyTSAD.Methods import BaseMethod


class KANADModel(nn.Module):
    def __init__(self, window: int, order: int, *args, **kwargs) -> None:
        super().__init__()
        self.window = window
        self.order = order
        self.channels = 2 * self.order + 1
        self.register_buffer(
            "orders",
            self._create_custom_periodic_cosine().unsqueeze(0),  # (1, order, window)
        )
        self.out_conv = nn.Conv1d(self.channels, 1, 1, bias=False)
        self.act = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.channels)
        self.bn3 = nn.BatchNorm1d(1)
        self.bn2 = nn.BatchNorm1d(self.channels)
        self.init_conv = nn.Conv1d(self.channels, self.channels, 3, 1, 1, bias=False)
        self.inner_conv = nn.Conv1d(self.channels, self.channels, 3, 1, 1, bias=False)
        self.final_conv = nn.Conv1d(1, 1, window, padding=0, stride=1, dilation=1)

    def forward(self, x: th.Tensor, *args, **kwargs) -> th.Tensor:
        res = []
        res.append(x.unsqueeze(1))
        ff = th.concat(
            [self.orders.repeat(x.size(0), 1, 1)]
            + [th.cos(order * x.unsqueeze(1)) for order in range(1, self.order + 1)]
            + [x.unsqueeze(1)],
            dim=1,
        )  # (batch, channels, window)
        res.append(ff)
        ff = self.init_conv(ff)
        ff = self.bn1(ff)
        ff = self.act(ff)
        ff = self.inner_conv(ff) + res.pop()
        ff = self.bn2(ff)
        ff = self.act(ff)
        ff = self.out_conv(ff) + res.pop()
        ff = self.bn3(ff)
        ff = self.act(ff)
        ff = self.final_conv(ff)
        return ff.squeeze(1)

    def _create_custom_periodic_cosine(self) -> th.Tensor:
        pl = [i for i in range(1, self.order + 1)]
        result = th.empty(self.order, self.window, dtype=th.float32)
        for i, p in enumerate(pl):
            range_value = th.arange(self.window, dtype=th.float32)
            result[i, :] = th.cos(2 * th.pi * range_value * p / self.window)
        return result


class KANAD(BaseMethod):
    def __init__(self, params: dict) -> None:
        super().__init__()
        self.__anomaly_score = None

        self.cuda = True
        if self.cuda and th.cuda.is_available():
            self.device = th.device("cuda")
            print("=== Using CUDA ===")
        else:
            if self.cuda and not th.cuda.is_available():
                print("=== CUDA is unavailable ===")
            self.device = th.device("cpu")
            print("=== Using CPU ===")

        self.batch_size = params["batch_size"]
        self.window = params["window"]
        self.model = KANADModel(**params).to(self.device)
        self.epochs = params["epochs"]
        self.optimizer = optim.Adam(self.model.parameters(), lr=params["lr"])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.75)
        self.loss = nn.MSELoss()
        self.early_stopping = EarlyStoppingTorch(save_path=None, patience=3)

    def _make_ci_tensors(self, data: np.ndarray):
        """
        Channel-independent reshape for MTS input (paper §MTS extension):
            (T, N) → windows (W, W, N) → reshape (num_windows*N, W)

        Each of the N channels is treated as an independent UTS instance.
        Returns X (num_windows*N, window), Y (num_windows*N,), num_windows, N.
        """
        T, N = data.shape
        num_windows = T - self.window
        if num_windows <= 0:
            return None, None, 0, N

        # Vectorised windowing
        idx = np.arange(num_windows)[:, None] + np.arange(self.window)[None, :]
        X = data[idx]                          # (num_windows, window, N)
        Y = data[self.window:]                 # (num_windows, N)

        # (num_windows, window, N) → (num_windows, N, window) → (num_windows*N, window)
        X_ci = X.transpose(0, 2, 1).reshape(num_windows * N, self.window)
        Y_ci = Y.reshape(num_windows * N)      # next-step target per (window, channel)

        return (
            th.tensor(X_ci, dtype=th.float32),
            th.tensor(Y_ci, dtype=th.float32),
            num_windows,
            N,
        )

    def _run_epochs(self, train_loader, valid_loader):
        for epoch in range(1, self.epochs + 1):
            self.model.train(mode=True)
            avg_loss = 0
            loop = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
            for idx, (x, target) in loop:
                x, target = x.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, target)
                loss.backward()
                self.optimizer.step()
                avg_loss += loss.cpu().item()
                loop.set_description(f"Training Epoch [{epoch}/{self.epochs}]")
                loop.set_postfix(loss=loss.item(), avg_loss=avg_loss / (idx + 1))

            self.model.eval()
            avg_loss = 0
            loop = tqdm.tqdm(enumerate(valid_loader), total=len(valid_loader), leave=True)
            with th.no_grad():
                for idx, (x, target) in loop:
                    x, target = x.to(self.device), target.to(self.device)
                    output = self.model(x)
                    loss = self.loss(output, target)
                    avg_loss += loss.cpu().item()
                    loop.set_description(f"Validation Epoch [{epoch}/{self.epochs}]")
                    loop.set_postfix(loss=loss.item(), avg_loss=avg_loss / (idx + 1))

            valid_loss = avg_loss / max(len(valid_loader), 1)
            self.scheduler.step()
            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                print("   Early stopping<<<")
                break

    def train_valid_phase(self, tsTrain: MTSData) -> None:
        data = tsTrain.train.astype(np.float32)          # (T, N)
        split = int(len(data) * 0.8)
        X_tr, Y_tr, _, _ = self._make_ci_tensors(data[:split])
        X_val, Y_val, _, _ = self._make_ci_tensors(data[split:])

        train_loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=self.batch_size, shuffle=False)
        self._run_epochs(train_loader, valid_loader)

    def train_valid_phase_all_in_one(self, tsTrains: Dict[str, MTSData]) -> None:
        all_train_data = np.concatenate([v.train.astype(np.float32) for v in tsTrains.values()], axis=0)
        split = int(len(all_train_data) * 0.8)
        X_tr, Y_tr, _, _ = self._make_ci_tensors(all_train_data[:split])
        X_val, Y_val, _, _ = self._make_ci_tensors(all_train_data[split:])

        train_loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=self.batch_size, shuffle=False)
        self._run_epochs(train_loader, valid_loader)

    def test_phase(self, tsData: MTSData) -> None:
        data = tsData.test.astype(np.float32)            # (T, N)
        X_ci, Y_ci, num_windows, N = self._make_ci_tensors(data)

        test_loader = DataLoader(
            TensorDataset(X_ci, Y_ci), batch_size=self.batch_size, shuffle=False
        )

        self.model.eval()
        scores = []
        loop = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), leave=True)
        with th.no_grad():
            for idx, (x, target) in loop:
                x, target = x.to(self.device), target.to(self.device)
                output = self.model(x)
                mse = th.sub(output, target).abs()
                scores.append(mse.cpu())
                loop.set_description("Testing:")

        # scores: (num_windows*N,) → (num_windows, N) → mean across channels → (num_windows,)
        scores_flat = th.cat(scores, dim=0)[..., -1].numpy().flatten()
        scores_flat[np.isnan(scores_flat)] = 1000
        scores_per_step = scores_flat.reshape(num_windows, N).mean(axis=1)

        # Pad front to align with full test length
        padded = np.zeros(len(data))
        padded[self.window:] = scores_per_step
        self.__anomaly_score = padded

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score

    def param_statistic(self, save_file):
        model_stats = torchinfo.summary(self.model, (self.batch_size, self.window), verbose=0)
        with open(save_file, "w") as f:
            f.write(str(model_stats))
