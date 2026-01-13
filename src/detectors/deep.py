from abc import abstractmethod, ABC
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from scipy import signal

from .tcn_ae import (
    TCNEncoder,
    TCNDecoder,
    TCN_AE,
    TCNAnomalyDetector,
    SlidingWindowDataset,
    StackingDataset,
)
from base import SubsequenceAnomalyDetector
from utils import (
    mahalanobis_anomaly_score,
    get_k_max_nonoverlapping,
    get_k_gaus_anomolus_windows_md,
)


class Loss(torch.nn.Module, ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pass


class LogCoshLoss(Loss):

    def __init__(self):
        super().__init__()

    @staticmethod
    def _logcosh(x):
        return x + F.softplus(-2.0 * x) - np.log(2)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        errors = self._logcosh(targets - preds)
        return torch.mean(errors)


class MSELoss(Loss):

    def __init__(self):
        super().__init__()
        self._loss = torch.nn.MSELoss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self._loss(preds, targets)


class TCNAEDetector(SubsequenceAnomalyDetector):

    def __init__(
        self,
        anom_len: int,
        ae_num_channels: list[int] | None = None,
        ae_enc_dilations: list[int] | None = None,
        ae_lr: float = 0.003,
        ae_n_epochs: int = 300,
        base_num_channels: list[int] | None = None,
        base_dilations_enc: list[int] | None = None,
        base_lr: float = 0.003,
        base_n_epochs: int = 100,
        batch_size: int = 64,
        train_window_len: int = 2048,
        kernel_size: int = 4,
        sampling_freq: int = 360,
        baseline_correct_bound: float = 1.0,
        loss_fun: Literal["mse", "logcosh"] = "logcosh",
        anom_windows_extraction_method: Literal["mahalanobis", "gauss"] = "mahalanobis",
        errors_baseline_correct: bool = True,
        verbose: bool = True,
    ):
        super().__init__(name="TCN_AE")
        self.ae_num_channels: list[int] = (
            ae_num_channels if ae_num_channels else [4, 8, 16, 32]
        )
        self.ae_enc_dilations: list[int] = (
            ae_enc_dilations
            if ae_enc_dilations
            else [int(np.power(2, i)) for i in range(len(self.ae_num_channels))]
        )
        self.ae_lr: float = ae_lr
        self.ae_n_epochs: int = ae_n_epochs
        self.base_num_channels: list[int] = (
            base_num_channels if base_num_channels else [4, 8, 16, 32]
        )
        self.base_dilations_enc: list[int] = (
            base_dilations_enc
            if base_dilations_enc
            else [int(np.power(2, i)) for i in range(len(self.base_num_channels))]
        )
        self.base_lr: float = base_lr
        self.base_n_epochs: int = base_n_epochs
        self.batch_size: int = batch_size
        self.anom_len: int = anom_len
        self.train_window_len: int = train_window_len
        self.kernel_size: int = kernel_size
        self.final_errors: np.ndarray | None = None
        self.recon_errors: np.ndarray | None = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._sampling_freq: int = sampling_freq
        self._baseline_correct_bound: float = baseline_correct_bound
        self.correct_sos = signal.butter(
            N=2,
            Wn=self._baseline_correct_bound,
            btype="highpass",
            fs=self._sampling_freq,
            output="sos",
        )
        if loss_fun not in ("mse", "logcosh"):
            raise ValueError(
                f'Possible values of `loss_fun` parameter: "mse", "logcosh", found: {loss_fun}'
            )
        self.loss_fun: Literal["mse", "logcosh"] = loss_fun
        if anom_windows_extraction_method not in ("gauss", "mahalanobis"):
            raise ValueError(
                f'Possible values of `anom_windows_extraction_method` parameter: "gauss", "mahalanobis", found: {anom_windows_extraction_method}'
            )
        self.anom_windows_extraction_method: Literal["mahalanobis", "gauss"] = (
            anom_windows_extraction_method
        )
        self.errors_baseline_correct: bool = errors_baseline_correct
        self.verbose = verbose

    def _baseline_correct(self, signal_to_correct: np.ndarray) -> np.ndarray:
        signal_corrected = signal.sosfilt(self.correct_sos, signal_to_correct)
        return signal_corrected

    def _log(self, msg) -> None:
        if self.verbose:
            print(msg)

    def _fit(self, X: np.ndarray) -> None:

        #### STEP 1. TRAIN TCN_AE ####

        # make dataset and dataloader from time series values
        dataset = SlidingWindowDataset(X, window_len=self.train_window_len)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        if self.loss_fun == "logcosh":
            loss = LogCoshLoss()
        elif self.loss_fun == "mse":
            loss = MSELoss()
        else:
            raise AssertionError

        # create autoencoder object and tools for training
        encoder = TCNEncoder(
            num_channels=self.ae_num_channels,
            dilations=self.ae_enc_dilations,
            kernel_size=self.kernel_size,
        ).to(self.device)
        decoder = TCNDecoder(
            num_channels=self.ae_num_channels,
            dilations=list(reversed(self.ae_enc_dilations)),
            kernel_size=self.kernel_size,
        ).to(self.device)
        autoencoder = TCN_AE(
            encoder=encoder, decoder=decoder, return_layers_outputs=True
        ).to(self.device)
        optim = Adam(autoencoder.parameters(), lr=self.ae_lr)

        # train AE
        self._log("Training AE ...")
        autoencoder.train()
        for i in range(self.ae_n_epochs):
            losses = []
            for x in dataloader:
                optim.zero_grad()
                x_recon, _ = autoencoder(x.to(self.device))
                loss_val = loss(
                    preds=x_recon.squeeze().cpu(), targets=x.squeeze().cpu()
                )
                loss_val.backward()
                optim.step()
                losses.append(loss_val.item())
            if i % 10 == 0:
                self._log(np.mean(losses))

        # pass input series to trained AE and obtain hidden layers outputs and reconstruction errors
        autoencoder.eval()
        X_recon, layers_outputs = autoencoder(
            torch.tensor(X, dtype=torch.float32).reshape((1, 1, -1)).to(self.device)
        )
        X_recon = X_recon.squeeze().cpu().detach().numpy()
        recon_errors = X_recon - X
        self.recon_errors = recon_errors

        #### STEP 2. TRAIN BASE TCN AE ON HIDEN LAYERS OUTPUTS AND RECONSTRUCTION ERRORS ####

        # make dataset and dataloader
        seq_len = layers_outputs[0].shape[-1]
        stacking_dataset = StackingDataset(
            layers_outputs=[
                output.reshape(-1, seq_len).detach() for output in layers_outputs
            ],
            recon_error=torch.tensor(recon_errors, dtype=torch.float32).reshape(1, -1),
            window_len=self.train_window_len,
        )
        stacking_dataloader = DataLoader(
            stacking_dataset, batch_size=self.batch_size, shuffle=True
        )

        # prepare base TCN autoencoder for anomaly detection and tools for training
        ad = TCNAnomalyDetector(
            num_inputs=len(layers_outputs) + 1,
            num_channels=self.base_num_channels,
            dilations_enc=self.base_dilations_enc,
            n_channels_convs=[t.shape[1] for t in layers_outputs],
            kernel_size=self.kernel_size,
        ).to(self.device)
        optim = Adam(ad.parameters(), lr=self.base_lr)

        # train base TCN autoencoder
        self._log("Training base TCN autoencoder ...")
        ad.train()
        for i in range(self.base_n_epochs):
            losses = []
            for layers_outputs_windows, recon_error_window in stacking_dataloader:
                optim.zero_grad()
                seq = ad.make_stacked_seq(
                    [lyr.to(self.device) for lyr in layers_outputs_windows],
                    recon_error_window.to(self.device),
                )
                batch_size = seq.shape[0]
                seq_recon = ad(seq)
                loss_val = loss(
                    preds=seq_recon.reshape((batch_size, -1)).cpu(),
                    targets=seq.reshape((batch_size, -1)).cpu(),
                )
                loss_val.backward()
                optim.step()
                losses.append(loss_val.item())
            if i % 10 == 0:
                self._log(np.mean(losses))

        # pass entire layers outputs and recontruction error and get final errors
        ad.eval()
        seq_entire = ad.make_stacked_seq(
            layers_outputs=[lyr.to(self.device) for lyr in layers_outputs],
            recon_error=torch.tensor(recon_errors, dtype=torch.float32)
            .reshape((1, 1, -1))
            .to(self.device),
        )
        seq_entire_recon = ad(seq_entire)
        self.final_errors = (
            (seq_entire_recon - seq_entire).detach().cpu().squeeze().numpy()
        )

        if self.errors_baseline_correct:
            self.final_errors = self._baseline_correct(self.final_errors)

    def _get_anom_inds_mahalanobis(self, k: int) -> list[int]:
        seq_len = self.final_errors.shape[-1]
        recon_error_windows = np.stack(
            [
                self.final_errors[:, i : (i + self.anom_len)].reshape((-1,))
                for i in range(seq_len - self.anom_len)
            ]
        )
        recon_error_anomaly_score = mahalanobis_anomaly_score(recon_error_windows)
        return get_k_max_nonoverlapping(
            inds_sorted=np.flip(np.argsort(recon_error_anomaly_score)).tolist(),
            window_len=self.anom_len,
            k=k,
        )

    def _get_anom_inds_gauss(self, k: int) -> list[int]:
        anoms_inds = get_k_gaus_anomolus_windows_md(
            self.final_errors, k=k, window_len=self.anom_len
        )
        return anoms_inds

    def _get_k_anoms(self, k: int) -> list[tuple[int, int]]:
        if self.anom_windows_extraction_method == "mahalanobis":
            anoms_inds = self._get_anom_inds_mahalanobis(k)
        elif self.anom_windows_extraction_method == "gauss":
            anoms_inds = self._get_anom_inds_gauss(k)
        else:
            raise AssertionError
        return [(anom_ind, anom_ind + self.anom_len) for anom_ind in anoms_inds]
