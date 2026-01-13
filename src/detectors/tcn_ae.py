import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from utils import nonoverlap_sliding_windows

from pytorch_tcn import TemporalConv1d, TCN


class TCNEncoder(nn.Module):

    def __init__(
        self,
        num_channels: list[int],
        dilations: list[int],
        kernel_size: int = 4,
        causal: bool = False,
    ):
        super().__init__()

        if len(num_channels) != len(dilations):
            raise ValueError("len of num_channels must be equal to len of dilations")

        self.kernel_size: int = kernel_size
        self.causal: bool = causal

        layers = [
            self.__create_tcn_layer(
                channels_in=1,
                channels_hidden=num_channels[0],
                channels_out=num_channels[0] // 4,
                dilation=dilations[0],
            )
        ]
        for channels_in, channels_hidden, dilation in zip(
            num_channels[:-1], num_channels[1:], dilations[1:]
        ):
            layers.append(
                self.__create_tcn_layer(
                    channels_in=channels_in // 4,
                    channels_hidden=channels_hidden,
                    channels_out=channels_hidden // 4,
                    dilation=dilation,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.end_conv = nn.Conv1d(
            in_channels=sum(n_channels // 4 for n_channels in num_channels),
            out_channels=2,
            kernel_size=1,
        )
        self.pool = nn.AvgPool1d(kernel_size=32, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list]:

        layers_outputs = []
        for layer in self.layers:
            x = layer.forward(x)
            layers_outputs.append(x.clone())

        x = torch.cat(layers_outputs, dim=(0 if len(x.shape) == 2 else 1))
        x = self.end_conv(x)
        return self.pool(x), layers_outputs

    def __create_tcn_layer(
        self, channels_in: int, channels_hidden: int, channels_out: int, dilation: int
    ) -> nn.Module:
        return nn.Sequential(
            TemporalConv1d(
                in_channels=channels_in,
                out_channels=channels_hidden,
                dilation=dilation,
                kernel_size=self.kernel_size,
                causal=self.causal,
            ),
            nn.Conv1d(
                in_channels=channels_hidden, out_channels=channels_out, kernel_size=1
            ),
        )


class TCNDecoder(nn.Module):

    def __init__(
        self,
        num_channels: list[int],
        dilations: list[int],
        kernel_size: int = 4,
        causal: bool = False,
    ):
        super().__init__()

        if len(num_channels) != len(dilations):
            raise ValueError("len of num_channels must be equal to len of dilations")

        self.kernel_size: int = kernel_size
        self.causal: bool = causal

        layers = [
            self.__create_tcn_layer(
                channels_in=2,
                channels_hidden=num_channels[0],
                channels_out=num_channels[0] // 4,
                dilation=dilations[0],
            )
        ]
        for channels_in, channels_hidden, dilation in zip(
            num_channels[:-1], num_channels[1:], dilations[1:]
        ):
            layers.append(
                self.__create_tcn_layer(
                    channels_in=channels_in // 4,
                    channels_hidden=channels_hidden,
                    channels_out=channels_hidden // 4,
                    dilation=dilation,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.end_conv = nn.Conv1d(
            in_channels=sum(n_channels // 4 for n_channels in num_channels),
            out_channels=1,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = nn.functional.interpolate(x, scale_factor=32, mode="nearest")

        layers_outputs = []
        for layer in self.layers:
            x = layer.forward(x)
            layers_outputs.append(x.clone())

        x = torch.cat(layers_outputs, dim=(0 if len(x.shape) == 2 else 1))
        return self.end_conv(x)

    def __create_tcn_layer(
        self, channels_in: int, channels_hidden: int, channels_out: int, dilation: int
    ) -> nn.Module:
        return nn.Sequential(
            TemporalConv1d(
                in_channels=channels_in,
                out_channels=channels_hidden,
                dilation=dilation,
                kernel_size=self.kernel_size,
                causal=self.causal,
            ),
            nn.Conv1d(
                in_channels=channels_hidden, out_channels=channels_out, kernel_size=1
            ),
        )


class TCN_AE(nn.Module):

    def __init__(
        self,
        encoder: TCNEncoder,
        decoder: TCNDecoder,
        return_layers_outputs: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.return_layers_outputs: bool = return_layers_outputs

    def encode_(self, x):
        return self.encoder(x)

    def decode_(self, z):
        return self.decoder(z)

    def forward(self, x):
        output_seq_size = x.shape[-1]
        z, layers_outputs = self.encode_(x)
        if self.return_layers_outputs:
            return self.decode_(z)[:, :, :output_seq_size], layers_outputs
        return self.decode_(z)[:, :, :output_seq_size]


class TCNBaseAE(nn.Module):

    def __init__(
        self,
        num_inputs: int,
        num_channels: list[int],
        dilations_enc: list[int],
        kernel_size: int = 4,
        use_skip_connections: bool = False,
    ):
        super().__init__()

        # encoder
        self.tcn1 = TCN(
            num_inputs=num_inputs,
            num_channels=num_channels,
            dilations=dilations_enc,
            kernel_size=kernel_size,
            use_skip_connections=use_skip_connections,
            causal=False,
        )
        self.conv1 = nn.Conv1d(
            kernel_size=1, in_channels=num_channels[-1], out_channels=2
        )
        self.pool = nn.MaxPool1d(32, ceil_mode=True)

        # decoder

        self.upsample = nn.Upsample(scale_factor=32)
        self.tcn2 = TCN(
            num_inputs=2,
            num_channels=num_channels,
            dilations=list(reversed(dilations_enc)),
            kernel_size=kernel_size,
            use_skip_connections=use_skip_connections,
            causal=False,
        )
        self.conv2 = nn.Conv1d(
            kernel_size=1, in_channels=num_channels[-1], out_channels=num_inputs
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tcn1(x)
        x = self.conv1(x)
        return self.pool(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(x, scale_factor=32, mode="nearest")
        x = self.tcn2(x)
        return self.conv2(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_size = x.shape[-1]
        z = self.encode(x)
        return self.decode(z)[:, :, :seq_size]


class SlidingWindowDataset(Dataset):

    def __init__(self, ts_values: np.ndarray, window_len: int):
        self.windows = nonoverlap_sliding_windows(ts_values, window_len)

    def __len__(self):
        return self.windows.shape[0]

    def __getitem__(self, idx):
        window = self.windows[idx, :]
        return torch.tensor(window, dtype=torch.float32).reshape((1, -1))


class StackingDataset(Dataset):

    def __init__(
        self,
        layers_outputs: list[torch.Tensor],
        recon_error: torch.Tensor,
        window_len: int,
    ):
        """
        Args:
            layers_outputs: list of outputs of subsequent encoder's layers,
                each of shape (n_channels, seq_len) (2D tensors)
            recon_error: reconstruction error, 2D tensor of shape (n_channels, seq_len)
            window_len: len of nonoverlapping sliding window
        """
        seq_len = recon_error.shape[-1]
        self.windows_inds = torch.tensor(
            nonoverlap_sliding_windows(np.arange(seq_len), window_len=window_len)
        )
        self.layers_outputs = layers_outputs
        self.recon_error = recon_error

    def __len__(self):
        return self.windows_inds.shape[0]

    def __getitem__(self, idx) -> tuple[list[torch.Tensor], torch.Tensor]:
        inds = self.windows_inds[idx]
        layers_outputs_windows = [
            layer_output[:, inds] for layer_output in self.layers_outputs
        ]
        recon_error_window = self.recon_error[:, inds]
        return layers_outputs_windows, recon_error_window


class TCNAnomalyDetector(nn.Module):

    def __init__(
        self, num_inputs, num_channels, dilations_enc, n_channels_convs, kernel_size=4
    ):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(kernel_size=1, in_channels=ch_size, out_channels=1)
                for ch_size in n_channels_convs
            ]
        )
        self.tcn = TCNBaseAE(
            num_inputs=num_inputs,
            num_channels=num_channels,
            dilations_enc=dilations_enc,
            kernel_size=kernel_size,
        )

    def make_stacked_seq(
        self, layers_outputs: list[torch.Tensor], recon_error: torch.Tensor
    ) -> torch.Tensor:
        layers_conved = [
            conv(layer_output) for conv, layer_output in zip(self.convs, layers_outputs)
        ]
        return torch.cat(layers_conved + [recon_error], dim=1)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        return self.tcn.forward(seq)
