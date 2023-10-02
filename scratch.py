from typing import Union
from math import ceil, floor
from einops.layers.torch import Rearrange
from torch import nn
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from plots import plot_reconstructed_ecg

class EEG2ECGModel(pl.LightningModule):
    def __init__(
        self,
        seconds: Union[int, float],
        eeg_channels: int,
        ecg_channels: int,
        eeg_sampling_rate: int,
        ecg_sampling_rate: int,
        encoder_layers: int = 4,
        kernel_size: int = 7,
    ):
        super(EEG2ECGModel, self).__init__()

        self.seconds = seconds

        self.eeg_channels = eeg_channels
        self.eeg_sampling_rate = eeg_sampling_rate
        self.eeg_samples = self.eeg_sampling_rate * ceil(self.seconds)

        self.ecg_channels = ecg_channels
        self.ecg_sampling_rate = ecg_sampling_rate
        self.ecg_samples = self.ecg_sampling_rate * ceil(self.seconds)

        self.kernel_size = kernel_size
        self.padding = floor(kernel_size / 2)

        self.encoder_layers = encoder_layers
        self.eeg_encoded_samples = ceil(self.eeg_samples * 2**-self.encoder_layers)
        self.s_dim = 64
        self.h_dim = self.s_dim * 2**self.encoder_layers
        self.eegs_encoder = self.build_encoder(channels=self.eeg_channels, s_dim=self.s_dim, layers=self.encoder_layers)

        self.decoder_layers = encoder_layers
        self.ecg_encoded_samples = ceil(self.ecg_samples * 2**-self.encoder_layers)
        self.ecgs_decoder = [
            nn.Sequential(
                Rearrange("b (c t) -> b c t", t=1),
                nn.ConvTranspose1d(
                    in_channels=self.h_dim,
                    out_channels=self.h_dim,
                    kernel_size=self.ecg_encoded_samples,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm1d(self.h_dim),
                nn.LeakyReLU(),
            )
        ]
        for i_layer in range(self.decoder_layers):
            layer = nn.Sequential(
                nn.ConvTranspose1d(
                    in_channels=ceil(self.h_dim * 2**-i_layer),
                    out_channels=ceil(self.h_dim * 2 ** -(i_layer + 1)),
                    kernel_size=self.kernel_size,
                    stride=2,
                    padding=self.padding,
                    output_padding=1,
                    bias=False,
                ),
                nn.BatchNorm1d(ceil(self.h_dim * 2 ** -(i_layer + 1))),
                nn.LeakyReLU(),
            )
            self.ecgs_decoder.append(layer)
        self.ecgs_decoder.append(
            nn.Conv1d(
                in_channels=self.s_dim,
                out_channels=self.ecg_channels,
                kernel_size=7,
                stride=1,
                padding=3,
            )
        )
        self.ecgs_decoder = nn.ModuleList(self.ecgs_decoder)

    def build_encoder(self, in_channels, layers, s_dim, kernel_size=7):
        padding = floor(kernel_size / 2)
        encoder = [
            nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=s_dim,
                    kernel_size=7,
                    stride=1,
                    padding=3,
                    bias=False,
                ),
                nn.BatchNorm1d(s_dim),
                nn.LeakyReLU(),
            )
        ]
        for i_layer in range(layers):
            norm = nn.BatchNorm1d(self.s_dim * 2 ** (i_layer + 1)) if i_layer < layers - 1 else nn.Identity()
            activation = nn.LeakyReLU() if i_layer < layers - 1 else nn.Identity()
            layer = nn.Sequential(
                nn.Conv1d(
                    in_channels=s_dim * 2**i_layer,
                    out_channels=s_dim * 2 ** (i_layer + 1),
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                    bias=i_layer >= layers - 1,
                ),
                norm, 
                activation
            )
            encoder.append(layer)
        encoder.append(
            nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                Rearrange("b c t -> b (c t)"),
            )
        )
        encoder = nn.ModuleList(encoder)
        return encoder
    
    def forward(self, x):
        pass

    def encode_eegs(self, eegs):
        eegs_encoded = eegs
        for block in self.eegs_encoder:
            eegs_encoded = block(eegs_encoded)
        assert len(eegs_encoded.shape) == 2
        assert eegs_encoded.shape[1] == self.h_dim
        return eegs_encoded

    def decode_ecgs(self, eegs_encoded):
        assert len(eegs_encoded.shape) == 2
        assert eegs_encoded.shape[1] == self.h_dim
        ecgs_encoded = eegs_encoded
        for block in self.ecgs_decoder:
            ecgs_encoded = block(ecgs_encoded)
        assert len(ecgs_encoded.shape) == 3
        assert ecgs_encoded.shape[1:] == (
            self.ecg_channels,
            self.ecg_samples,
        ), f"expected {(self.ecg_channels, self.ecg_samples)}, got {ecgs_encoded.shape[1:]}"
        return ecgs_encoded

    def shared_step(self, batch, batch_idx):
        eegs = batch["eegs"].to(self.device)
        ecgs_gt = batch["ecgs"].to(self.device)
        eeg_encoded = self.encode_eegs(eegs)
        ecgs_pred = self.decode_ecgs(eeg_encoded)

        if not self.training and self.current_epoch >= 1 and batch_idx == 0:
            plot_reconstructed_ecg(eegs[0], ecgs_gt[0], ecgs_pred[0], path=f"images/epoch={self.current_epoch}.png")

        outs = {"loss": F.l1_loss(ecgs_pred, ecgs_gt)}
        for k, v in outs.items():
            self.log(k, v, prog_bar=True, on_step=False, on_epoch=True)
        return outs

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)

model = EEG2ECGModel(
    eeg_channels=32, ecg_channels=2,
    eeg_sampling_rate=128, ecg_sampling_rate=256,
    seconds=2
)