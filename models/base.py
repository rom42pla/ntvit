from typing import Union
from math import ceil, floor, log2
from einops.layers.torch import Rearrange
from torch import nn
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.autograd.profiler as profiler
import torchaudio
import wandb

try:
    from plots import (
        plot_reconstructed_ecgs_waveforms,
        plot_reconstructed_spectrograms,
    )
    from models.sdtw import SoftDTW
except:
    print("error loading libraries")


class EEG2ECGModel(pl.LightningModule):
    def __init__(
        self,
        seconds: Union[int, float],
        eeg_channels: int,
        ecg_channels: int,
        eeg_sampling_rate: int,
        ecg_sampling_rate: int,
        spectrogram_scale: int = 8,
        n_mels: int = 16,
        encoder_layers: int = 3,
        kernel_size: int = 5,
        learning_rate: float = 5e-5,
        use_vocoder: bool = False,
        activation_fn: str = "leaky_relu",
        norm_fn: str = "instance",
    ):
        super(EEG2ECGModel, self).__init__()

        self.automatic_optimization = False
        self.learning_rate = learning_rate

        self.seconds = seconds
        self.encoder_layers = encoder_layers

        self.spectrogram_n_fft = 128
        assert (
            self.spectrogram_n_fft & (self.spectrogram_n_fft - 1)
        ) == 0, "spectrogram_n_fft must be a power of 2"
        # self.spectrogram_frequencies = (self.spectrogram_n_fft // 2) + 1
        self.spectrogram_frequencies = n_mels
        self.spectrogram_latent_frequencies = ceil(
            self.spectrogram_frequencies / 2**self.encoder_layers
        )
        self.spectrogram_scale = spectrogram_scale
        self.spectrogram_kernel_size = self.spectrogram_scale * 2 + 1
        self.spectrogram_kernel_stride = self.spectrogram_scale
        # self.spectrogram_kernel_size = self.spectrogram_n_fft // 2
        # self.spectrogram_kernel_stride = self.spectrogram_kernel_size // 2

        self.eeg_channels = eeg_channels
        self.eeg_sampling_rate = eeg_sampling_rate
        # formula taken from https://stackoverflow.com/questions/42821425/define-an-interval-for-frequency-of-stft-outputs
        self.eeg_min_frequency, self.eeg_max_frequency = 4, 45
        self.eeg_frequency_resolution: float = (
            self.eeg_sampling_rate / self.spectrogram_n_fft
        )
        self.eeg_bins = min(
            self.spectrogram_frequencies,
            int(1 + floor(self.eeg_max_frequency / self.eeg_frequency_resolution)),
        )
        self.eeg_samples = self.eeg_sampling_rate * ceil(self.seconds)
        self.eeg_spectrogram_samples = (
            ceil(self.eeg_samples / self.spectrogram_kernel_stride) + 1
        )
        self.eeg_spectrogram_latent_samples = ceil(
            self.eeg_spectrogram_samples / 2**self.encoder_layers
        )
        self.eeg_mel_spectrogrammer = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.eeg_sampling_rate,
            n_fft=self.spectrogram_n_fft,
            win_length=self.spectrogram_kernel_size,
            hop_length=self.spectrogram_kernel_stride,
            f_min=self.eeg_min_frequency,
            f_max=self.eeg_max_frequency,
            n_mels=self.spectrogram_frequencies,
            power=1,
            normalized=True,
        )

        self.ecg_channels = ecg_channels
        self.ecg_sampling_rate = ecg_sampling_rate
        # formula taken from https://stackoverflow.com/questions/42821425/define-an-interval-for-frequency-of-stft-outputs
        self.ecg_min_frequency, self.ecg_max_frequency = 0, 60
        self.ecg_frequency_resolution: float = (
            self.ecg_sampling_rate / self.spectrogram_n_fft
        )
        self.ecg_bins = max(
            self.spectrogram_frequencies,
            int(1 + floor(self.ecg_max_frequency / self.ecg_frequency_resolution)),
        )
        self.ecg_samples = self.ecg_sampling_rate * ceil(self.seconds)
        self.ecg_spectrogram_samples = (
            ceil(self.ecg_samples / self.spectrogram_kernel_stride) + 1
        )
        self.ecg_spectrogram_latent_samples = ceil(
            self.ecg_spectrogram_samples / 2**self.encoder_layers
        )
        self.ecg_mel_spectrogrammer = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.ecg_sampling_rate,
            n_fft=self.spectrogram_n_fft,
            win_length=self.spectrogram_kernel_size,
            hop_length=self.spectrogram_kernel_stride,
            f_min=self.ecg_min_frequency,
            f_max=self.ecg_max_frequency,
            n_mels=self.spectrogram_frequencies,
            power=1,
            normalized=True,
        )

        # self.ecg_spectrogram_n_fft = 128
        # assert (
        #     self.ecg_spectrogram_n_fft & (self.ecg_spectrogram_n_fft - 1)
        # ) == 0, f"ecg_spectrogram_n_fft must be a power of 2"
        # self.eec_spectrogram_frequencies = (self.ecg_spectrogram_n_fft // 2) + 1

        self.kernel_size = kernel_size
        self.padding = floor(kernel_size / 2)

        # self.eeg_encoded_samples = ceil(self.eeg_samples * 2**-self.encoder_layers)
        # self.ecg_encoded_samples = ceil(self.ecg_samples * 2**-self.encoder_layers)
        self.s_dim = 64
        self.h_dim = self.s_dim * 2**self.encoder_layers

        ####################################
        # MODULES
        ####################################
        if activation_fn == "gelu":
            self.activation_fn = nn.GELU()
        elif activation_fn == "leaky_relu":
            self.activation_fn = nn.LeakyReLU()
        if norm_fn == "instance":
            self.norm_fn = nn.InstanceNorm2d
        elif norm_fn == "batch":
            self.norm_fn = nn.BatchNorm2d
        elif norm_fn in {None, "none"}:
            self.norm_fn = None
        self.eegs_encoder = self.build_encoder(
            in_channels=self.eeg_channels,
            s_dim=self.s_dim,
            layers=self.encoder_layers,
            kernel_size=self.kernel_size,
            limit_outputs=False,
            dropout=0.1,
        )
        self.ecgs_encoder = self.build_encoder(
            in_channels=self.ecg_channels,
            s_dim=self.s_dim,
            layers=self.encoder_layers,
            kernel_size=self.kernel_size,
            limit_outputs=True,
            dropout=0.1,
        )
        self.upsampler = nn.Sequential(
            nn.Upsample(
                size=(
                    self.spectrogram_latent_frequencies,
                    self.ecg_spectrogram_latent_samples,
                ),
                mode="bilinear",
            )
            if self.eeg_spectrogram_latent_samples
            != self.ecg_spectrogram_latent_samples
            else nn.Identity(),
        )
        self.reasoner = self.build_reasoner(
            channels=self.h_dim, frequencies=self.spectrogram_latent_frequencies,
        )
        # self.reasoner = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=self.h_dim,
        #         out_channels=self.s_dim,
        #         kernel_size=1,
        #         stride=1,
        #         padding=0,
        #         bias=False,
        #     ),
        #     self.norm_fn(self.s_dim),
        #     self.activation_fn,
        #     nn.Dropout(0.1),
        #     nn.Conv2d(
        #         in_channels=self.s_dim,
        #         out_channels=self.h_dim,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #         bias=False,
        #     ),
        #     self.norm_fn(self.h_dim),
        #     self.activation_fn,
        #     nn.Dropout(0.1),
        #     nn.Conv2d(
        #         in_channels=self.h_dim,
        #         out_channels=self.h_dim,
        #         kernel_size=1,
        #         stride=1,
        #         padding=0,
        #         bias=False,
        #     ),
        #     self.norm_fn(self.h_dim),
        #     nn.Dropout(0.1),
        #     # nn.Tanh(),
        #     # Rearrange("b c h w -> b (c h w)"),
        #     # nn.Linear(
        #     #     self.h_dim
        #     #     * self.spectrogram_latent_frequencies
        #     #     * self.eeg_spectrogram_latent_samples,
        #     #     self.h_dim,
        #     # ),
        #     # nn.GELU(),
        #     # nn.Linear(
        #     #     self.h_dim,
        #     #     self.h_dim
        #     #     * self.spectrogram_latent_frequencies
        #     #     * self.ecg_spectrogram_latent_samples,
        #     # ),
        #     # Rearrange(
        #     #     "b (c h w) -> b c h w",
        #     #     c=self.h_dim,
        #     #     h=self.spectrogram_latent_frequencies,
        #     #     w=self.ecg_spectrogram_latent_samples,
        #     # ),
        # )

        self.ecgs_decoder = self.build_decoder(
            out_channels=self.ecg_channels,
            s_dim=self.s_dim,
            h_dim=self.h_dim,
            layers=self.encoder_layers,
            kernel_size=self.kernel_size,
        )
        self.eegs_decoder = self.build_decoder(
            out_channels=self.eeg_channels,
            s_dim=self.s_dim,
            h_dim=self.h_dim,
            layers=self.encoder_layers,
            kernel_size=self.kernel_size,
        )

        ####################################
        # MODULES
        # VOCODER
        ####################################
        assert isinstance(use_vocoder, bool)
        self.use_vocoder = use_vocoder
        if self.use_vocoder:
            self.ecgs_vocoder = self.build_vocoder(
                channels=self.ecg_channels,
                frequencies=self.spectrogram_frequencies,
                s_dim=self.s_dim,
                h_dim=self.h_dim,
                scale=self.spectrogram_scale,
                kernel_size=self.kernel_size,
            )

        ####################################
        # MODULES
        # DISCRIMINATOR
        ####################################
        # self.discriminator_encoder = self.build_encoder(
        #     in_channels=self.ecg_channels,
        #     s_dim=self.s_dim,
        #     layers=self.encoder_layers,
        #     kernel_size=self.kernel_size,
        # )
        # self.discriminator_cls = nn.Sequential(
        #     Rearrange("b c h w -> b (c h w)"),
        #     nn.Linear(
        #         self.h_dim
        #         * self.spectrogram_latent_frequencies
        #         * self.ecg_spectrogram_latent_samples,
        #         1,
        #     ),
        # )
        self.discriminator_latent = nn.Sequential(
            Rearrange("b c h w -> b (c h w)"),
            nn.Linear(
                self.h_dim
                * self.spectrogram_latent_frequencies
                * self.ecg_spectrogram_latent_samples,
                1,
            ),
        )

        self.save_hyperparameters(
            ignore=["eeg_mel_spectrogrammer", "ecg_mel_spectrogrammer"]
        )

    def configure_optimizers(self):
        optimizer_encoders = torch.optim.AdamW(
            [
                p
                for module in [
                    self.eegs_decoder,
                    self.ecgs_encoder,
                    self.ecgs_decoder,
                    self.reasoner,
                    self.upsampler,
                ]
                for p in module.parameters()
            ],
            lr=self.learning_rate,
        )
        optimizer_decoders = torch.optim.AdamW(
            [
                p
                for module in [
                    self.ecgs_encoder,
                    self.ecgs_decoder,
                ]
                for p in module.parameters()
            ],
            lr=self.learning_rate,
        )
        if not self.use_vocoder:
            return (
                optimizer_encoders,
                optimizer_decoders,
            )
        else:
            optimizer_vocoder = torch.optim.AdamW(
                [
                    p
                    for module in [
                        self.ecgs_vocoder,
                    ]
                    for p in module.parameters()
                ],
                lr=self.learning_rate,
            )
            return (
                optimizer_encoders,
                optimizer_decoders,
                optimizer_vocoder,
            )

    def build_encoder(
        self,
        in_channels,
        layers,
        s_dim,
        kernel_size,
        limit_outputs=True,
        dropout: float = 0.0,
    ):
        assert 0 <= dropout < 1
        padding = floor(kernel_size / 2)
        return nn.ModuleDict(
            {
                "pre": nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=s_dim,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding,
                        bias=False if self.norm_fn else True,
                    ),
                    self.norm_fn(s_dim) if self.norm_fn else nn.Identity(),
                    self.activation_fn,
                ),
                "downs": nn.Sequential(
                    *[
                        nn.Sequential(
                            nn.Conv2d(
                                in_channels=s_dim * 2**i_layer,
                                out_channels=s_dim * 2 ** (i_layer + 1),
                                kernel_size=kernel_size,
                                stride=2,
                                padding=padding,
                                bias=False if self.norm_fn else True,
                            ),
                            self.norm_fn(s_dim * 2 ** (i_layer + 1))
                            if self.norm_fn
                            else nn.Identity(),
                            self.activation_fn,
                            nn.Conv2d(
                                in_channels=ceil(s_dim * 2 ** (i_layer + 1)),
                                out_channels=ceil(s_dim * 2 ** (i_layer + 1)),
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False if self.norm_fn else True,
                            ),
                            self.norm_fn(ceil(s_dim * 2 ** (i_layer + 1)))
                            if self.norm_fn
                            else nn.Identity(),
                            self.activation_fn,
                        )
                        for i_layer in range(layers)
                    ]
                ),
                "post": nn.Sequential(
                    # nn.Tanh() if limit_outputs else nn.Identity(),
                    nn.Dropout(dropout),
                    # nn.Identity(),
                    # nn.Conv2d(
                    #     in_channels=h_dim,
                    #     out_channels=h_dim,
                    #     kernel_size=(latent_frequencies, latent_samples),
                    #     stride=1,
                    #     padding=0,
                    #     bias=True,
                    # ),
                    # Rearrange("b c h w -> b (c h w)"),
                    # nn.BatchNorm1d(h_dim),
                    # nn.ELU(),
                    # nn.Linear(h_dim, h_dim * 4),
                    # nn.BatchNorm1d(h_dim * 4),
                    # nn.ELU(),
                    # nn.Linear(h_dim * 4, h_dim),
                ),
            }
        )

    def build_decoder(
        self,
        out_channels,
        # latent_frequencies,
        # latent_samples,
        # input_samples,
        layers,
        s_dim,
        h_dim,
        kernel_size=7,
    ):
        padding = floor(kernel_size / 2)
        return nn.ModuleDict(
            {
                "pre": nn.Sequential(
                    nn.Conv2d(
                        in_channels=h_dim,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    self.norm_fn(h_dim) if self.norm_fn else nn.Identity(),
                    self.activation_fn,
                    # nn.Identity()
                    # Rearrange("b c h w -> b (c h w)"),
                    # nn.Linear(
                    #     in_features=h_dim * latent_frequencies * input_samples,
                    #     out_features=h_dim,
                    # ),
                    # nn.GELU(),
                    # nn.Linear(
                    #     in_features=h_dim,
                    #     out_features=h_dim * latent_frequencies * latent_samples,
                    # ),
                    # Rearrange(
                    #     "b (c h w) -> b c h w",
                    #     c=h_dim,
                    #     h=latent_frequencies,
                    #     w=latent_samples,
                    # ),
                    # # Rearrange("b (c h w) -> b c h w", h=1, w=1),
                    # # nn.ConvTranspose2d(
                    # #     in_channels=h_dim,
                    # #     out_channels=h_dim,
                    # #     kernel_size=(latent_frequencies, latent_samples),
                    # #     stride=1,
                    # #     padding=0,
                    # #     bias=False,
                    # # ),
                    # # nn.BatchNorm2d(h_dim),
                    # nn.InstanceNorm2d(h_dim),
                    # nn.GELU(),
                ),
                "ups": nn.Sequential(
                    *[
                        nn.Sequential(
                            nn.ConvTranspose2d(
                                in_channels=ceil(h_dim * 2**-i_layer),
                                out_channels=ceil(h_dim * 2 ** -(i_layer + 1)),
                                kernel_size=kernel_size,
                                stride=2,
                                padding=padding,
                                output_padding=(1, 0),
                                bias=False if self.norm_fn else True,
                            ),
                            self.norm_fn(ceil(h_dim * 2 ** -(i_layer + 1)))
                            if self.norm_fn
                            else nn.Identity(),
                            self.activation_fn,
                            nn.Conv2d(
                                in_channels=ceil(h_dim * 2 ** -(i_layer + 1)),
                                out_channels=ceil(h_dim * 2 ** -(i_layer + 1)),
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False if self.norm_fn else True,
                            ),
                            # nn.BatchNorm2d(ceil(h_dim * 2 ** -(i_layer + 1))),
                            self.norm_fn(ceil(h_dim * 2 ** -(i_layer + 1)))
                            if self.norm_fn
                            else nn.Identity(),
                            self.activation_fn,
                        )
                        for i_layer in range(layers)
                    ]
                ),
                "post": nn.Sequential(
                    nn.Conv2d(
                        in_channels=s_dim,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True,
                    ),
                    nn.Softplus(),
                ),
            }
        )

    def build_vocoder(
        self,
        channels,
        frequencies,
        scale,
        s_dim,
        h_dim,
        kernel_size=7,
    ):
        padding = floor(kernel_size / 2)
        layers = ceil(log2(scale))
        latent_frequencies = ceil(frequencies / 2**layers)
        h_dim = s_dim * 2**layers

        return nn.ModuleDict(
            {
                "pre": nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=channels,
                        out_channels=s_dim,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding,
                        bias=False if self.norm_fn else True,
                    ),
                    # nn.BatchNorm2d(s_dim),
                    self.norm_fn(s_dim) if self.norm_fn else nn.Identity(),
                    self.activation_fn,
                ),
                "downs": nn.Sequential(
                    *[
                        nn.Sequential(
                            nn.Conv2d(
                                in_channels=s_dim * 2**i_layer,
                                out_channels=s_dim * 2 ** (i_layer + 1),
                                kernel_size=kernel_size,
                                stride=(2, 1),
                                padding=padding,
                                bias=False if self.norm_fn else True,
                            ),
                            self.activation_fn,
                            nn.ConvTranspose2d(
                                in_channels=s_dim * 2 ** (i_layer + 1),
                                out_channels=s_dim * 2 ** (i_layer + 1),
                                kernel_size=kernel_size,
                                stride=(1, 2),
                                padding=padding,
                                output_padding=(0, 1),
                                bias=False if self.norm_fn else True,
                            ),
                            # nn.BatchNorm2d(s_dim * 2 ** (i_layer + 1)),
                            self.norm_fn(s_dim * 2 ** (i_layer + 1))
                            if self.norm_fn
                            else nn.Identity(),
                            self.activation_fn,
                        )
                        for i_layer in range(layers)
                    ]
                ),
                "post": nn.Sequential(
                    nn.Conv2d(
                        in_channels=h_dim,
                        out_channels=channels,
                        kernel_size=(latent_frequencies, kernel_size),
                        stride=1,
                        padding=(0, padding),
                        bias=True,
                    ),
                    Rearrange("b c h w -> b (c h) w"),
                    # nn.Tanh(),
                ),
            }
        )

    def build_reasoner(
        self,
        channels,
        frequencies,
        h_dim = 768,
    ):
        return nn.ModuleDict(
            {
                "in_reshaper": nn.Sequential(
                    Rearrange("b c h w -> b w (c h)", c=channels, h=frequencies),
                    nn.Linear(channels * frequencies, h_dim),
                ),
                "encoder": nn.TransformerEncoder(
                    encoder_layer=nn.TransformerEncoderLayer(
                        d_model=h_dim,
                        nhead=8,
                        dim_feedforward=h_dim * 4,
                        activation=F.leaky_relu,
                        batch_first=True,
                    ),
                    num_layers=4,
                ),
                "out_reshaper": nn.Sequential(
                    nn.Linear(h_dim, channels * frequencies),
                    Rearrange("b w (c h) -> b c h w", c=channels, h=frequencies)
                ),
                "pos_embeddings": nn.Embedding(num_embeddings=2048, embedding_dim=h_dim),
            }
        )

    def forward(self, x):
        pass

    def encode_mel_spectrogram(self, mel_spectrogram, encoder, expected_shape=None):
        x = torch.log1p(mel_spectrogram)
        # pre convolution for reshaping and channel matching
        x = encoder.pre(x)
        # downscaling
        for down_block in encoder.downs:
            # print("pre", x.shape)
            x = down_block(x)
            # print("post", x.shape)
        # post convolution for channel matching
        latent = encoder.post(x)
        # OPTIONAL: does a shape check
        if expected_shape:
            self.shape_check(
                tensor=latent,
                expected_shape=expected_shape,
            )
        return latent

    def decode_mel_spectrogram(self, latent, decoder, expected_shape=None):
        # assert len(latent.shape) == 2
        assert latent.shape[1] == self.h_dim
        # pre convolution for channel matching
        x = decoder.pre(latent)
        # upscaling
        for up_block in decoder.ups:
            x = up_block(x)
        # post convolution for channel matching
        mel_spectrogram_pred = decoder.post(x)
        # sets the correct ranges for the spectrogram
        mel_spectrogram_pred = torch.expm1(mel_spectrogram_pred)
        # OPTIONAL: does a shape check
        if expected_shape:
            self.shape_check(
                tensor=mel_spectrogram_pred,
                expected_shape=expected_shape,
            )
        # returns the mel spectrogram
        return mel_spectrogram_pred
    
    def convert_eeg_latent(self, latent, expected_shape=None):
        latent_tokens = self.reasoner.in_reshaper(latent)
        pos_embeddings = self.reasoner.pos_embeddings(torch.arange(latent_tokens.shape[1], device=latent_tokens.device).unsqueeze(0).repeat(latent_tokens.shape[0], 1))
        latent_tokens = latent_tokens + pos_embeddings
        latent_tokens_transformed = self.reasoner.encoder(latent_tokens)
        latent_transformed = self.reasoner.out_reshaper(latent_tokens_transformed)
        # OPTIONAL: does a shape check
        if expected_shape:
            self.shape_check(
                tensor=latent_transformed,
                expected_shape=expected_shape,
            )
        # returns the mel spectrogram
        return latent_transformed

    def vocode(self, mel_spectrogram, vocoder, length, expected_shape=None):
        assert len(mel_spectrogram.shape) == 4
        latent = vocoder.pre(mel_spectrogram)
        for block in vocoder.downs:
            latent = block(latent)
        # trims the waveform
        waveform = vocoder.post(latent)[:, :, :length]
        # OPTIONAL: does a shape check
        if expected_shape:
            self.shape_check(
                tensor=waveform,
                expected_shape=expected_shape,
            )
        return waveform

    def generate_ecgs(self, eegs):
        # encodes the eeg waveform into a latent representation
        eegs_encoded, _ = self.encode(
            eegs, self.eegs_encoder, mel_spectrogrammer=self.eeg_mel_spectrogrammer
        )
        # decode the eeg latent representation into an ECG
        ecgs_pred, ecgs_mel_spectrogram = self.decode(
            eegs_encoded, self.ecgs_decoder, length=self.ecg_samples
        )
        return ecgs_pred, ecgs_mel_spectrogram

    def discriminate_mel_spectrogram(self, ecgs_mel_spectrogram):
        features = self.encode_mel_spectrogram(
            ecgs_mel_spectrogram, self.discriminator_encoder
        )
        return self.discriminator_cls(features)

    def discriminate_latent(self, latent):
        return self.discriminator_latent(latent)

    def waveform_to_mel_spectrogram(
        self, waveform, mel_spectrogrammer, expected_shape=None
    ):
        mel_spectrogram = mel_spectrogrammer(waveform)
        if expected_shape:
            self.shape_check(
                tensor=mel_spectrogram,
                expected_shape=expected_shape,
            )
        mel_spectrogram = mel_spectrogram + 1e-6
        return mel_spectrogram

    def waveform_to_spectrogram(
        self,
        waveform,
    ):
        # Compute STFT
        spec = torch.stack(
            [
                torch.stft(
                    waveform[:, i_channel, :],
                    n_fft=self.spectrogram_n_fft,
                    win_length=self.spectrogram_kernel_size,
                    hop_length=self.spectrogram_kernel_stride,
                    # window=self.stft_window,
                    center=True,
                    normalized=True,
                    onesided=True,
                    return_complex=True,
                )
                for i_channel in range(waveform.shape[1])
            ],
            dim=1,
        )

        # Separate magnitude and phase
        magnitude = torch.abs(spec)
        phase = torch.angle(spec)
        assert magnitude.shape == phase.shape

        return magnitude, phase

    def spectrogram_to_waveform(
        self,
        magnitude,
        phase,
        length,
    ):
        # Convert magnitude and phase to complex-valued spectrogram
        complex_spec = magnitude.float() * torch.exp(1j * phase.float())

        # Compute iSTFT
        return torch.stack(
            [
                torch.istft(
                    complex_spec[i_batch, :, :, :],
                    n_fft=self.spectrogram_n_fft,
                    win_length=self.spectrogram_kernel_size,
                    hop_length=self.spectrogram_kernel_stride,
                    # window=self.stft_window,
                    center=True,
                    normalized=True,
                    onesided=True,
                    length=length,
                    return_complex=False,
                )
                for i_batch in range(complex_spec.shape[0])
            ],
            dim=0,
        )

    def shared_step(self, batch, batch_idx):
        has_trainer = False
        try:
            self.trainer
            has_trainer = True
        except Exception:
            print("trainer not found")

        # retrieves the optimizers
        if self.training and has_trainer:
            if self.use_vocoder:
                (
                    optimizer_encoders,
                    optimizer_decoders,
                    optimizer_vocoder,
                ) = self.optimizers()
            else:
                (
                    optimizer_encoders,
                    optimizer_decoders,
                ) = self.optimizers()

        # INPUTS
        eegs = batch["eegs"].to(self.device)
        assert not torch.isnan(eegs).any(), "there are nans in the eegs"
        ecgs_gt = batch["ecgs"].to(self.device)
        assert not torch.isnan(ecgs_gt).any(), "there are nans in the input ecgs"
        batch_size = eegs.shape[0]
        outs = {}

        ##################################
        ##################################
        # SPECTROGRAMS
        # GENERATION
        ##################################
        ##################################

        # generates the mel spectrogram of the ground truth ecgs
        ecgs_mel_spectrogram_gt = self.waveform_to_mel_spectrogram(
            waveform=ecgs_gt,
            mel_spectrogrammer=self.ecg_mel_spectrogrammer,
            expected_shape=(
                batch_size,
                self.ecg_channels,
                self.spectrogram_frequencies,
                self.ecg_spectrogram_samples,
            ),
        )
        # generates the mel spectrogram of the ground truth eegs
        eegs_mel_spectrogram = self.waveform_to_mel_spectrogram(
            waveform=eegs,
            mel_spectrogrammer=self.eeg_mel_spectrogrammer,
            expected_shape=(
                batch_size,
                self.eeg_channels,
                self.spectrogram_frequencies,
                self.eeg_spectrogram_samples,
            ),
        )

        ##################################
        ##################################
        # VOCODER
        ##################################
        ##################################
        if self.use_vocoder:
            if self.training and has_trainer:
                optimizer_vocoder.zero_grad(set_to_none=True)

            # converts the ecg mel spectrogram into a waveform
            with profiler.record_function("ecg vocoder".upper()):
                # vocodes it into a waveform
                ecgs_gt_rec = self.vocode(
                    mel_spectrogram=ecgs_mel_spectrogram_gt,
                    vocoder=self.ecgs_vocoder,
                    length=self.ecg_samples,
                    expected_shape=ecgs_gt.shape,
                )

            outs["loss_V"] = self.reconstruction_loss(
                ecgs_gt_rec,
                ecgs_gt,
            )
            if self.training and has_trainer:
                self.manual_backward(outs["loss_V"])
                self.clip_gradients(
                    optimizer_vocoder,
                    gradient_clip_val=1.0,
                    gradient_clip_algorithm="norm",
                )
                optimizer_vocoder.step()

        ##################################
        ##################################
        # ENCODERS
        ##################################
        ##################################
        if self.training and has_trainer:
            optimizer_encoders.zero_grad(set_to_none=True)

        # encodes the ecg mel spectrogram into a latent representation
        with profiler.record_function("ecg mel to ecg latent".upper()):
            ecgs_latent = self.encode_mel_spectrogram(
                mel_spectrogram=ecgs_mel_spectrogram_gt,
                encoder=self.ecgs_encoder,
                expected_shape=(
                    batch_size,
                    self.h_dim,
                    self.spectrogram_latent_frequencies,
                    self.ecg_spectrogram_latent_samples,
                ),
            )

        # encodes the eeg mel spectrogram into a latent representation
        with profiler.record_function("eeg mel to eeg latent".upper()):
            eegs_latent = self.encode_mel_spectrogram(
                mel_spectrogram=eegs_mel_spectrogram,
                encoder=self.eegs_encoder,
                expected_shape=(
                    batch_size,
                    self.h_dim,
                    self.spectrogram_latent_frequencies,
                    self.eeg_spectrogram_latent_samples,
                ),
            )
            eegs_latent_upsampled = self.upsampler(eegs_latent)
            self.shape_check(eegs_latent_upsampled, ecgs_latent.shape)
            # eegs_latent_transformed = self.reasoner(eegs_latent_upsampled)
            eegs_latent_transformed = self.convert_eeg_latent(eegs_latent_upsampled)

            # computes the reconstruction loss between latent representations
            outs["loss_R"] = self.reconstruction_loss(
                gt=eegs_latent_transformed,
                pred=ecgs_latent,
                norm=1,
            )
            # outs["loss_R"] = F.cosine_embedding_loss(eegs_latent_transformed.view(batch_size, -1), ecgs_latent.view(batch_size, -1), torch.ones_like(eegs_latent[:, 0, 0, 0]))

        ##################################
        ##################################
        # DECODER
        ##################################
        ##################################
        # decodes the ecg latent into an ecg mel spectrogram
        with profiler.record_function("ecg latent to ecg mel".upper()):
            ecgs_mel_spectrogram_rec = self.decode_mel_spectrogram(
                latent=ecgs_latent,
                decoder=self.ecgs_decoder,
                expected_shape=(
                    batch_size,
                    self.ecg_channels,
                    self.spectrogram_frequencies,
                    self.ecg_spectrogram_samples,
                ),
            )
            # computes the reconstruction loss between latent representations
            outs["loss_G_ecg"] = self.reconstruction_loss(
                ecgs_mel_spectrogram_rec,
                ecgs_mel_spectrogram_gt,
            )
        # decodes the eeg latent into an ecg mel spectrogram
        with profiler.record_function("eeg latent to eeg mel".upper()):
            ecgs_mel_spectrogram_gen = self.decode_mel_spectrogram(
                latent=eegs_latent_transformed,
                decoder=self.ecgs_decoder,
                expected_shape=(
                    batch_size,
                    self.ecg_channels,
                    self.spectrogram_frequencies,
                    self.ecg_spectrogram_samples,
                ),
            )
            outs["loss_G_eeg"] = self.reconstruction_loss(
                ecgs_mel_spectrogram_gen,
                ecgs_mel_spectrogram_gt,
            )
        if self.training and has_trainer:
            if self.global_step % 2 == 0:
                loss_to_use = outs["loss_G_ecg"]
            else:
                loss_to_use = outs["loss_G_eeg"]
            self.manual_backward(loss_to_use + outs["loss_R"])
            self.clip_gradients(
                optimizer_encoders,
                gradient_clip_val=1.0,
                gradient_clip_algorithm="norm",
            )
            optimizer_encoders.step()

        # plot_reconstructed_spectrograms(
        #             sg_pred=ecgs_mel_spectrogram_pred[0],
        #             sg_gt=ecgs_mel_spectrogram_gt[0],
        #             vmin=0,
        #             path="./tmp.png"
        #         )

        # LOGGING
        if (
            has_trainer
            and batch_idx == 0
            and (self.training or self.current_epoch != 0)
        ):
            with torch.no_grad():
                if self.use_vocoder:
                    # vocodes the generated spectrogram into a waveform
                    ecgs_pred = self.vocode(
                        mel_spectrogram=ecgs_mel_spectrogram_rec,
                        vocoder=self.ecgs_vocoder,
                        length=self.ecg_samples,
                        expected_shape=ecgs_gt.shape,
                    )
                    outs["images/pred_ecg_wf"] = wandb.Image(
                        plot_reconstructed_ecgs_waveforms(
                            ecgs_pred_wf=ecgs_pred[0],
                            ecgs_gt_wf=ecgs_gt[0],
                            # path=f"images/epoch={self.current_epoch}.png",
                        ),
                        caption="generated ECGs waveforms",
                    )
                    outs["images/rec_ecg_wf"] = wandb.Image(
                        plot_reconstructed_ecgs_waveforms(
                            ecgs_pred_wf=ecgs_gt_rec[0],
                            ecgs_gt_wf=ecgs_gt[0],
                        ),
                        caption="reconstructed ECGs waveforms",
                    )

            outs["images/gen_ecg_mel"] = wandb.Image(
                plot_reconstructed_spectrograms(
                    sg_pred=ecgs_mel_spectrogram_gen[0],
                    sg_gt=ecgs_mel_spectrogram_gt[0],
                    vmin=0,
                ),
                caption="generated ECGs mel spectrograms",
            )
            outs["images/rec_ecg_mel"] = wandb.Image(
                plot_reconstructed_spectrograms(
                    sg_pred=ecgs_mel_spectrogram_rec[0],
                    sg_gt=ecgs_mel_spectrogram_gt[0],
                    vmin=0,
                    # path=f"images/epoch={self.current_epoch}.png",
                ),
                caption="reconstructed ECGs mel spectrograms",
            )
            # outs["images/rec_eeg_mel"] = wandb.Image(
            #     plot_reconstructed_spectrograms(
            #         sg_pred=eegs_mel_spectrogram_rec[0],
            #         sg_gt=eegs_mel_spectrogram[0],
            #         vmin=0,
            #     ),
            #     caption="reconstructed EEGs mel spectrograms",
            # )

            outs["images/latent"] = wandb.Image(
                plot_reconstructed_spectrograms(
                    sg_pred=eegs_latent_transformed[0, :8],
                    sg_gt=ecgs_latent[0, :8],
                    # path=f"images/epoch={self.current_epoch}.png",
                ),
                caption="reconstructed latent representation",
            )

        for loss_name, loss in [
            (k, v) for k, v in outs.items() if k.startswith("loss")
        ]:
            assert not torch.isnan(
                loss
            ).any(), f"{loss_name} has become None at epoch {self.current_epoch} and step {batch_idx}"
        outs["loss"] = sum(v for k, v in outs.items() if k.startswith("loss"))
        for k, v in outs.items():
            key_and_phase = f"{self.get_phase_name()}/{k}"
            # logs images
            if isinstance(v, wandb.Image):
                wandb.log({key_and_phase: v})
            # logs values
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                outs[k] = v.detach().cpu()
                self.log(
                    key_and_phase,
                    v,
                    batch_size=batch_size,
                    logger=True,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=k in {"loss_D", "loss_G"},
                )
            # else:
            #     raise Exception(f"unrecognized logging type {type(v)} for key '{k}'")
        return outs

    def discriminator_loss(self, logits, labels: int):
        batch_size = logits.shape[0]
        return F.binary_cross_entropy_with_logits(
            logits,
            torch.ones([batch_size, 1], device=self.device) * labels,
        )

    def reconstruction_loss(self, pred, gt, norm=1):
        if norm == 1:
            loss_fn = F.l1_loss
        elif norm == 2:
            loss_fn = F.mse_loss
        else:
            raise Exception(f"norm must be 1 or 2, got {norm}")
        return loss_fn(input=pred, target=gt)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def get_phase_name(self):
        return "train" if self.training else "val"

    @staticmethod
    def shape_check(tensor, expected_shape, name=None):
        error_msg = f"expected shape {expected_shape}, got {tuple(tensor.shape)}"
        if name is not None:
            error_msg += f" for tensor {name}"
        assert tensor.shape == expected_shape, error_msg


if __name__ == "__main__":
    from icecream import ic

    model = EEG2ECGModel(
        eeg_channels=14,
        ecg_channels=2,
        eeg_sampling_rate=128,
        ecg_sampling_rate=256,
        seconds=2,
        spectrogram_scale=2,
        encoder_layers=4,
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = {
        "eegs": torch.randn(
            [8, model.eeg_channels, model.seconds * model.eeg_sampling_rate],
            device=model.device,
        ),
        "ecgs": torch.randn(
            [8, model.ecg_channels, model.seconds * model.ecg_sampling_rate],
            device=model.device,
        ),
    }
    print(model)
    with profiler.profile(
        with_stack=False,
        profile_memory=True,
        use_cuda=True if torch.cuda.is_available() else False,
    ) as prof:
        sample_outs = model.shared_step(sample_batch, 0)
    print(
        prof.key_averages().table(
            sort_by="cuda_time",
            row_limit=10,
        )
    )
