from typing import Any, Dict, Optional, Union
from math import ceil, floor, log2
from einops.layers.torch import Rearrange
from torch import nn
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.autograd.profiler as profiler
import torchaudio
import torchmetrics
import wandb

try:
    from plots import (
        plot_reconstructed_ecgs_waveforms,
        plot_reconstructed_spectrograms,
    )
    from models.sdtw import SoftDTW
except:
    print("error loading libraries")


class LambdaModule(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        import types

        assert type(lambd) is types.LambdaType
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class AutoEncoder(pl.LightningModule):
    def __init__(
        self,
        seconds: Union[int, float],
        channels: int,
        sampling_rate: int,
        min_frequency: int,
        max_frequency: int,
        architecture: str = "conv",
        n_mels: int = 16,
        spectrogram_power: int = 2,
        patches_size: int = 3,
        layers: int = 3,
        h_dim: int = 1024,
        learning_rate: float = 1e-3,
        activation_fn: str = "leaky_relu",
        norm_fn: str = "batch",
        dropout: float = 0.,
    ):
        super(AutoEncoder, self).__init__()

        self.automatic_optimization = False
        self.available_architectures = {"full", "conv"}
        assert (
            architecture in self.available_architectures
        ), f"expected one in {self.available_architectures}, got {architecture}"
        self.architecture = architecture

        self.seconds = seconds
        self.spectrogram_power = spectrogram_power

        self.channels = channels
        self.sampling_rate = sampling_rate

        def find_closest_power_of_2(x):
            for i in range(100):
                if 2**i >= x:
                    break
            return 2**i

        # self.spectrogram_n_fft = find_closest_power_of_2(self.sampling_rate)
        self.spectrogram_n_fft = 256
        self.spectrogram_frequencies = n_mels
        self.spectrogram_kernel_size = floor(self.sampling_rate / 8)
        self.spectrogram_kernel_stride = self.spectrogram_kernel_size // 2
        self.min_frequency, self.max_frequency = min_frequency, max_frequency
        # self.frequency_resolution: float = (
        #     self.sampling_rate / self.spectrogram_n_fft
        # )
        self.samples = self.sampling_rate * ceil(self.seconds)
        self.spectrogram_samples = (
            ceil(self.samples / self.spectrogram_kernel_stride) + 1
        )
        self.mel_spectrogrammer = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sampling_rate,
            n_fft=self.spectrogram_n_fft,
            win_length=self.spectrogram_kernel_size,
            hop_length=self.spectrogram_kernel_stride,
            f_min=self.min_frequency,
            f_max=self.max_frequency,
            n_mels=self.spectrogram_frequencies,
            power=self.spectrogram_power,
            normalized=False,
        )

        ##################################
        ##################################
        # OPTIMIZER
        ##################################
        ##################################
        self.learning_rate = learning_rate

        ##################################
        ##################################
        # NEURAL NETWORK
        # PARAMETERS
        ##################################
        ##################################
        self.layers = layers
        self.h_dim = h_dim
        self.patches_size = patches_size
        self.padding = self.get_padding_for_patches(
            width=self.spectrogram_samples,
            height=self.spectrogram_frequencies,
            kernel_size=self.patches_size,
        )
        self.num_patches = (
            (self.spectrogram_samples + self.padding[0] + self.padding[1])
            // self.patches_size
        ) * (
            (self.spectrogram_frequencies + self.padding[2] + self.padding[3])
            // self.patches_size
        )

        ####################################
        # MODULES
        ####################################
        if activation_fn == "gelu":
            self.activation_fn = nn.GELU()
        elif activation_fn == "leaky_relu":
            self.activation_fn = nn.LeakyReLU()
        elif activation_fn == "selu":
            self.activation_fn = nn.SELU()
        if norm_fn == "instance":
            self.norm_fn = nn.InstanceNorm2d
        elif norm_fn == "batch":
            self.norm_fn = nn.BatchNorm2d
        elif norm_fn in {None, "none"}:
            self.norm_fn = None
        self.latent_frequencies = ceil(self.spectrogram_frequencies / 2**self.layers)
        self.latent_samples = ceil(self.spectrogram_samples / 2**self.layers)
        self.encoder = self.build_encoder(
            in_channels=self.channels,
            patches_size=self.patches_size,
            h_dim=self.h_dim,
            padding=self.padding,
            layers=self.layers,
            architecture=self.architecture,
            dropout=dropout,
            latent_frequencies=self.latent_frequencies,
            latent_samples=self.latent_samples,
        )

        self.decoder = self.build_decoder(
            out_channels=self.channels,
            out_frequencies=self.spectrogram_frequencies
            + self.padding[2]
            + self.padding[3],
            out_length=self.spectrogram_samples + self.padding[0] + self.padding[1],
            h_dim=self.h_dim,
            padding=self.padding,
            layers=self.layers,
            patches_size=self.patches_size,
            dropout=dropout,
            architecture=self.architecture,
            latent_frequencies=self.latent_frequencies,
            latent_samples=self.latent_samples,
        )
        self.save_hyperparameters()

    def configure_optimizers(self):
        # optimizers
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt,
            T_0=10,
            eta_min=self.learning_rate * 9e-1,
            verbose=False,
        )
        return [opt], [
            {"scheduler": sch, "interval": "step"},
        ]

    def build_encoder(
        self,
        in_channels,
        layers,
        h_dim,
        padding=(0, 0, 0, 0),
        patches_size: Optional[int] = None,
        dropout: float = 0.0,
        architecture="full",
        latent_frequencies: Optional[int] = None,
        latent_samples: Optional[int] = None,
    ):
        assert architecture in {"full", "conv"}
        assert 0 <= dropout < 1
        if architecture == "full":
            modules = {
                "encoder": nn.TransformerEncoder(
                    encoder_layer=nn.TransformerEncoderLayer(
                        d_model=h_dim,
                        nhead=8,
                        dim_feedforward=h_dim * 4,
                        activation=F.selu,
                        batch_first=True,
                        dropout=dropout,
                    ),
                    num_layers=layers,
                ),
                "pos_embeddings": nn.Embedding(
                    num_embeddings=2048, embedding_dim=h_dim
                ),
                "learned_tokens": nn.Embedding(num_embeddings=1, embedding_dim=h_dim),
            }
            if patches_size is not None:
                modules["in_reshaper"] = nn.Sequential(
                    nn.ZeroPad2d(padding),
                    nn.Unfold(kernel_size=patches_size, stride=patches_size),
                    Rearrange("b c w -> b w c"),
                    nn.Linear(in_channels * patches_size**2, h_dim * 4),
                    self.activation_fn,
                    # nn.Dropout(dropout),
                    nn.Linear(h_dim * 4, h_dim),
                )
        elif architecture == "conv":
            assert latent_frequencies is not None
            assert latent_samples is not None
            kernel_size = patches_size
            padding = floor(kernel_size / 2)
            s_dim = h_dim // 2**layers
            modules = {
                "pre": nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=s_dim,
                        kernel_size=7,
                        stride=1,
                        padding=3,
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
                            nn.Dropout(dropout),
                            # nn.Conv2d(
                            #     in_channels=ceil(s_dim * 2 ** (i_layer + 1)),
                            #     out_channels=ceil(s_dim * 2 ** (i_layer + 1)),
                            #     kernel_size=3,
                            #     stride=1,
                            #     padding=1,
                            #     bias=False if self.norm_fn else True,
                            # ),
                            # self.norm_fn(ceil(s_dim * 2 ** (i_layer + 1)))
                            # if self.norm_fn
                            # else nn.Identity(),
                            # self.activation_fn,
                        )
                        for i_layer in range(layers)
                    ]
                ),
                "post": nn.Sequential(
                    # nn.Conv2d(
                    #     in_channels=h_dim,
                    #     out_channels=h_dim,
                    #     kernel_size=(latent_frequencies, latent_samples),
                    #     stride=1,
                    #     padding=0,
                    #     bias=True,
                    # ),
                    Rearrange("b c h w -> b (c h w)"),
                    nn.Linear(
                        in_features=h_dim * latent_frequencies * latent_samples,
                        out_features=self.h_dim,
                    ),
                ),
            }
        return nn.ModuleDict(modules)

    def build_decoder(
        self,
        out_channels,
        out_frequencies,
        out_length,
        layers,
        h_dim,
        dropout: float = 0.0,
        padding=(0, 0, 0, 0),
        patches_size=16,
        architecture="full",
        latent_frequencies: Optional[int] = None,
        latent_samples: Optional[int] = None,
    ):
        assert 0 <= dropout < 1
        assert architecture in {"full", "conv"}
        if architecture == "full":
            modules = {
                "decoder": nn.TransformerEncoder(
                    encoder_layer=nn.TransformerEncoderLayer(
                        d_model=h_dim,
                        nhead=8,
                        dim_feedforward=h_dim * 4,
                        activation=F.selu,
                        batch_first=True,
                        dropout=dropout,
                    ),
                    num_layers=layers,
                ),
                "out_reshaper": nn.Sequential(
                    nn.Linear(h_dim, h_dim * 4),
                    self.activation_fn,
                    # nn.Dropout(dropout),
                    nn.Linear(h_dim * 4, out_channels * patches_size**2),
                    self.activation_fn,
                    Rearrange("b w c -> b c w"),
                    nn.Fold(
                        output_size=(out_frequencies, out_length),
                        kernel_size=patches_size,
                        stride=patches_size,
                    ),
                    LambdaModule(
                        lambda x: x[
                            :,
                            :,
                            padding[2] : -padding[3] if padding[3] > 0 else None,
                            padding[0] : -padding[1] if padding[1] > 0 else 0,
                        ]
                    ),
                    nn.Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=7,
                        stride=1,
                        padding=3,
                        bias=True,
                    ),
                    # nn.Softplus(),
                    # nn.Sigmoid(),
                ),
                "pos_embeddings": nn.Embedding(
                    num_embeddings=2048, embedding_dim=h_dim
                ),
            }
        elif architecture == "conv":
            assert latent_frequencies is not None
            assert latent_samples is not None
            kernel_size = patches_size
            padding = floor(kernel_size / 2)
            s_dim = h_dim // 2**layers
            modules = {
                "pre": nn.Sequential(
                    # Rearrange("b c -> b c () ()"),
                    # nn.ConvTranspose2d(
                    #     in_channels=h_dim,
                    #     out_channels=h_dim,
                    #     kernel_size=(latent_frequencies, latent_samples),
                    #     stride=1,
                    #     padding=0,
                    # ),
                    # self.norm_fn(h_dim) if self.norm_fn else nn.Identity(),
                    # self.activation_fn,
                    nn.Linear(
                        in_features=self.h_dim,
                        out_features=self.h_dim
                        * self.latent_frequencies
                        * self.latent_samples,
                    ),
                    self.activation_fn,
                    Rearrange(
                        "b (c h w) -> b c h w",
                        c=self.h_dim,
                        h=self.latent_frequencies,
                        w=self.latent_samples,
                    ),
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
                            # nn.Conv2d(
                            #     in_channels=ceil(h_dim * 2 ** -(i_layer + 1)),
                            #     out_channels=ceil(h_dim * 2 ** -(i_layer + 1)),
                            #     kernel_size=3,
                            #     stride=1,
                            #     padding=1,
                            #     bias=False if self.norm_fn else True,
                            # ),
                            # # nn.BatchNorm2d(ceil(h_dim * 2 ** -(i_layer + 1))),
                            # self.norm_fn(ceil(h_dim * 2 ** -(i_layer + 1)))
                            # if self.norm_fn
                            # else nn.Identity(),
                            # self.activation_fn,
                        )
                        for i_layer in range(layers)
                    ]
                ),
                "post": nn.Sequential(
                    nn.Conv2d(
                        in_channels=s_dim,
                        out_channels=out_channels,
                        kernel_size=7,
                        stride=1,
                        padding=3,
                        bias=True,
                    ),
                    # nn.Softplus(),
                    nn.Sigmoid(),
                ),
            }
        return nn.ModuleDict(modules)

    def shared_step(self, batch, batch_idx):
        has_trainer = False
        try:
            self.trainer
            has_trainer = True
        except Exception:
            print("trainer not found")

        # retrieves the optimizers
        if self.training and has_trainer:
            opt = self.optimizers()
            sch = self.lr_schedulers()

        outs: Dict[str, Any] = {}
        outs.update(self(batch))

        # computes the reconstruction loss between latent representations
        outs["loss_rec"] = self.reconstruction_loss(
            outs["mel_spectrogram_rec"], outs["mel_spectrogram_gt"], norm=2
        )
        outs["loss"] = sum(v for k, v in outs.items() if k.startswith("loss"))

        if self.training and has_trainer:
            opt.zero_grad(set_to_none=True)
            self.manual_backward(outs["loss"])
            self.clip_gradients(
                opt,
                gradient_clip_val=1.0,
                gradient_clip_algorithm="norm",
            )
            opt.step()
            sch.step(self.current_epoch + batch_idx / self.get_dataloader_length())
            outs["lr"] = sch.get_last_lr()[-1]

        # asserts that no loss is corrupted
        for loss_name, loss in [
            (k, v) for k, v in outs.items() if k.startswith("loss")
        ]:
            assert not torch.isnan(
                loss
            ).any(), f"{loss_name} has become None at epoch {self.current_epoch} and step {batch_idx}"

        ##################################
        ##################################
        # LOGGING
        ##################################
        ##################################
        if has_trainer and batch_idx == 0:
            outs["images/mel_rec"] = wandb.Image(
                plot_reconstructed_spectrograms(
                    sg_pred=outs["mel_spectrogram_rec"][0, :2],
                    sg_gt=outs["mel_spectrogram_gt"][0, :2],
                    vmin=0,
                ),
                caption="generated Mel spectrograms",
            )
            self.log_dict(
                {
                    f"{self.get_phase_name()}/{k}": v
                    for k, v in outs.items()
                    if (isinstance(v, torch.Tensor) and v.numel() == 1)
                    or isinstance(v, (float, int))
                },
                on_step=False,
                on_epoch=True,
            )
            try:
                wandb.log(
                    {
                        f"{self.get_phase_name()}/{k}": v
                        for k, v in outs.items()
                        if isinstance(v, wandb.Image)
                    }
                )
            except:
                pass
        return outs

    def forward(self, batch):
        ##################################
        ##################################
        # INPUTS
        ##################################
        ##################################
        assert isinstance(batch, torch.Tensor)
        batch_size = batch.shape[0]
        assert batch.shape == (
            batch_size,
            self.channels,
            self.samples,
        ), f"got {batch.shape}"
        waveforms = batch.to(self.device)

        ##################################
        ##################################
        # SPECTROGRAM
        ##################################
        ##################################
        mel_spectrogram_gt = self.waveform_to_mel_spectrogram(
            waveform=waveforms,
            limit_outputs=True,
            expected_shape=(
                batch_size,
                self.channels,
                self.spectrogram_frequencies,
                self.spectrogram_samples,
            ),
        )

        ##################################
        ##################################
        # ENCODER
        ##################################
        ##################################
        with profiler.record_function("encoder".upper()):
            latent = self.encode_mel_spectrogram(
                mel_spectrogram=mel_spectrogram_gt,
                pooled_outputs=True,
                return_hidden_states=False,
                expected_shape=(
                    batch_size,
                    self.h_dim,
                ),
            )
        ##################################
        ##################################
        # DECODER
        ##################################
        ##################################
        with profiler.record_function("decoder".upper()):
            mel_spectrogram_rec = self.decode_mel_spectrogram(
                latent=latent,
                out_length=self.num_patches,
                expected_shape=(
                    batch_size,
                    self.channels,
                    self.spectrogram_frequencies,
                    self.spectrogram_samples,
                ),
            )
        return {
            "waveforms": waveforms,
            "latent": latent,
            "mel_spectrogram_gt": mel_spectrogram_gt,
            "mel_spectrogram_rec": mel_spectrogram_rec,
        }

    def waveform_to_mel_spectrogram(
        self, waveform, limit_outputs: bool = False, expected_shape=None
    ):
        mel_spectrogram = self.mel_spectrogrammer(waveform)
        if limit_outputs:
            v_max = torch.amax(mel_spectrogram, dim=(1, 2, 3), keepdim=True)
            mel_spectrogram = mel_spectrogram / v_max
        if expected_shape:
            self.shape_check(
                tensor=mel_spectrogram,
                expected_shape=expected_shape,
            )
        return mel_spectrogram

    def encode_mel_spectrogram(
        self,
        mel_spectrogram,
        limit_outputs: bool = False,
        pooled_outputs: bool = False,
        use_reasoner: bool = False,
        return_hidden_states: bool = False,
        expected_shape=None,
    ):
        if self.architecture == "full":
            latent_tokens = self.encoder.in_reshaper(mel_spectrogram)
            if pooled_outputs:
                cls_token = self.encoder.learned_tokens(
                    torch.arange(1, device=latent_tokens.device)
                    .unsqueeze(0)
                    .repeat(latent_tokens.shape[0], 1)
                )
                latent_tokens = torch.cat([cls_token, latent_tokens], dim=1)
            pos_embeddings = self.encoder.pos_embeddings(
                torch.arange(latent_tokens.shape[1], device=latent_tokens.device)
                .unsqueeze(0)
                .repeat(latent_tokens.shape[0], 1)
            )
            latent_tokens = sum([latent_tokens, pos_embeddings])
            states = []
            for layer in self.encoder.encoder.layers:
                latent_tokens = layer(latent_tokens)
                states.append(latent_tokens)
            latent_tokens_transformed = latent_tokens
            if pooled_outputs:
                latent_tokens_transformed = latent_tokens_transformed[:, 0]
        elif self.architecture == "conv":
            # pre convolution for reshaping and channel matching
            x = self.encoder.pre(mel_spectrogram)
            # downscaling
            for down_block in self.encoder.downs:
                x = down_block(x)
            # post convolution for channel matching
            latent_tokens_transformed = self.encoder.post(x)
        # if limit_outputs:
        # latent_tokens_transformed = F.tanh(latent_tokens_transformed)
        # OPTIONAL: does a shape check
        if expected_shape:
            self.shape_check(
                tensor=latent_tokens_transformed,
                expected_shape=expected_shape,
            )
        if return_hidden_states and self.architecture == "full":
            return latent_tokens_transformed, states
        else:
            return latent_tokens_transformed

    def decode_mel_spectrogram(self, latent, out_length, expected_shape=None):
        if self.architecture == "full":
            pos_embeddings = self.decoder.pos_embeddings(
                torch.arange(out_length, device=latent.device)
                .unsqueeze(0)
                .repeat(latent.shape[0], 1)
            )
            # latent_tokens_tgt = torch.cat([latent_tokens_tgt, pos_embeddings_tgt], dim=-1)
            latent_tgt = sum([latent.unsqueeze(1), pos_embeddings])
            mask = nn.Transformer.generate_square_subsequent_mask(
                out_length, device=latent.device
            )
            latent_tokens_pred = self.decoder.decoder(
                latent_tgt, mask=mask, is_causal=True
            )
            mel_spectrogram_pred = self.decoder.out_reshaper(latent_tokens_pred)
        elif self.architecture == "conv":
            # pre convolution for channel matching
            x = self.decoder.pre(latent)
            # upscaling
            for up_block in self.decoder.ups:
                x = up_block(x)
            # post convolution for channel matching
            mel_spectrogram_pred = self.decoder.post(x)
        # mel_spectrogram_pred = torch.expm1(mel_spectrogram_pred)
        # OPTIONAL: does a shape check
        if expected_shape:
            self.shape_check(
                tensor=mel_spectrogram_pred,
                expected_shape=expected_shape,
            )
        # returns the mel spectrogram
        return mel_spectrogram_pred

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

    def get_dataloader_length(self):
        if self.training:
            return len(self.trainer.train_dataloader)
        else:
            return len(self.trainer.val_dataloader)

    def get_phase_name(self):
        return "train" if self.training else "val"

    @staticmethod
    def shape_check(tensor, expected_shape, name=None):
        error_msg = f"expected shape {expected_shape}, got {tuple(tensor.shape)}"
        if name is not None:
            error_msg += f" for tensor {name}"
        assert tensor.shape == expected_shape, error_msg

    @staticmethod
    def get_padding_for_patches(height, width, kernel_size):
        padding_w = kernel_size * ceil(width / kernel_size) - width
        padding_h = kernel_size * ceil(height / kernel_size) - height
        if padding_w % 2 == 0:
            padding_left = padding_right = padding_w // 2
        else:
            padding_left = floor(padding_w / 2)
            padding_right = padding_left + 1
        if padding_h % 2 == 0:
            padding_up = padding_down = padding_h // 2
        else:
            padding_up = floor(padding_h / 2)
            padding_down = padding_up + 1
        return padding_left, padding_right, padding_up, padding_down


if __name__ == "__main__":
    from icecream import ic

    model = AutoEncoder(
        channels=14,
        sampling_rate=128,
        seconds=2,
        layers=4,
        min_frequency=1,
        max_frequency=50,
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.randn(
        [8, model.channels, model.seconds * model.sampling_rate],
        device=model.device,
    )
    print(model)
    with profiler.profile(
        with_stack=False,
        profile_memory=True,
        use_cuda=True if torch.cuda.is_available() else False,
    ) as prof:
        # sample_outs = model.shared_step(sample_batch, 0)
        sample_outs = model.shared_step(sample_batch, 0)
    print(
        prof.key_averages().table(
            sort_by="cuda_time",
            row_limit=10,
        )
    )
