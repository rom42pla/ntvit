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


class LambdaModule(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        import types

        assert type(lambd) is types.LambdaType
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class EEG2ECGModel(pl.LightningModule):
    def __init__(
        self,
        seconds: Union[int, float],
        eeg_channels: int,
        ecg_channels: int,
        eeg_sampling_rate: int,
        ecg_sampling_rate: int,
        spectrogram_scale: int = 8,
        n_mels: int = 32,
        layers: int = 3,
        h_dim: int = 512,
        learning_rate: float = 1e-3,
        activation_fn: str = "leaky_relu",
        norm_fn: str = "instance",
        dropout: float = 0.2,
    ):
        super(EEG2ECGModel, self).__init__()

        self.automatic_optimization = False
        self.learning_rate = learning_rate

        self.seconds = seconds
        self.layers = layers
        self.h_dim = h_dim

        self.spectrogram_n_fft = 256
        assert (
            self.spectrogram_n_fft & (self.spectrogram_n_fft - 1)
        ) == 0, "spectrogram_n_fft must be a power of 2"
        # self.spectrogram_frequencies = (self.spectrogram_n_fft // 2) + 1
        self.spectrogram_frequencies = n_mels
        self.spectrogram_latent_frequencies = ceil(
            self.spectrogram_frequencies / 2**self.layers
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
            self.eeg_spectrogram_samples / 2**self.layers
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
            self.ecg_spectrogram_samples / 2**self.layers
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

        self.kernel_size = 16
        self.padding_ecg = self.get_padding_for_patches(
            width=self.ecg_spectrogram_samples,
            height=self.spectrogram_frequencies,
            kernel_size=self.kernel_size,
        )
        self.padding_eeg = self.get_padding_for_patches(
            width=self.eeg_spectrogram_samples,
            height=self.spectrogram_frequencies,
            kernel_size=self.kernel_size,
        )
        self.num_ecg_patches = ((self.ecg_spectrogram_samples + self.padding_ecg[0] + self.padding_ecg[1]) // self.kernel_size) * ((self.spectrogram_frequencies + self.padding_ecg[2] + self.padding_ecg[3]) // self.kernel_size)
        self.num_eeg_patches = ((self.eeg_spectrogram_samples + self.padding_eeg[0] + self.padding_eeg[1]) // self.kernel_size) * ((self.spectrogram_frequencies + self.padding_eeg[2] + self.padding_eeg[3]) // self.kernel_size)
        # print(self.num_ecg_patches)
        # raise

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
            in_frequencies=self.spectrogram_frequencies,
            h_dim=self.h_dim,
            padding=self.padding_eeg,
            layers=self.layers,
            limit_outputs=False,
            dropout=dropout,
        )
        self.reasoner = self.build_reasoner(h_dim=self.h_dim)
        self.ecgs_encoder = self.build_encoder(
            in_channels=self.ecg_channels,
            in_frequencies=self.spectrogram_frequencies,
            h_dim=self.h_dim,
            padding=self.padding_ecg,
            layers=self.layers,
            limit_outputs=False,
            dropout=dropout,
        )

        self.ecgs_decoder = self.build_decoder(
            out_channels=self.ecg_channels,
            out_frequencies=self.spectrogram_frequencies + self.padding_ecg[2] + self.padding_ecg[3],
            out_length=self.ecg_spectrogram_samples + self.padding_ecg[0] + self.padding_ecg[1],
            h_dim=self.h_dim,
            padding=self.padding_ecg,
            layers=self.layers,
            dropout=dropout,
        )
        # self.eegs_decoder = self.build_decoder(
        #     out_channels=self.eeg_channels,
        #     s_dim=self.s_dim,
        #     h_dim=self.h_dim,
        #     layers=self.encoder_layers,
        #     kernel_size=self.kernel_size,
        # )

        self.save_hyperparameters(
            ignore=["eeg_mel_spectrogrammer", "ecg_mel_spectrogrammer"]
        )

    def configure_optimizers(self):
        opt_ecgs = torch.optim.AdamW(
            [p for m in [self.ecgs_encoder, self.ecgs_decoder] for p in m.parameters()],
            lr=self.learning_rate,
        )
        opt_eegs = torch.optim.AdamW(
            [p for m in [self.eegs_encoder, self.reasoner] for p in m.parameters()],
            lr=self.learning_rate,
        )
        return opt_ecgs, opt_eegs

    def build_encoder(
        self,
        in_channels,
        in_frequencies,
        layers,
        h_dim,
        padding=(0, 0, 0, 0),
        limit_outputs=True,
        kernel_size=16,
        dropout: float = 0.0,
    ):
        assert 0 <= dropout < 1
        return nn.ModuleDict(
            {
                "in_reshaper": nn.Sequential(
                    nn.ZeroPad2d(padding),
                    nn.Unfold(kernel_size=kernel_size, stride=kernel_size),
                    Rearrange("b c w -> b w c"),
                    # Rearrange("b c h w -> b w (c h)", c=in_channels, h=in_frequencies),
                    nn.Linear(in_channels * kernel_size**2, h_dim * 4),
                    self.activation_fn,
                    nn.Linear(h_dim * 4, h_dim),
                ),
                "encoder": nn.TransformerEncoder(
                    encoder_layer=nn.TransformerEncoderLayer(
                        d_model=h_dim,
                        nhead=8,
                        dim_feedforward=h_dim * 4,
                        activation=F.leaky_relu,
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
        )

    def build_decoder(
        self,
        out_channels,
        out_frequencies,
        out_length,
        layers,
        h_dim,
        dropout: float = 0.0,
        padding=(0, 0, 0, 0),
        kernel_size=16,
    ):
        assert 0 <= dropout < 1
        return nn.ModuleDict(
            {
                "decoder": nn.TransformerEncoder(
                    encoder_layer=nn.TransformerEncoderLayer(
                        d_model=h_dim,
                        nhead=8,
                        dim_feedforward=h_dim * 4,
                        activation=F.leaky_relu,
                        batch_first=True,
                        dropout=dropout,
                    ),
                    num_layers=layers,
                ),
                "out_reshaper": nn.Sequential(
                    nn.Linear(h_dim, h_dim * 4),
                    self.activation_fn,
                    nn.Linear(h_dim * 4, out_channels * kernel_size**2),
                    Rearrange("b w c -> b c w"),
                    nn.Fold(
                        output_size=(out_frequencies, out_length),
                        kernel_size=kernel_size,
                        stride=kernel_size,
                    ),
                    LambdaModule(lambda x: x[:, :, 
                                             padding[2]:-padding[3] if padding[3]>0 else None,
                                          padding[0]:-padding[1] if padding[1] > 0 else 0]),
                    nn.Softplus(),
                ),
                "pos_embeddings": nn.Embedding(
                    num_embeddings=2048, embedding_dim=h_dim
                ),
            }
        )

    def build_reasoner(
        self,
        h_dim=768,
    ):
        return nn.ModuleDict(
            {
                "reasoner": nn.Sequential(
                    nn.Linear(h_dim, h_dim * 4),
                    self.activation_fn,
                    nn.Linear(h_dim * 4, h_dim),
                )
            }
        )

    def forward(self, x):
        pass

    def encode_mel_spectrogram(self, mel_spectrogram, encoder, expected_shape=None):
        # mel_spectrogram_padded = nn.ZeroPad2d(padding=(padding_left,padding_right, padding_up,padding_down))(mel_spectrogram)

        # unfolded = F.unfold(mel_spectrogram_padded, kernel_size=(16, 16), stride=(16, 16))
        # assert unfolded.numel() == mel_spectrogram_padded.numel(), f"{unfolded.numel()} != {mel_spectrogram_padded.numel()}"

        # folded =  F.fold(unfolded, output_size=mel_spectrogram_padded.shape[2:], kernel_size=(16, 16), stride=(16, 16))
        # assert folded.shape == mel_spectrogram_padded.shape, f"{folded.shape} != {mel_spectrogram_padded.shape}"
        # assert torch.isclose(mel_spectrogram_padded, folded).all()
        # mel_spectrogram_restored = folded
        # mel_spectrogram_restored = folded[:, :,
        #                                   padding_down:-padding_up if padding_up>0 else None,
        #                                   padding_left:-padding_right if padding_right > 0 else 0]
        # assert mel_spectrogram_restored.shape == mel_spectrogram.shape, f"{mel_spectrogram_restored.shape} != {mel_spectrogram.shape}"
        # assert torch.isclose(mel_spectrogram_restored, mel_spectrogram).all()
        # raise
        latent_tokens = encoder.in_reshaper(mel_spectrogram)
        cls_token = encoder.learned_tokens(
            torch.arange(1, device=latent_tokens.device)
            .unsqueeze(0)
            .repeat(latent_tokens.shape[0], 1)
        )
        latent_tokens = torch.cat([cls_token, latent_tokens], dim=1)
        pos_embeddings = encoder.pos_embeddings(
            torch.arange(latent_tokens.shape[1], device=latent_tokens.device)
            .unsqueeze(0)
            .repeat(latent_tokens.shape[0], 1)
        )
        # latent_tokens = torch.cat([latent_tokens, pos_embeddings], dim=-1)
        latent_tokens = sum([latent_tokens, pos_embeddings])
        latent_tokens_transformed = encoder.encoder(latent_tokens)[:, 0]
        # latent_tokens_transformed = F.tanh(latent_tokens_transformed)
        # OPTIONAL: does a shape check
        if expected_shape:
            self.shape_check(
                tensor=latent_tokens_transformed,
                expected_shape=expected_shape,
            )
        return latent_tokens_transformed

    def decode_mel_spectrogram(self, latent, decoder, out_length, expected_shape=None):
        # latent_tokens_tgt = torch.randn(latent.shape[0], out_length, latent.shape[-1] // 2, device=latent.device)
        pos_embeddings_tgt = decoder.pos_embeddings(
            torch.arange(out_length, device=latent.device)
            .unsqueeze(0)
            .repeat(latent.shape[0], 1)
        )
        # latent_tokens_tgt = torch.cat([latent_tokens_tgt, pos_embeddings_tgt], dim=-1)
        # print(latent.shape, pos_embeddings_tgt.shape)
        latent_tgt = latent.unsqueeze(1) + pos_embeddings_tgt
        mask = nn.Transformer.generate_square_subsequent_mask(
            out_length, device=latent.device
        )
        latent_tokens_pred = decoder.decoder(latent_tgt, mask=mask, is_causal=True)
        mel_spectrogram_pred = decoder.out_reshaper(latent_tokens_pred)
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
        pos_embeddings = self.reasoner.pos_embeddings(
            torch.arange(latent_tokens.shape[1], device=latent_tokens.device)
            .unsqueeze(0)
            .repeat(latent_tokens.shape[0], 1)
        )
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

    def waveform_to_mel_spectrogram(
        self, waveform, mel_spectrogrammer, expected_shape=None
    ):
        mel_spectrogram = mel_spectrogrammer(waveform)
        if expected_shape:
            self.shape_check(
                tensor=mel_spectrogram,
                expected_shape=expected_shape,
            )
        # mel_spectrogram = mel_spectrogram + 1e-6
        return mel_spectrogram

    def shared_step(self, batch, batch_idx):
        has_trainer = False
        try:
            self.trainer
            has_trainer = True
        except Exception:
            print("trainer not found")

        # retrieves the optimizers
        if self.training and has_trainer:
            opt_ecgs, opt_eegs = self.optimizers()

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
        # ECGs
        ##################################
        ##################################
        if self.training and has_trainer:
            opt_ecgs.zero_grad(set_to_none=True)

        # encodes the ecg mel spectrogram into a latent representation
        with profiler.record_function("ecg mel to ecg latent".upper()):
            ecgs_latent = self.encode_mel_spectrogram(
                mel_spectrogram=ecgs_mel_spectrogram_gt,
                encoder=self.ecgs_encoder,
                expected_shape=(
                    batch_size,
                    self.h_dim,
                ),
            )

        # decodes the ecg latent into an ecg mel spectrogram
        with profiler.record_function("ecg latent to ecg mel".upper()):
            ecgs_mel_spectrogram_rec = self.decode_mel_spectrogram(
                latent=ecgs_latent,
                decoder=self.ecgs_decoder,
                # out_length=self.ecg_spectrogram_samples,
                out_length=self.num_ecg_patches,
                expected_shape=(
                    batch_size,
                    self.ecg_channels,
                    self.spectrogram_frequencies,
                    self.ecg_spectrogram_samples,
                ),
            )
            # computes the reconstruction loss between latent representations
            outs["loss_G_ecg"] = self.reconstruction_loss(
                ecgs_mel_spectrogram_rec, ecgs_mel_spectrogram_gt, norm=2
            )

        if self.training and has_trainer:
            self.manual_backward(outs["loss_G_ecg"])
            self.clip_gradients(
                opt_ecgs,
                gradient_clip_val=1.0,
                gradient_clip_algorithm="norm",
            )
            opt_ecgs.step()

        ##################################
        ##################################
        # EEGs
        ##################################
        ##################################
        if self.training and has_trainer:
            opt_eegs.zero_grad(set_to_none=True)

        # encodes the eeg mel spectrogram into a latent representation
        with profiler.record_function("eeg mel to eeg latent".upper()):
            eegs_latent = self.encode_mel_spectrogram(
                mel_spectrogram=eegs_mel_spectrogram,
                encoder=self.eegs_encoder,
                expected_shape=(
                    batch_size,
                    self.h_dim,
                ),
            )
            eegs_latent = self.reasoner.reasoner(eegs_latent)
            ecgs_latent_2 = self.encode_mel_spectrogram(
                mel_spectrogram=ecgs_mel_spectrogram_gt,
                encoder=self.ecgs_encoder,
                expected_shape=(
                    batch_size,
                    self.h_dim,
                ),
            )
            # computes the reconstruction loss between latent representations
            outs["loss_R"] = self.reconstruction_loss(
                pred=eegs_latent,
                gt=ecgs_latent_2.detach(),
                norm=2,
            )

        # decodes the eeg latent into an ecg mel spectrogram
        with profiler.record_function("eeg latent to eeg mel".upper()):
            ecgs_mel_spectrogram_gen = self.decode_mel_spectrogram(
                latent=eegs_latent,
                decoder=self.ecgs_decoder,
                out_length=self.num_ecg_patches,
                expected_shape=(
                    batch_size,
                    self.ecg_channels,
                    self.spectrogram_frequencies,
                    self.ecg_spectrogram_samples,
                ),
            )
            outs["loss_G_eeg"] = self.reconstruction_loss(
                ecgs_mel_spectrogram_gen, ecgs_mel_spectrogram_gt, norm=2
            )
        if self.training and has_trainer:
            self.manual_backward(outs["loss_G_eeg"] + outs["loss_R"])
            self.clip_gradients(
                opt_eegs,
                gradient_clip_val=1.0,
                gradient_clip_algorithm="norm",
            )
            opt_eegs.step()

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
                    sg_pred=eegs_latent.unsqueeze(1)[:8],
                    sg_gt=ecgs_latent.unsqueeze(1)[:8],
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

    model = EEG2ECGModel(
        eeg_channels=14,
        ecg_channels=2,
        eeg_sampling_rate=128,
        ecg_sampling_rate=256,
        seconds=2,
        spectrogram_scale=2,
        layers=4,
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
