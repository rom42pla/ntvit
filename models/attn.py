from typing import Optional, Union
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


class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims

        self.T = nn.Parameter(torch.Tensor(in_features, out_features, kernel_dims))
        nn.init.normal_(self.T, 0, 1)

    def forward(self, x):
        # x is NxA
        # T is AxBxC
        matrices = x.mm(self.T.view(self.in_features, -1))
        matrices = matrices.view(-1, self.out_features, self.kernel_dims)

        M = matrices.unsqueeze(0)  # 1xNxBxC
        M_T = M.permute(1, 0, 2, 3)  # Nx1xBxC
        norm = torch.abs(M - M_T).sum(3)  # NxNxB
        expnorm = torch.exp(-norm)
        o_b = expnorm.sum(0) - 1  # NxB, subtract self distance

        x = torch.cat([x, o_b], 1)
        return x


class EEG2ECGModel(pl.LightningModule):
    def __init__(
        self,
        seconds: Union[int, float],
        eeg_channels: int,
        ecg_channels: int,
        eeg_sampling_rate: int,
        ecg_sampling_rate: int,
        architecture: str = "full",
        n_mels_ecgs: int = 16,
        n_mels_eegs: int = 16,
        spectrogram_power: int = 1,
        patches_size: int = 4,
        layers_for_ecgs: int = 1,
        layers_for_eegs: int = 4,
        h_dim: int = 768,
        learning_rate_ecgs: float = 1e-3,
        learning_rate_eegs: float = 1e-3,
        learning_rate_disc: float = 1e-3,
        activation_fn: str = "selu",
        norm_fn: str = "batch",
        dropout: float = 0.2,
    ):
        super(EEG2ECGModel, self).__init__()

        self.automatic_optimization = False
        self.available_architectures = {"full", "simple"}
        assert architecture in self.available_architectures, f"expected one in {self.available_architectures}, got {architecture}"
        self.architecture = architecture

        ##################################
        ##################################
        # EEGs
        ##################################
        ##################################
        self.seconds = seconds
        self.spectrogram_power = spectrogram_power

        self.eeg_channels = eeg_channels
        self.eeg_sampling_rate = eeg_sampling_rate
        # formula taken from https://stackoverflow.com/questions/42821425/define-an-interval-for-frequency-of-stft-outputs
        self.eeg_min_frequency, self.eeg_max_frequency = 1, 50

        def find_closest_power_of_2(x):
            for i in range(100):
                if 2**i >= x:
                    break
            return 2**i

        self.eeg_spectrogram_n_fft = find_closest_power_of_2(self.eeg_sampling_rate)
        self.eeg_spectrogram_frequencies = n_mels_eegs
        self.eeg_spectrogram_kernel_size = floor(self.eeg_sampling_rate * 0.25)
        self.eeg_spectrogram_kernel_stride = self.eeg_spectrogram_kernel_size // 2
        self.eeg_frequency_resolution: float = (
            self.eeg_sampling_rate / self.eeg_spectrogram_n_fft
        )
        self.eeg_samples = self.eeg_sampling_rate * ceil(self.seconds)
        self.eeg_spectrogram_samples = (
            ceil(self.eeg_samples / self.eeg_spectrogram_kernel_stride) + 1
        )
        self.eeg_mel_spectrogrammer = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.eeg_sampling_rate,
            n_fft=self.eeg_spectrogram_n_fft,
            win_length=self.eeg_spectrogram_kernel_size,
            hop_length=self.eeg_spectrogram_kernel_stride,
            f_min=self.eeg_min_frequency,
            f_max=self.eeg_max_frequency,
            n_mels=self.eeg_spectrogram_frequencies,
            power=spectrogram_power,
            normalized=False,
        )

        ##################################
        ##################################
        # ECGs
        ##################################
        ##################################
        self.ecg_channels = ecg_channels
        self.ecg_sampling_rate = ecg_sampling_rate

        self.ecg_spectrogram_n_fft = find_closest_power_of_2(self.ecg_sampling_rate)
        self.ecg_spectrogram_frequencies = n_mels_ecgs
        self.ecg_spectrogram_kernel_size = floor(self.ecg_sampling_rate * 0.25)
        self.ecg_spectrogram_kernel_stride = self.ecg_spectrogram_kernel_size // 2
        # formula taken from https://stackoverflow.com/questions/42821425/define-an-interval-for-frequency-of-stft-outputs
        self.ecg_min_frequency, self.ecg_max_frequency = 5, 60
        self.ecg_frequency_resolution: float = (
            self.ecg_sampling_rate / self.ecg_spectrogram_n_fft
        )
        self.ecg_samples = self.ecg_sampling_rate * ceil(self.seconds)
        self.ecg_spectrogram_samples = (
            ceil(self.ecg_samples / self.ecg_spectrogram_kernel_stride) + 1
        )
        self.ecg_mel_spectrogrammer = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.ecg_sampling_rate,
            n_fft=self.ecg_spectrogram_n_fft,
            win_length=self.ecg_spectrogram_kernel_size,
            hop_length=self.ecg_spectrogram_kernel_stride,
            f_min=self.ecg_min_frequency,
            f_max=self.ecg_max_frequency,
            n_mels=self.ecg_spectrogram_frequencies,
            power=spectrogram_power,
            normalized=False,
        )

        ##################################
        ##################################
        # OPTIMIZER
        ##################################
        ##################################
        self.learning_rate_ecgs = learning_rate_ecgs
        self.learning_rate_eegs = learning_rate_eegs
        self.learning_rate_disc = learning_rate_disc

        ##################################
        ##################################
        # NEURAL NETWORK
        # PARAMETERS
        ##################################
        ##################################
        self.layers_for_ecgs = layers_for_ecgs
        self.layers_for_eegs = layers_for_eegs
        self.h_dim = h_dim
        self.patches_size = patches_size
        self.padding_ecg = self.get_padding_for_patches(
            width=self.ecg_spectrogram_samples,
            height=self.ecg_spectrogram_frequencies,
            kernel_size=self.patches_size,
        )
        self.padding_eeg = self.get_padding_for_patches(
            width=self.eeg_spectrogram_samples,
            height=self.eeg_spectrogram_frequencies,
            kernel_size=self.patches_size,
        )
        self.num_ecg_patches = (
            (self.ecg_spectrogram_samples + self.padding_ecg[0] + self.padding_ecg[1])
            // self.patches_size
        ) * (
            (
                self.ecg_spectrogram_frequencies
                + self.padding_ecg[2]
                + self.padding_ecg[3]
            )
            // self.patches_size
        )
        self.num_eeg_patches = (
            (self.eeg_spectrogram_samples + self.padding_eeg[0] + self.padding_eeg[1])
            // self.patches_size
        ) * (
            (
                self.eeg_spectrogram_frequencies
                + self.padding_eeg[2]
                + self.padding_eeg[3]
            )
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
        self.eegs_encoder = self.build_encoder(
            in_channels=self.eeg_channels,
            patches_size=self.patches_size,
            h_dim=self.h_dim,
            padding=self.padding_eeg,
            # layers=self.layers_for_eegs,
            layers=self.layers_for_ecgs,
            add_reasoner=True,
            dropout=dropout,
        )
        # self.reasoner = self.build_classifier(in_features=self.h_dim, dropout=dropout)
        self.ecgs_encoder = self.build_encoder(
            in_channels=self.ecg_channels,
            patches_size=self.patches_size,
            h_dim=self.h_dim,
            padding=self.padding_ecg,
            layers=self.layers_for_ecgs,
            dropout=dropout,
        )

        self.ecgs_decoder = self.build_decoder(
            out_channels=self.ecg_channels,
            out_frequencies=self.ecg_spectrogram_frequencies
            + self.padding_ecg[2]
            + self.padding_ecg[3],
            out_length=self.ecg_spectrogram_samples
            + self.padding_ecg[0]
            + self.padding_ecg[1],
            h_dim=self.h_dim,
            padding=self.padding_ecg,
            layers=self.layers_for_ecgs,
            patches_size=self.patches_size,
            dropout=dropout,
        )
        self.latent_discriminator = self.build_classifier(
            in_features=self.h_dim,
            out_features=1,
            use_batchnorm=True,
            use_minibatch_discrimination=False,
        )

        self.save_hyperparameters(
            ignore=["eeg_mel_spectrogrammer", "ecg_mel_spectrogrammer"]
        )

    def configure_optimizers(self):
        if self.architecture == "full":
            # optimizers
            opt_ecgs = torch.optim.AdamW(
                [
                    {"params": self.ecgs_encoder.parameters()},
                    {"params": self.ecgs_decoder.parameters()},
                ],
                lr=self.learning_rate_ecgs,
            )
            opt_eegs = torch.optim.AdamW(
                [
                    {"params": self.eegs_encoder.parameters()},
                    # {"params": self.reasoner.parameters()},
                ],
                lr=self.learning_rate_eegs,
            )
            opt_disc = torch.optim.AdamW(
                [{"params": self.latent_discriminator.parameters()}],
                lr=self.learning_rate_disc,
            )
            # schedulers
            sch_ecgs = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt_ecgs, T_0=50, eta_min=5e-6, verbose=False
            )
            sch_eegs = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt_eegs, T_0=50, eta_min=5e-6, verbose=False
            )
            sch_disc = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt_disc, T_0=50, eta_min=1e-5, verbose=False
            )
            return [opt_ecgs, opt_eegs, opt_disc], [
                {"scheduler": sch_ecgs, "interval": "step"},
                {"scheduler": sch_eegs, "interval": "step"},
                {"scheduler": sch_disc, "interval": "step"},
            ]
        elif self.architecture == "simple":
            opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate_eegs)
            sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt, T_0=10, eta_min=1e-5, verbose=False
            )
            return [opt], {"scheduler": sch, "interval": "step"}

    def build_encoder(
        self,
        in_channels,
        layers,
        h_dim,
        padding=(0, 0, 0, 0),
        patches_size: Optional[int] = None,
        dropout: float = 0.0,
        add_reasoner: bool = False,
    ):
        assert 0 <= dropout < 1
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
            "pos_embeddings": nn.Embedding(num_embeddings=2048, embedding_dim=h_dim),
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
        if add_reasoner:
            modules["reasoner"] = self.build_classifier(
                in_features=h_dim,
                out_features=h_dim,
                dropout=0.0,
                use_batchnorm=False,
                use_minibatch_discrimination=False,
            ).classifier
        modules = nn.ModuleDict(modules)
        return modules

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
    ):
        assert 0 <= dropout < 1
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
                nn.Sigmoid(),
            ),
            "pos_embeddings": nn.Embedding(num_embeddings=2048, embedding_dim=h_dim),
        }
        return nn.ModuleDict(modules)

    def build_classifier(
        self,
        in_features,
        out_features,
        dropout=0.1,
        use_batchnorm: bool = False,
        use_minibatch_discrimination: bool = False,
    ):
        return nn.ModuleDict(
            {
                "classifier": nn.Sequential(
                    nn.Linear(in_features, in_features * 4),
                    self.activation_fn,
                    nn.BatchNorm1d(in_features * 4) if use_batchnorm else nn.Identity(),
                    MinibatchDiscrimination(
                        in_features=in_features * 4,
                        out_features=in_features,
                        kernel_dims=5,
                    )
                    if use_minibatch_discrimination
                    else nn.Identity(),
                    nn.Dropout(dropout),
                    nn.Linear(
                        in_features * (5 if use_minibatch_discrimination else 4),
                        out_features,
                    ),
                )
            }
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
            opt_ecgs, opt_eegs, opt_disc = self.optimizers()
            sch_ecgs, sch_eegs, sch_disc = self.lr_schedulers()

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
                self.ecg_spectrogram_frequencies,
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
                self.eeg_spectrogram_frequencies,
                self.eeg_spectrogram_samples,
            ),
        )

        ##################################
        ##################################
        # DISCRIMINATOR
        ##################################
        ##################################
        if self.training and has_trainer:
            opt_disc.zero_grad(set_to_none=True)

        # trains the discriminator on true samples
        with profiler.record_function("discriminator true".upper()):
            # generates true samples
            real_samples, real_samples_states = self.encode_mel_spectrogram(
                mel_spectrogram=ecgs_mel_spectrogram_gt,
                encoder=self.ecgs_encoder,
                pooled_outputs=True,
                return_hidden_states=True,
                expected_shape=(
                    batch_size,
                    self.h_dim,
                ),
            )
            # computes the loss
            logits_D_true = self.discriminate_latent(
                real_samples.detach(),
                discriminator=self.latent_discriminator,
            )
            outs["loss_D_true"] = F.binary_cross_entropy_with_logits(
                input=logits_D_true, target=torch.zeros_like(logits_D_true)
            )
            outs["acc_D_true"] = torchmetrics.functional.accuracy(
                preds=logits_D_true,
                target=torch.zeros_like(logits_D_true),
                task="binary",
            )
            if self.training and has_trainer:
                self.manual_backward(outs["loss_D_true"])

        # trains the discriminator on fake samples
        with profiler.record_function("discriminator fake".upper()):
            # generates fake samples
            fake_samples, fake_samples_states = self.encode_mel_spectrogram(
                mel_spectrogram=eegs_mel_spectrogram,
                encoder=self.eegs_encoder,
                pooled_outputs=True,
                use_reasoner=False,
                return_hidden_states=True,
                expected_shape=(
                    batch_size,
                    self.h_dim,
                ),
            )
            # fake_samples = self.reasoner.reasoner(fake_samples)
            # fake_samples = F.tanh(fake_samples)
            # computes the loss
            logits_D_fake = self.discriminate_latent(
                fake_samples.detach(), discriminator=self.latent_discriminator
            )
            outs["loss_D_fake"] = F.binary_cross_entropy_with_logits(
                input=logits_D_fake, target=torch.ones_like(logits_D_fake)
            )
            outs["acc_D_fake"] = torchmetrics.functional.accuracy(
                preds=logits_D_fake,
                target=torch.ones_like(logits_D_fake),
                task="binary",
            )
            if self.training and has_trainer:
                self.manual_backward(outs["loss_D_fake"])
                self.clip_gradients(
                    opt_disc,
                    gradient_clip_val=1.0,
                    gradient_clip_algorithm="norm",
                )
                opt_disc.step()
                sch_disc.step(
                    self.current_epoch + batch_idx / self.get_dataloader_length()
                )
                outs["lr_disc"] = sch_disc.get_last_lr()[-1]

            ##################################
            ##################################
            # ECGs
            ##################################
            ##################################
            if self.training and has_trainer:
                opt_ecgs.zero_grad(set_to_none=True)

            # decodes the ecg latent into an ecg mel spectrogram
            with profiler.record_function("ecg latent to ecg mel".upper()):
                ecgs_mel_spectrogram_rec = self.decode_mel_spectrogram(
                    latent=real_samples,
                    decoder=self.ecgs_decoder,
                    # out_length=self.ecg_spectrogram_samples,
                    out_length=self.num_ecg_patches,
                    expected_shape=(
                        batch_size,
                        self.ecg_channels,
                        self.ecg_spectrogram_frequencies,
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
                sch_ecgs.step(self.current_epoch + batch_idx / self.get_dataloader_length())
                outs["lr_ecgs"] = sch_ecgs.get_last_lr()[-1]

        ##################################
        ##################################
        # EEGs
        ##################################
        ##################################
        if self.training and has_trainer:
            opt_eegs.zero_grad(set_to_none=True)

        # computes the reconstruction loss between latent representations
        outs["loss_R"] = self.reconstruction_loss(
            pred=fake_samples,
            gt=real_samples.detach(),
            norm=2,
        )
        ws = torch.softmax(
            torch.arange(len(real_samples_states), dtype=torch.float), dim=0
        )
        outs["loss_G_dist"] = sum(
            [
                self.reconstruction_loss(
                    pred=fake_samples_states[i],
                    gt=real_samples_states[i].detach(),
                    norm=2,
                )
                * ws[i]
                for i in range(len(real_samples_states))
            ]
        )
        # computes the discriminative loss
        logits_D_for_G = self.discriminate_latent(
            fake_samples, discriminator=self.latent_discriminator
        )
        outs["loss_G_disc"] = F.binary_cross_entropy_with_logits(
            input=logits_D_for_G, target=torch.zeros_like(logits_D_for_G)
        )

        # decodes the eeg latent into an ecg mel spectrogram
        # with profiler.record_function("eeg latent to eeg mel".upper()):
        #     # with torch.no_grad():
        #     ecgs_mel_spectrogram_gen = self.decode_mel_spectrogram(
        #         latent=fake_samples,
        #         decoder=self.ecgs_decoder,
        #         out_length=self.num_ecg_patches,
        #         expected_shape=(
        #             batch_size,
        #             self.ecg_channels,
        #             self.spectrogram_frequencies,
        #             self.ecg_spectrogram_samples,
        #         ),
        #     )
        #     outs["loss_G_eeg"] = self.reconstruction_loss(
        #         ecgs_mel_spectrogram_gen, ecgs_mel_spectrogram_gt, norm=2
        #     )

        if self.training and has_trainer:
            # self.manual_backward(outs["loss_R"] + outs["loss_G_disc"] * 0.01)
            self.manual_backward(outs["loss_G_dist"] * 2 + outs["loss_R"] + outs["loss_G_disc"] * 0.01)
            self.clip_gradients(
                opt_eegs,
                gradient_clip_val=1.0,
                gradient_clip_algorithm="norm",
            )
            opt_eegs.step()
            sch_eegs.step(self.current_epoch + batch_idx / self.get_dataloader_length())
            outs["lr_eegs"] = sch_eegs.get_last_lr()[-1]

        # asserts that no loss is corrupted
        for loss_name, loss in [
            (k, v) for k, v in outs.items() if k.startswith("loss")
        ]:
            assert not torch.isnan(
                loss
            ).any(), f"{loss_name} has become None at epoch {self.current_epoch} and step {batch_idx}"
        outs["loss"] = sum(v for k, v in outs.items() if k.startswith("loss"))

        # LOGGING
        if (
            has_trainer
            and batch_idx == 0
            and (self.training or self.current_epoch != 0)
        ):
            with torch.no_grad():
                ecgs_mel_spectrogram_gen = self.decode_mel_spectrogram(
                    latent=fake_samples,
                    decoder=self.ecgs_decoder,
                    out_length=self.num_ecg_patches,
                    expected_shape=(
                        batch_size,
                        self.ecg_channels,
                        self.ecg_spectrogram_frequencies,
                        self.ecg_spectrogram_samples,
                    ),
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
            outs["images/latent"] = wandb.Image(
                plot_reconstructed_spectrograms(
                    sg_pred=fake_samples.unsqueeze(1)[:4],
                    sg_gt=real_samples.unsqueeze(1)[:4],
                ),
                caption="reconstructed latent representation",
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
                print("wandb may not be the current logger")
        return outs
    
    def shared_step_simple(self, batch, batch_idx):
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
                self.ecg_spectrogram_frequencies,
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
                self.eeg_spectrogram_frequencies,
                self.eeg_spectrogram_samples,
            ),
        )

        ##################################
        ##################################
        # END-to-END
        # MODEL
        ##################################
        ##################################
        if self.training and has_trainer:
            opt.zero_grad(set_to_none=True)
            
        # generates fake samples
        fake_samples, fake_samples_states = self.encode_mel_spectrogram(
            mel_spectrogram=eegs_mel_spectrogram,
            encoder=self.eegs_encoder,
            pooled_outputs=True,
            use_reasoner=False,
            return_hidden_states=True,
            expected_shape=(
                batch_size,
                self.h_dim,
            ),
        )

        # decodes the ecg latent into an ecg mel spectrogram
        with profiler.record_function("ecg latent to ecg mel".upper()):
            ecgs_mel_spectrogram_gen = self.decode_mel_spectrogram(
                latent=fake_samples,
                decoder=self.ecgs_decoder,
                # out_length=self.ecg_spectrogram_samples,
                out_length=self.num_ecg_patches,
                expected_shape=(
                    batch_size,
                    self.ecg_channels,
                    self.ecg_spectrogram_frequencies,
                    self.ecg_spectrogram_samples,
                ),
            )
            # computes the reconstruction loss between latent representations
            outs["loss"] = self.reconstruction_loss(
                ecgs_mel_spectrogram_gen, ecgs_mel_spectrogram_gt, norm=2
            )

        if self.training and has_trainer:
            self.manual_backward(outs["loss"])
            self.clip_gradients(
                opt,
                gradient_clip_val=1.0,
                gradient_clip_algorithm="norm",
            )
            opt.step()
            sch.step(self.current_epoch + batch_idx / self.get_dataloader_length())
            outs["lr_ecgs"] = sch.get_last_lr()[-1]

        # asserts that no loss is corrupted
        for loss_name, loss in [
            (k, v) for k, v in outs.items() if k.startswith("loss")
        ]:
            assert not torch.isnan(
                loss
            ).any(), f"{loss_name} has become None at epoch {self.current_epoch} and step {batch_idx}"
        outs["loss"] = sum(v for k, v in outs.items() if k.startswith("loss"))

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
                print("wandb may not be the current logger")
        return outs

    def forward(self, x):
        pass

    def encode_mel_spectrogram(
        self,
        mel_spectrogram,
        encoder,
        limit_outputs: bool = False,
        pooled_outputs: bool = False,
        use_reasoner: bool = False,
        return_hidden_states: bool = False,
        expected_shape=None,
    ):
        latent_tokens = encoder.in_reshaper(mel_spectrogram)
        if pooled_outputs:
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
        latent_tokens = sum([latent_tokens, pos_embeddings])
        states = []
        for layer in encoder.encoder.layers:
            latent_tokens = layer(latent_tokens)
            states.append(latent_tokens)
        latent_tokens_transformed = latent_tokens
        if pooled_outputs:
            latent_tokens_transformed = latent_tokens_transformed[:, 0]
        if use_reasoner:
            latent_tokens_transformed = encoder.reasoner(latent_tokens_transformed)
        if limit_outputs:
            latent_tokens_transformed = F.tanh(latent_tokens_transformed)
        # OPTIONAL: does a shape check
        if expected_shape:
            self.shape_check(
                tensor=latent_tokens_transformed,
                expected_shape=expected_shape,
            )
        if return_hidden_states:
            return latent_tokens_transformed, states
        else:
            return latent_tokens_transformed

    def discriminate_latent(
        self,
        latent_tokens,
        discriminator,
    ):
        logits = discriminator.classifier(latent_tokens)
        return logits

    def decode_mel_spectrogram(self, latent, decoder, out_length, expected_shape=None):
        pos_embeddings = decoder.pos_embeddings(
            torch.arange(out_length, device=latent.device)
            .unsqueeze(0)
            .repeat(latent.shape[0], 1)
        )
        # latent_tokens_tgt = torch.cat([latent_tokens_tgt, pos_embeddings_tgt], dim=-1)
        latent_tgt = sum([latent.unsqueeze(1), pos_embeddings])
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
        from torchvision.transforms import Normalize

        mel_spectrogram = Normalize(mean=0.5, std=0.5)(mel_spectrogram)
        min, max = torch.amin(mel_spectrogram, dim=(2, 3), keepdim=True), torch.amax(
            mel_spectrogram, dim=(2, 3), keepdim=True
        )
        mel_spectrogram = (mel_spectrogram - min) / (max - min)
        if expected_shape:
            self.shape_check(
                tensor=mel_spectrogram,
                expected_shape=expected_shape,
            )
        # mel_spectrogram = mel_spectrogram + 1e-6
        return mel_spectrogram

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
        if self.architecture == "full":
            return self.shared_step(batch, batch_idx)
        elif self.architecture == "simple":
            return self.shared_step_simple(batch, batch_idx)
        
    def validation_step(self, batch, batch_idx):
        if self.architecture == "full":
            return self.shared_step(batch, batch_idx)
        elif self.architecture == "simple":
            return self.shared_step_simple(batch, batch_idx)

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

    model = EEG2ECGModel(
        eeg_channels=14,
        ecg_channels=2,
        eeg_sampling_rate=128,
        ecg_sampling_rate=256,
        seconds=2,
        layers_for_ecgs=4,
        architecture="simple",
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
        # sample_outs = model.shared_step(sample_batch, 0)
        sample_outs = model.shared_step_simple(sample_batch, 0)
    print(
        prof.key_averages().table(
            sort_by="cuda_time",
            row_limit=10,
        )
    )
