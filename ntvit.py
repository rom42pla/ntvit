from typing import Any, Dict, Iterable, Optional, Tuple, Union
from math import ceil, floor, log2
import einops
from einops.layers.torch import Rearrange
from torch import nn
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.autograd.profiler as profiler
import torchaudio
import torchmetrics
from thop import profile
import wandb

from losses import centroid_loss, custom_loss, energy_loss, shape_loss, ssim_loss

try:
    from plots import (
        plot_reconstructed_spectrograms,
        plot_reconstructed_fmris,
    )
    from models.sdtw import SoftDTW
except:
    print("error loading libraries")


class Unfolder(nn.Module):
    def __init__(self, patches_size, padding):
        super().__init__()
        self.patches_size = patches_size
        self.padding = padding

    def forward(self, input):
        padded_input = F.pad(input, pad=self.padding)
        patches = padded_input.unfold(2, self.patches_size, self.patches_size).unfold(
            3, self.patches_size, self.patches_size
        )
        patches = einops.rearrange(patches, "b c p1 p2 h w -> b (p1 p2) (c h w)")
        return patches


class Folder(nn.Module):
    def __init__(self, patches_size, padding, out_height, out_length):
        super().__init__()
        self.patches_size = patches_size
        self.padding = padding
        self.out_height = out_height
        self.out_length = out_length

    def forward(self, input):
        rearranged_input = einops.rearrange(input, "b w c -> b c w")
        folded_input = F.fold(
            rearranged_input,
            output_size=(self.out_height, self.out_length),
            kernel_size=self.patches_size,
            stride=self.patches_size,
        )
        folded_input = folded_input[
            :,
            :,
            self.padding[2] : -self.padding[3] if self.padding[3] > 0 else None,
            self.padding[0] : -self.padding[1] if self.padding[1] > 0 else None,
        ]
        return folded_input


class LambdaModule(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        import types

        assert type(lambd) is types.LambdaType
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.sigma
            return x + noise
        else:
            return x


class NTViT(pl.LightningModule):
    def __init__(
        self,
        fmris_shape: Tuple[int, int, int],
        eegs_seconds: Union[int, float],
        eegs_channels: int,
        eegs_sampling_rate: int,
        normalized_fmris: bool,
        fmris_downsampling_factor: int = 3,
        spectrogram_scale: float = 2,
        use_domain_matching: bool = False,
        use_discriminator: bool = False,
        alpha_disc: float = 1,
        n_mels: int = 16,
        eegs_spectrogram_power: int = 2,
        eeg_patches_size: int = 8,
        fmri_patches_size: int = 8,
        layers: int = 2,
        h_dim: int = 256,
        learning_rate: float = 1e-4,
        use_lr_scheduler: bool = False,
        activation: str = "gelu",
        use_kl_loss: bool = True,
        use_cm_loss: bool = False,
        use_disp_loss: bool = False,
        # regularization
        dropout: float = 0.2,
        input_noise: float = 0.2,
        weight_decay: float = 1e-2,
        plot_images: bool = False,
        logger_prefix: Optional[str] = None,
    ):
        super(NTViT, self).__init__()

        self.automatic_optimization = False
        assert (logger_prefix is None) or isinstance(logger_prefix, str)
        self.logger_prefix = logger_prefix
        assert isinstance(plot_images, bool)
        self.plot_images = plot_images

        ##################################
        ##################################
        # Losses
        ##################################
        ##################################
        assert isinstance(use_kl_loss, bool)
        self.use_kl_loss = use_kl_loss
        assert isinstance(use_cm_loss, bool)
        self.use_cm_loss = use_cm_loss
        assert isinstance(use_disp_loss, bool)
        self.use_disp_loss = use_disp_loss

        ##################################
        ##################################
        # EEGs
        ##################################
        ##################################
        self.seconds = eegs_seconds
        self.eegs_channels = eegs_channels
        self.eegs_sampling_rate = eegs_sampling_rate
        self.spectrogram_power = eegs_spectrogram_power
        assert spectrogram_scale > 0
        self.spectrogram_scale = spectrogram_scale

        def find_closest_power_of_2(x):
            for i in range(100):
                if 2**i >= x:
                    break
            return 2**i

        self.eegs_spectrogram_n_fft = find_closest_power_of_2(self.eegs_sampling_rate)
        self.eegs_spectrogram_frequencies = n_mels
        self.eegs_spectrogram_kernel_size = floor(self.eegs_sampling_rate / 16)
        self.eegs_spectrogram_kernel_stride = round(
            self.eegs_spectrogram_kernel_size / self.spectrogram_scale
        )
        self.eegs_min_frequency, self.eegs_max_frequency = 1, 45
        self.eegs_frequency_resolution: float = (
            self.eegs_sampling_rate / self.eegs_spectrogram_n_fft
        )
        self.eegs_samples = ceil(self.eegs_sampling_rate * self.seconds)
        self.eegs_mel_spectrogrammer = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.eegs_sampling_rate,
            n_fft=self.eegs_spectrogram_n_fft,
            win_length=self.eegs_spectrogram_kernel_size,
            hop_length=self.eegs_spectrogram_kernel_stride,
            f_min=self.eegs_min_frequency,
            f_max=self.eegs_max_frequency,
            n_mels=self.eegs_spectrogram_frequencies,
            power=self.spectrogram_power,
            normalized=False,
        )
        with torch.no_grad():
            self.eegs_spectrogram_samples = self.eegs_mel_spectrogrammer(
                torch.randn(1, self.eegs_channels, self.eegs_samples)
            ).shape[-1]
        assert 0 <= input_noise < 1
        self.input_noise = input_noise

        ##################################
        ##################################
        # fMRI
        ##################################
        ##################################
        assert isinstance(
            fmris_shape, (list, tuple)
        ), f"fmris_shape must be an iterable, got {fmris_shape}"
        assert (
            len(fmris_shape) == 3
        ), f"fmris_shape must have three elements (z, y, x), got {fmris_shape}"
        assert all(
            [isinstance(v, int) for v in fmris_shape]
        ), f"fmris_shape must have three ints, got {fmris_shape}"
        self.fmris_shape = fmris_shape
        assert isinstance(normalized_fmris, bool)
        self.normalized_fmris = normalized_fmris
        assert fmris_downsampling_factor >= 1
        self.fmris_downsampling_factor = fmris_downsampling_factor
        if self.fmris_downsampling_factor > 1:
            self.fmris_shape = tuple(F.interpolate(torch.empty([1, 1, *self.fmris_shape]), scale_factor=1/self.fmris_downsampling_factor)[0, 0].shape)

        ##################################
        ##################################
        # OPTIMIZER
        ##################################
        ##################################
        assert learning_rate > 0
        self.learning_rate = learning_rate
        assert isinstance(use_lr_scheduler, bool)
        self.use_lr_scheduler = use_lr_scheduler
        assert weight_decay >= 0
        self.weight_decay = weight_decay

        ##################################
        ##################################
        # NEURAL NETWORK
        # PARAMETERS
        ##################################
        ##################################
        self.layers = layers
        self.h_dim = h_dim
        self.eegs_patches_size = eeg_patches_size
        self.eegs_padding = self.get_padding_for_patches(
            width=self.eegs_spectrogram_samples,
            height=self.eegs_spectrogram_frequencies,
            kernel_size=self.eegs_patches_size,
        )
        self.fmris_patches_size = fmri_patches_size
        self.fmris_padding = self.get_padding_for_patches(
            width=self.fmris_shape[2],
            height=self.fmris_shape[1],
            kernel_size=self.fmris_patches_size,
        )
        self.num_eeg_patches = (
            (
                self.eegs_spectrogram_samples
                + self.eegs_padding[0]
                + self.eegs_padding[1]
            )
            // self.eegs_patches_size
        ) * (
            (
                self.eegs_spectrogram_frequencies
                + self.eegs_padding[2]
                + self.eegs_padding[3]
            )
            // self.eegs_patches_size
        )
        self.num_fmris_patches = (
            (self.fmris_shape[2] + self.fmris_padding[0] + self.fmris_padding[1])
            // self.fmris_patches_size
        ) * (
            (self.fmris_shape[1] + self.fmris_padding[2] + self.fmris_padding[3])
            // self.fmris_patches_size
        )

        ####################################
        # MODULES
        ####################################
        if activation == "softplus":
            self.activation_module = nn.Softplus()
            self.activation_functional = F.softplus
        elif activation == "gelu":
            self.activation_module = nn.GELU()
            self.activation_functional = F.gelu
        elif activation == "selu":
            self.activation_module = nn.SELU()
            self.activation_functional = F.selu
        self.eegs_encoder = self.build_encoder(
            in_channels=self.eegs_channels,
            patches_size=self.eegs_patches_size,
            h_dim=self.h_dim,
            padding=self.eegs_padding,
            layers=self.layers,
            dropout=dropout,
            input_noise=self.input_noise,
        )
        self.fmris_decoder = self.build_decoder(
            out_channels=self.fmris_shape[0],
            out_frequencies=self.fmris_shape[1]
            + self.fmris_padding[2]
            + self.fmris_padding[3],
            out_length=self.fmris_shape[2]
            + self.fmris_padding[0]
            + self.fmris_padding[1],
            h_dim=self.h_dim,
            padding=self.fmris_padding,
            layers=self.layers,
            patches_size=self.fmris_patches_size,
            dropout=dropout,
        )

        # knowledge distillation
        assert isinstance(use_domain_matching, bool)
        self.use_domain_matching = use_domain_matching
        if self.use_domain_matching:
            self.fmris_encoder = self.build_encoder(
                in_channels=self.fmris_shape[0],
                patches_size=self.fmris_patches_size,
                h_dim=self.h_dim,
                padding=self.fmris_padding,
                layers=self.layers,
                dropout=dropout,
                input_noise=self.input_noise,
            )

        # discriminator
        assert isinstance(use_discriminator, bool)
        self.use_discriminator = use_discriminator
        if self.use_discriminator:
            assert alpha_disc > 0
            self.alpha_disc = alpha_disc
            self.fmris_discriminator_encoder = self.build_encoder(
                in_channels=self.fmris_shape[0],
                patches_size=self.fmris_patches_size,
                h_dim=self.h_dim,
                padding=self.fmris_padding,
                layers=self.layers,
                dropout=dropout,
                input_noise=self.input_noise,
            )
            self.fmris_discriminator = self.build_classifier(
                in_features=self.h_dim,
                out_features=1,
                dropout=dropout,
            )

        ###################################
        # count macs and parameters
        ##################################
        class DummyModel(nn.Module):
            def __init__(self, original_model):
                super(DummyModel, self).__init__()
                self.original_model = original_model

            def forward(self, x):
                return self.original_model.shared_step(x, 0)

        with torch.no_grad():
            try:
                macs, params = profile(
                    DummyModel(self).to(self.device),
                    (
                        {
                            "eegs": torch.randn(
                                [1, self.eegs_channels, self.eegs_samples],
                                device=self.device,
                            ),
                            "fmris": torch.rand(
                                [1, *fmris_shape],
                                device=self.device,
                            ),
                        },
                    ),
                )
                self.macs = macs / 1e9
                self.n_params = params / 1e6
            except Exception as e:
                print("error with thop", e)
                self.macs, self.n_params = 0, 0

        self.save_hyperparameters(ignore=["eegs_mel_spectrogrammer"])

    def configure_optimizers(self):
        # optimizers
        gen_modules = [self.eegs_encoder, self.fmris_decoder]
        if self.use_domain_matching:
            gen_modules += [self.fmris_encoder]
        opt = torch.optim.AdamW(
            [p for m in gen_modules for p in m.parameters()],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt,
            T_0=10,
            eta_min=5e-5,
            verbose=False,
        )
        if self.use_discriminator:
            disc_modules = [self.fmris_discriminator_encoder, self.fmris_discriminator]
            opt_disc = torch.optim.AdamW(
                [p for m in disc_modules for p in m.parameters()],
                lr=self.learning_rate * 1e-1,
                weight_decay=self.weight_decay,
            )
            sch_disc = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt_disc,
                T_0=10,
                eta_min=5e-5,
                verbose=False,
            )
            opts = [opt, opt_disc]
            schs = [{"scheduler": sch, "interval": "step"} for sch in [sch, sch_disc]]
        else:
            opts = [opt]
            schs = [
                {"scheduler": sch, "interval": "step"},
            ]
        return opts, schs

    def build_encoder(
        self,
        in_channels,
        layers,
        h_dim,
        padding=(0, 0, 0, 0),
        patches_size: Optional[int] = None,
        dropout: float = 0.0,
        input_noise: float = 0.0,
        architecture="full",
    ):
        assert architecture in {"full", "conv"}
        assert 0 <= dropout < 1
        assert 0 <= input_noise < 1
        modules = {
            "in_reshaper": nn.Sequential(
                GaussianNoise(input_noise),
                Unfolder(patches_size=patches_size, padding=padding),
                # nn.ZeroPad2d(padding),
                # nn.Unfold(kernel_size=patches_size, stride=patches_size),
                # Rearrange("b c w -> b w c"),
                nn.Linear(in_channels * patches_size**2, 2048),
                nn.LayerNorm(2048),
                self.activation_module,
                nn.Linear(2048, h_dim),
            ),
            "encoder": nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=h_dim,
                    nhead=8,
                    # dim_feedforward=h_dim * 4,
                    dim_feedforward=2048,
                    activation=self.activation_functional,
                    batch_first=True,
                    dropout=dropout,
                ),
                num_layers=layers,
            ),
            "pos_embeddings": nn.Embedding(num_embeddings=2048, embedding_dim=h_dim),
            "learned_tokens": nn.Embedding(num_embeddings=1, embedding_dim=h_dim),
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
    ):
        assert 0 <= dropout < 1
        modules = {
            "decoder": nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=h_dim,
                    nhead=8,
                    # dim_feedforward=h_dim * 4,
                    dim_feedforward=2048,
                    activation=self.activation_functional,
                    batch_first=True,
                    dropout=dropout,
                ),
                num_layers=layers,
            ),
            "out_reshaper": nn.Sequential(
                nn.Linear(h_dim, 2048),
                nn.LayerNorm(2048),
                self.activation_module,
                nn.Linear(2048, out_channels * patches_size**2),
                # Rearrange("b w c -> b c w"),
                # nn.Fold(
                #     output_size=(out_frequencies, out_length),
                #     kernel_size=patches_size,
                #     stride=patches_size,
                # ),
                # LambdaModule(
                #     lambda x: x[
                #         :,
                #         :,
                #         padding[2] : -padding[3] if padding[3] > 0 else None,
                #         padding[0] : -padding[1] if padding[1] > 0 else None,
                #     ]
                # ),
                Folder(
                    patches_size=patches_size,
                    padding=padding,
                    out_height=out_frequencies,
                    out_length=out_length,
                ),
                Rearrange("b z y x -> b () z y x"),
                nn.Conv3d(
                    in_channels=1,
                    out_channels=1,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                Rearrange("b () z y x -> b z y x"),
                # nn.Sigmoid() if self.normalized_fmris else nn.Softplus(),
                nn.Sigmoid()
                if self.normalized_fmris
                else LambdaModule(lambda x: torch.exp(x)),
                # else nn.Softplus(),
            ),
            "pos_embeddings": nn.Embedding(num_embeddings=2048, embedding_dim=h_dim),
        }
        return nn.ModuleDict(modules)

    def build_classifier(
        self,
        in_features,
        out_features,
        dropout=0.1,
        use_batchnorm: bool = True,
        use_minibatch_discrimination: bool = False,
    ):
        return nn.ModuleDict(
            {
                "classifier": nn.Sequential(
                    nn.Linear(in_features, in_features * 4),
                    self.activation_module,
                    nn.Dropout(dropout),
                    nn.Linear(
                        in_features * 4,
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
            if self.use_discriminator:
                opt, opt_disc = self.optimizers()
                sch, sch_disc = self.lr_schedulers()
            else:
                opt = self.optimizers()
                sch = self.lr_schedulers()

        outs: Dict[str, Any] = self(batch)
        if self.use_discriminator:
            if self.training and has_trainer:
                opt_disc.zero_grad(set_to_none=True)
            # all-real phase
            outs["D_fmris_real"] = self.discriminate(
                t=outs["fmris_gt"].detach(),
                encoder=self.fmris_discriminator_encoder,
                discriminator=self.fmris_discriminator,
            )
            outs["loss_D_real_fmris"] = self.discriminator_loss(
                logits=outs["D_fmris_real"]["logits"], labels=1
            )
            if self.training and has_trainer:
                self.manual_backward(
                    sum(v for k, v in outs.items() if k.startswith("loss_D_real")),
                )
            # all-fake phase
            outs["D_fmris_fake"] = self.discriminate(
                t=outs["fmris_rec"].detach(),
                encoder=self.fmris_discriminator_encoder,
                discriminator=self.fmris_discriminator,
            )
            outs["loss_D_fake_fmris"] = self.discriminator_loss(
                logits=outs["D_fmris_fake"]["logits"], labels=0
            )
            if self.training and has_trainer:
                self.manual_backward(
                    sum(v for k, v in outs.items() if k.startswith("loss_D_fake"))
                    # sum(v for k, v in outs.items() if k.startswith("loss_D"))
                )
                self.clip_gradients(
                    opt_disc,
                    gradient_clip_val=1.0,
                    gradient_clip_algorithm="norm",
                )
                opt_disc.step()
                if self.use_lr_scheduler:
                    sch_disc.step(
                        self.current_epoch + batch_idx / self.get_dataloader_length()
                    )
        if self.training and has_trainer:
            opt.zero_grad(set_to_none=True)
        if self.use_discriminator:
            outs["D_for_G"] = self.discriminate(
                t=outs["fmris_rec"],
                encoder=self.fmris_discriminator_encoder,
                discriminator=self.fmris_discriminator,
            )
            outs["loss_G_disc"] = (
                self.discriminator_loss(
                    logits=outs["D_for_G"]["logits"],
                    labels=1,
                )
                * self.alpha_disc
            )
        # generator loss
        outs["loss_G_rec"] = self.reconstruction_loss(
            pred=torch.log1p(outs["fmris_rec"])
            if not self.normalized_fmris
            else outs["fmris_rec"],
            # pred=outs["fmris_rec"],
            gt=torch.log1p(outs["fmris_gt"])
            if not self.normalized_fmris
            else outs["fmris_gt"],
            norm=2,
        )
        if self.use_disp_loss:
            outs["loss_G_disp"] = custom_loss(
                pred=outs["fmris_rec"],
                gt=outs["fmris_gt"],
            )
        if self.use_cm_loss:
            outs["loss_G_cm"] = centroid_loss(
                pred=outs["fmris_rec"],
                gt=outs["fmris_gt"],
            )        
        if self.use_kl_loss:
            outs["loss_G_kl"] = sum([
                F.kl_div(
                    input=F.log_softmax(input, dim=1),
                    target=F.softmax(target, dim=1),
                    reduction="batchmean",
                ) * 1e-3
                for input, target in [(outs["fmris_rec"], outs["fmris_gt"]), 
                                      (outs["eegs_latent"], torch.randn_like(outs["eegs_latent"]))]
            ]) 
        # outs["loss_G_energy"] = energy_loss(
        #     pred=outs["fmris_rec"],
        #     gt=outs["fmris_gt"],
        # )
        # outs["loss_G_shape"] = shape_loss(
        #     pred=outs["fmris_rec"],
        #     gt=outs["fmris_gt"],
        # )
        # discriminative loss for the generator
        # outs["D_fmris_real_2"] = self.discriminate(
        #     t=outs["fmris_gt"].detach(),
        #     encoder=self.fmris_discriminator_encoder,
        #     discriminator=self.fmris_discriminator,
        # )

        if self.use_domain_matching:
            outs["fmris_latent"] = self.encode_mel_spectrogram(
                outs["fmris_gt"], encoder=self.fmris_encoder
            )
            outs["fmris_latent_rec"] = self.decode_mel_spectrogram(
                outs["fmris_latent"].detach(),
                decoder=self.fmris_decoder,
                out_length=self.num_fmris_patches,
            )
            outs["loss_G_fmris_rec"] = self.reconstruction_loss(
                pred=outs["fmris_latent_rec"],
                gt=outs["fmris_gt"].detach(),
                norm=2,
            )
            if self.use_kl_loss:
                outs["loss_G_dm_kl"] = sum([
                    F.kl_div(
                        input=F.log_softmax(outs["fmris_latent"], dim=1),
                        target=F.softmax(torch.randn_like(outs["fmris_latent"]), dim=1),
                        reduction="batchmean",
                    ) * 1e-3
                ]) 
            outs["loss_G_dm"] = self.reconstruction_loss(
                pred=outs["eegs_latent"],
                gt=outs["fmris_latent"].detach(),
                norm=2,
            )
        outs["loss_G"] = sum(v for k, v in outs.items() if k.startswith("loss_G"))
        if self.training and has_trainer:
            self.manual_backward(outs["loss_G"])
            self.clip_gradients(
                opt,
                gradient_clip_val=1.0,
                gradient_clip_algorithm="norm",
            )
            opt.step()
            if self.use_lr_scheduler:
                sch.step(self.current_epoch + batch_idx / self.get_dataloader_length())
                outs["lr"] = sch.get_last_lr()[-1]
            else:
                outs["lr"] = self.learning_rate
        # computes reconstruction metrics
        # if not self.normalized_fmris:
            # outs["fmris_gt"] = torch.expm1(outs["fmris_gt"])
            # outs["fmris_rec"] = torch.exp(outs["fmris_rec"])
        for metric_name, metric_fn in [
            ("mse", self.mse),
            ("rmse", self.rmse),
            ("mae", self.mae),
            ("cfv", self.cfv),
            ("psnr", self.psnr),
            ("ssim", self.ssim),
        ]:
            outs[metric_name] = metric_fn(outs["fmris_rec"], outs["fmris_gt"])

        if self.normalized_fmris:
            max_value = 1
        else:
            max_value = torch.max(
                outs["fmris_rec"].amax(dim=(1, 2, 3), keepdim=True),
                outs["fmris_gt"].amax(dim=(1, 2, 3), keepdim=True),
            )
            assert max_value.shape == (outs["fmris_rec"].shape[0], 1, 1, 1)
        outs[
            "ssim"
        ] = torchmetrics.functional.image.structural_similarity_index_measure(
            preds=outs["fmris_rec"] / max_value,
            target=outs["fmris_gt"] / max_value,
        )

        # asserts that no loss is corrupted
        for loss_name, loss in [
            (k, v) for k, v in outs.items() if k.startswith("loss")
        ]:
            assert not torch.isnan(
                loss
            ).any(), f"{loss_name} has become None at epoch {self.current_epoch} and step {batch_idx}"

        # LOGGING
        if hasattr(self, "macs"):
            outs["macs"] = self.macs
            outs["n_params"] = self.n_params
        if has_trainer:
            self.log_dict(
                {
                    f"{self.logger_prefix + '/' if self.logger_prefix else ''}{self.get_phase_name()}/{k}": v
                    for k, v in outs.items()
                    if (isinstance(v, torch.Tensor) and v.numel() == 1)
                    or isinstance(v, (float, int))
                },
                on_step=False,
                on_epoch=True,
            )
            if self.plot_images and batch_idx == 0:
                for i in range(1):
                    outs[f"images/fmris_pc_{i}"] = wandb.Image(
                        plot_reconstructed_fmris(
                            fmris_pred=outs["fmris_rec"][i],
                            fmris_gt=outs["fmris_gt"][i],
                            vmin=0,
                            vmax=1,
                            mode="pc",
                        ),
                        caption=f"generated fMRIs for sample {i}, point cloud",
                    )
                    outs[f"images/fmris_mip_{i}"] = wandb.Image(
                        plot_reconstructed_fmris(
                            fmris_pred=outs["fmris_rec"][i],
                            fmris_gt=outs["fmris_gt"][i],
                            vmin=0,
                            vmax=1,
                            mode="mip",
                        ),
                        caption=f"generated fMRIs for sample {i}, MIP",
                    )
                try:
                    wandb.log(
                        {
                            f"{self.logger_prefix + '/' if self.logger_prefix else ''}{self.get_phase_name()}/{k}": v
                            for k, v in outs.items()
                            if isinstance(v, wandb.Image)
                        }
                    )
                except:
                    print("wandb may not be the current logger")
            # outs[f"images/eegs_gt"] = wandb.Image(
            #     plot_reconstructed_spectrograms(
            #         sg_pred=outs["eegs_mel_spectrogram_gt"][0, :8],
            #         sg_gt=outs["eegs_mel_spectrogram_gt"][0, :8],
            #         vmin=0,
            #     ),
            #     caption=f"ground truth EEGs for sample 0",
            # )

        return outs

    def forward(self, batch):
        outs: Dict[str, torch.Tensor] = {}

        # parses eegs
        outs["eegs_gt"] = batch["eegs"].to(self.device)
        batch_size = outs["eegs_gt"].shape[0]
        assert not torch.isnan(outs["eegs_gt"]).any(), "there are nans in the eegs"
        self.shape_check(
            outs["eegs_gt"], (batch_size, self.eegs_channels, self.eegs_samples)
        )

        # parses fmris
        outs["fmris_gt"] = batch["fmris"].to(self.device)
        if self.fmris_downsampling_factor > 1:
            outs["fmris_gt"] = F.interpolate(outs["fmris_gt"].unsqueeze(1), scale_factor=1/self.fmris_downsampling_factor)[:,0]
        assert not torch.isnan(
            outs["fmris_gt"]
        ).any(), "there are nans in the input fmris"
        self.shape_check(outs["fmris_gt"], (batch_size, *self.fmris_shape))
        # if not self.normalized_fmris:
        # outs["fmris_gt"] = torch.log1p(outs["fmris_gt"])

        # generates the mel spectrogram of the ground truth eegs
        outs["eegs_mel_spectrogram_gt"] = self.waveform_to_mel_spectrogram(
            waveform=outs["eegs_gt"],
            mel_spectrogrammer=self.eegs_mel_spectrogrammer,
            expected_shape=(
                batch_size,
                self.eegs_channels,
                self.eegs_spectrogram_frequencies,
                self.eegs_spectrogram_samples,
            ),
        )
        assert not torch.isnan(
            outs["eegs_mel_spectrogram_gt"]
        ).any(), "there are nans in eegs_mel_spectrogram_gt"

        # encodes the eegs
        with profiler.record_function("eegs".upper()):
            outs["eegs_latent"] = self.encode_mel_spectrogram(
                mel_spectrogram=outs["eegs_mel_spectrogram_gt"],
                encoder=self.eegs_encoder,
                expected_shape=(
                    batch_size,
                    self.h_dim,
                ),
            )
            assert not torch.isnan(
                outs["eegs_latent"]
            ).any(), "there are nans in eegs_latent"

        # decodes the fmris
        with profiler.record_function("fmris".upper()):
            outs["fmris_rec"] = self.decode_mel_spectrogram(
                latent=outs["eegs_latent"],
                decoder=self.fmris_decoder,
                out_length=self.num_fmris_patches,
                expected_shape=(batch_size, *self.fmris_shape),
            )
            assert not torch.isnan(
                outs["fmris_rec"]
            ).any(), "there are nans in the generated fmris"

        # makes the tensor contiguous for stability purposes
        # for k, v in outs.items():
        #     if isinstance(v, torch.Tensor):
        #         outs[k] = v.contiguous()
        return outs

    def encode_mel_spectrogram(
        self,
        mel_spectrogram,
        encoder,
        expected_shape=None,
    ):
        mel_spectrogram = torch.log1p(mel_spectrogram)
        # mel_spectrogram = mel_spectrogram / mel_spectrogram.amax(dim=(1,2,3), keepdim=True)
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
        latent_tokens = sum([latent_tokens, pos_embeddings])
        latent_tokens_transformed = encoder.encoder(latent_tokens)[:, 0]
        # OPTIONAL: does a shape check
        if expected_shape:
            self.shape_check(
                tensor=latent_tokens_transformed,
                expected_shape=expected_shape,
            )
        return latent_tokens_transformed

    def discriminate(
        self,
        t,
        encoder,
        discriminator,
    ):
        outs: Dict[str, torch.Tensor] = {}
        batch_size = t.shape[0]
        # encodes the eegs
        with profiler.record_function("eegs".upper()):
            outs["encoded"] = self.encode_mel_spectrogram(
                mel_spectrogram=t.float(),
                encoder=encoder,
                expected_shape=(
                    batch_size,
                    self.h_dim,
                ),
            )
        # classifies
        outs["logits"] = discriminator.classifier(outs["encoded"])
        return outs

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
        waveform_scaled = torch.sign(waveform) * torch.log1p(torch.abs(waveform))
        mel_spectrogram = mel_spectrogrammer(waveform_scaled.float())
        if expected_shape:
            self.shape_check(
                tensor=mel_spectrogram,
                expected_shape=expected_shape,
            )
        return mel_spectrogram

    def discriminator_loss(self, logits, labels: int, smoothing=0.1):
        assert labels in {0, 1}
        batch_size = logits.shape[0]
        labels = torch.ones([batch_size, 1], device=logits.device) * labels
        if smoothing > 0:
            labels = labels + (smoothing - torch.rand_like(labels) * smoothing * 2)
        return F.binary_cross_entropy_with_logits(input=logits, target=labels)

    def reconstruction_loss(self, pred, gt, norm=1):
        if norm == 1:
            loss_fn = F.l1_loss
        elif norm == 2:
            loss_fn = F.mse_loss
        else:
            raise Exception(f"norm must be 1 or 2, got {norm}")
        return loss_fn(input=pred, target=gt)

    @staticmethod
    def mse(pred, gt):
        return F.mse_loss(
            input=pred.detach(),
            target=gt.detach(),
            reduction="mean",
        )
        
    @staticmethod
    def rmse(pred, gt):
        return torch.sqrt(F.mse_loss(
            input=pred.detach(),
            target=gt.detach(),
            reduction="mean",
        ))

    @staticmethod
    def mae(pred, gt):
        return torchmetrics.functional.regression.mean_absolute_error(
            preds=pred.detach(),
            target=gt.detach(),
        )

    @staticmethod
    def cfv(pred, gt):
        batch_size = pred.shape[0]
        return F.cosine_similarity(
            x1=pred.view(batch_size, -1).detach(),
            x2=gt.view(batch_size, -1).detach(),
        ).mean()

    @staticmethod
    def emv(pred, gt):
        batch_size = pred.shape[0]
        return torchmetrics.functional.pairwise_euclidean_distance(
            pred.view(batch_size, -1),
            gt.view(batch_size, -1),
            reduction="mean",
        )

    @staticmethod
    def psnr(pred, gt):
        return torchmetrics.functional.image.peak_signal_noise_ratio(
            preds=pred.detach(),
            target=gt.detach(),
        )

    @staticmethod
    def ssim(pred, gt):
        if pred.max() > 1:
            max_value = 1
        else:
            max_value = torch.max(
                pred.amax(dim=(1, 2, 3), keepdim=True),
                gt.amax(dim=(1, 2, 3), keepdim=True),
            )
            assert max_value.shape == (pred.shape[0], 1, 1, 1)
        return torchmetrics.functional.image.structural_similarity_index_measure(
            preds=pred / max_value,
            target=gt / max_value,
        )

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

    fmris_shape = (30, 64, 64)
    eegs_seconds = 2.16
    eegs_sampling_rate = 5000
    eegs_electrodes = [f"EL{i}" for i in range(64)]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NTViT(
        fmris_shape=fmris_shape,
        eegs_seconds=eegs_seconds,
        eegs_channels=len(eegs_electrodes),
        eegs_sampling_rate=eegs_sampling_rate,
        use_domain_matching=True,
        normalized_fmris=False,
    ).to(device)
    sample_batch = {
        "eegs": torch.randn(
            [8, model.eegs_channels, model.eegs_samples],
            device=model.device,
        ),
        "fmris": torch.rand(
            [8, *fmris_shape],
            device=model.device,
        ),
    }
    print(model)
    with profiler.profile(
        with_stack=False,
        profile_memory=True,
        use_cuda=True if device == "cuda" else False,
    ) as prof:
        # sample_outs = model.shared_step(sample_batch, 0)
        sample_outs = model.shared_step(sample_batch, 0)
    print(
        prof.key_averages().table(
            sort_by="cuda_time",
            row_limit=10,
        )
    )
